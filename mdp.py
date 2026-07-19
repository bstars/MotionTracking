import numpy as np
import gymnasium as gym
from gymnasium import Env, spaces
import mujoco
import os
from scipy.io import loadmat
from scipy.spatial.transform import Rotation as R

from configurations import Constants, TrackingConfig, RobotConfig

class TrackMDP():
    """
    Throughout this class
        ref_xxx stands for referemce (from reference motion clip)
        rob_xxx stands for robot (from current simulation)
        tar_xxx stands for target
    """
    def __init__(self, ref_q_hist, ref_qdot_hist, ref_poses_hist , ref_vels_hist, actor_obs_noise=True, train=True, max_time=10) -> None:

        #
        # Reference Motion
        #
        #   ref_q_hist [T, 36]
        #   ref_q_hist[:,:3] is the position of free joint
        #   ref_q_hist[:,3:7] is the rotation (in quaternion, wxyz convention) of free joint
        #   ref_q_hist[:,7:] is the joint angles of 29-dof humanoid
        #
        #   ref_qdot_hist [T, 35]
        #   ref_qdot_hist[:,:3] is the linear velocity (in world frame) of free joint
        #   ref_qdot_hist[:,3:6] is the angualr velocity (in body frame) of free joint
        #   ref_qdot_hist[:,6:] is the joint velocity of 29-dof humanoid
        #
        #   ref_poses_hist [T, num_tracked_body, 7]
        #   ref_poses_hist[:,i,:3] is the position (world frame) of ith body
        #   ref_poses_hist[:,i,3:7] is the quaternion (world frame) of ith body
        #   ref_poses_hist[:,0,:] is the pose of anchor body

        #   ref_vels_hist [T, num_tracked_body, 6]
        #   ref_vels_hist[:,i,:3] is the linear velocity (body frame) of ith body
        #   ref_vels_hist[:,i,3:6] is the angulat velocity (body frame) of ith body
        #   ref_vels_hist[:,0,:] is the velocity of anchor body

        self.ref_q_hist = ref_q_hist
        self.ref_qdot_hist = ref_qdot_hist
        self.ref_poses_hist = ref_poses_hist 
        self.ref_vels_hist = ref_vels_hist
        self.ref_idx = 0


        #
        # Mujoco Model
        #
        self.model = mujoco.MjModel.from_xml_path(Constants.ROBOT_PATH)
        self.data = mujoco.MjData(self.model)
        self.home_qpos = self.model.key_qpos[0].copy()
        self.dt = 1./Constants.INTERPOLATE_FPS
        self.sim_step = int(self.dt / self.model.opt.timestep)
        self.max_episode_steps = int(max_time * Constants.INTERPOLATE_FPS) if train else np.inf# max_time_second

        joint_lim_lower = []
        joint_lim_upper = []
        for jname in RobotConfig.joint_names[1:]:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            lower, upper = self.model.jnt_range[jid]
            joint_lim_lower.append(lower)
            joint_lim_upper.append(upper)
        self.joint_lim_lower = np.array(joint_lim_lower)
        self.joint_lim_upper = np.array(joint_lim_upper)


        joint_mid = (self.joint_lim_upper + self.joint_lim_lower) / 2
        half = 0.5 * (self.joint_lim_upper - self.joint_lim_lower)
        self.soft_joint_lim_lower = joint_mid - 0.9 * half
        self.soft_joint_lim_upper = joint_mid + 0.9 * half

        floor_geom_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_GEOM,
            "floor",
        )
        self.floor_body_id = self.model.geom_bodyid[floor_geom_id]

        #
        # Action Space
        #
        self.action_scale = np.array([
            RobotConfig.JOINT_PARAMS[joint_name]["TORQUE_LIM"]/RobotConfig.JOINT_PARAMS[joint_name]["STIFFNESS"] 
            for joint_name in RobotConfig.joint_names[1:]
        ]) * 0.25
        self.action_space = spaces.Box(-1, 1, shape=(self.model.nu,), dtype=np.float32)
        self.previous_action = np.zeros([self.model.nu,])

        for joint_name in RobotConfig.joint_names[1:]:
            joint_id = mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_JOINT,
                joint_name,
            )
            dof_id = self.model.jnt_dofadr[joint_id]

            actuator_id = mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_ACTUATOR,
                joint_name,
            )

            kp = RobotConfig.JOINT_PARAMS[joint_name]["STIFFNESS"]
            kd = RobotConfig.JOINT_PARAMS[joint_name]["DAMPING"]
            armature = RobotConfig.JOINT_PARAMS[joint_name]["ARMATURE"]

            self.model.dof_armature[dof_id] = armature

            self.model.actuator_gainprm[actuator_id, 0] = kp
            self.model.actuator_biasprm[actuator_id, 1] = -kp
            self.model.actuator_biasprm[actuator_id, 2] = -kd

        #
        # State space
        #
        self.actor_state_dim = 160
        self.critic_state_dim = 286

        #
        # Other stuff
        #
        self.gravity = np.array([0., 0., -1.])
        self.num_step = 0
        self.actor_obs_noise = actor_obs_noise
        self.train = train


    def get_obs(self):
        """
        :return a_obs: np.array, [160,]
            the observation of actor, containing:

            1. ref_q: reference joint position, [29,]
            2. ref_qdot: reference joint velocity [29,]
            3. rob_q: robot joint position, [29,]
            4. rob_qdot: robot joint velocity, [29,]
            5. previous_action: previous action, [29,]
            6. rel_anchor_pos_b: 
                reference anchor position relative to current anchor position, [3,]
                expressed in body frame
            7. rel_anchor_ori_b_mat:
                reference anchor orientation relative to current anchor orientation, [6,]
                first 2 columns of the rotation matrix
                expressed in body frame
            8. rob_anchor_lin_vel_b: linear velocity of anchor in body frame, [3,]
            9. rob_anchor_ang_vel_b: angular velocity of anchor in body frame, [3,]
            Total dimension is 29*5+3+6+3+3=160


        :return c_obs: np.array, [286,]
            the observation of critic, containing:

            1. all actor observation, [160],
            2. tracked body pose relative to anchor body, [9*14], where 14 is the number of tracked body
        """

        mujoco.mj_forward(self.model, self.data)

        rob_anchor_pos_w = self.data.xpos[TrackingConfig.anchor_body_id].copy()
        rob_anchor_ori_w = R.from_quat(self.data.xquat[TrackingConfig.anchor_body_id].copy(), scalar_first=True)
        rob_anchor_ori_w_inv = rob_anchor_ori_w.inv()

        ref_anchor_pos_w = self.ref_poses_hist[self.ref_idx, 0, :3]
        ref_anchor_ori_w = R.from_quat(self.ref_poses_hist[self.ref_idx, 0, 3:7], scalar_first=True)


        rob_anchor_vel_b = np.zeros([6])
        mujoco.mj_objectVelocity(
            self.model,
            self.data,
            mujoco.mjtObj.mjOBJ_BODY,
            TrackingConfig.anchor_body_id,
            rob_anchor_vel_b,
            1,
        )


        # Actor obs: reference joint pos and vel
        ref_q = self.ref_q_hist[self.ref_idx, 7:] - self.home_qpos[7:] # [29,]
        ref_qdot = self.ref_qdot_hist[self.ref_idx, 6:] # [29,]

        # Actor obs: robot joint pos and vel
        rob_q = self.data.qpos[7:] - self.home_qpos[7:] # [29,]
        rob_qdot = self.data.qvel[6:] # [29,]

        # Actor obs: previous action
        previous_action = self.previous_action # [29,]

        # Actor obs: reference anchor postiion relative to current anchor position
        # expressed in body frame
        rel_anchor_pos_b = rob_anchor_ori_w_inv.apply((ref_anchor_pos_w - rob_anchor_pos_w)) # [3,]

        # Actor obs: reference anchor orientation relative to current anchor orientation
        # expressed in body frame
        rel_anchor_ori_b = rob_anchor_ori_w_inv * ref_anchor_ori_w
        rel_anchor_ori_b_mat = rel_anchor_ori_b.as_matrix()[:,:2].reshape(-1) # [6,]

        # Actor obs: robot anchor linear velocity
        # expressed in body frame
        rob_anchor_lin_vel_b = rob_anchor_vel_b[3:6] # [3,]

        # Actor obs: robot anchor angular velocity
        # expressed in body frame
        rob_anchor_ang_vel_b = rob_anchor_vel_b[0:3] # [3,]


        # Critic obs: position and orientation of tracked body relative to anchor body
        # expressed in body frame
        rel_pose_track_b = []
        for bid in TrackingConfig.tracked_body_ids:
            body_pos_w = self.data.xpos[bid]
            body_ori_w = R.from_quat(self.data.xquat[bid].copy(), scalar_first=True)
            rel_pos = rob_anchor_ori_w_inv.apply(body_pos_w - rob_anchor_pos_w) # [3,]
            rel_ori = (rob_anchor_ori_w_inv * body_ori_w).as_matrix()[:,:2].reshape(-1)
            rel_pose_track_b.append(rel_pos)
            rel_pose_track_b.append(rel_ori)
        rel_pose_track_b = np.concatenate(rel_pose_track_b) # [126,]
        
        a_obs = np.concatenate([
            ref_q, 
            ref_qdot, 
            rob_q + np.random.uniform(low=-0.01, high=0.01, size=[self.model.nu]) * self.actor_obs_noise, 
            rob_qdot + np.random.uniform(low=-0.5, high=0.5, size=[self.model.nu]) * self.actor_obs_noise,
            previous_action,
            rel_anchor_pos_b + np.random.uniform(low=-0.25, high=0.25, size=[3,]) * self.actor_obs_noise, 
            rel_anchor_ori_b_mat + np.random.uniform(low=-0.05, high=0.05, size=[6,]) * self.actor_obs_noise,
            rob_anchor_lin_vel_b + np.random.uniform(low=-0.5, high=0.5, size=[3,]) * self.actor_obs_noise, 
            rob_anchor_ang_vel_b + np.random.uniform(low=-0.2, high=0.2, size=[3,]) * self.actor_obs_noise
        ]) # [160,]

        c_obs = np.concatenate([
            ref_q, 
            ref_qdot, 
            rob_q, 
            rob_qdot,
            previous_action,
            rel_anchor_pos_b, 
            rel_anchor_ori_b_mat,
            rob_anchor_lin_vel_b, 
            rob_anchor_ang_vel_b,
            rel_pose_track_b
        ]) # [286,]
        return a_obs, c_obs


    def reset(self):
        """
        TODO: 
            1. Adaptive sampling for starting frame
            2. Domain randomization
        """

        # Adaptive sampling
        if self.train:
            self.ref_idx = np.random.randint(0, len(self.ref_q_hist)-2)
        else:
            self.ref_idx = 0


        root_pos_noise = np.random.uniform(
            low=np.array([-0.05, -0.05, -0.02]),
            high=np.array([0.05, 0.05, 0.02]),
        )
        root_rot_noise = R.from_euler(
            "xyz",
            np.random.uniform(
                low=np.array([-0.05, -0.05, -0.05]),
                high=np.array([0.05, 0.05, 0.05]),
            ),
        )
        root_lin_vel_noise = np.random.uniform(
            low=np.array([-0.10, -0.10, -0.05]),
            high=np.array([0.10, 0.10, 0.05]),
        )
        root_ang_vel_noise = np.random.uniform(
            low=np.array([-0.10, -0.10, -0.10]),
            high=np.array([0.10, 0.10, 0.10]),
        )
        joint_pos_noise = np.random.uniform(
            low=-0.01,
            high=0.01,
            size=[self.model.nu],
        )

        ref_root_ori = R.from_quat(self.ref_q_hist[self.ref_idx, 3:7], scalar_first=True)
        noisy_root_ori = root_rot_noise * ref_root_ori


        self.data.qpos[0:3] = self.ref_q_hist[self.ref_idx, 0:3] + root_pos_noise
        self.data.qpos[3:7] = noisy_root_ori.as_quat(scalar_first=True)
        self.data.qpos[7:] = np.clip(
            self.ref_q_hist[self.ref_idx, 7:] + joint_pos_noise,
            self.soft_joint_lim_lower,
            self.soft_joint_lim_upper,
        )

        self.data.qvel[0:3] = self.ref_qdot_hist[self.ref_idx, 0:3] + root_lin_vel_noise
        self.data.qvel[3:6] = self.ref_qdot_hist[self.ref_idx, 3:6] + root_ang_vel_noise
        self.data.qvel[6:] = self.ref_qdot_hist[self.ref_idx, 6:]

        self.data.ctrl[:] = self.ref_q_hist[self.ref_idx, 7:]

        mujoco.mj_forward(self.model, self.data)
        self.previous_action = (self.ref_q_hist[self.ref_idx, 7:] - self.home_qpos[7:]) / self.action_scale
        self.num_step = 0

        return self.get_obs()

    def anchor_transform(self, ref_pose, rob_anchor_pos_w, rob_anchor_quat_w):
        """
        The anchor transform described in Sec III.A of the short version of BeyondMimic
        Compute the target pose of tracked bodies given the current robot anchor pose

        :param ref_pose: np.array, [num_tracked_body, 7]
            ref_pose[i,:3] is the position (world frame) of ith body
            ref_pose[i,3:7] is the quaternion (world frame) of ith body
            ref_pose[0] is the reference pose of the anchor body

        :param rob_anchor_pos_w: np.array, [3,]
            position of current robot anchor
            expressed in world frame

        :param rob_anchor_pos_w: np.array, [4,]
            orientation of current robot anchor
            expressed as quaternion in world frame

        :return tar_pose: np.array, [num_tracked_body, 7]
        """

        ref_anchor_pos_w = ref_pose[0, :3]
        ref_anchor_ori_w = R.from_quat(ref_pose[0, 3:7], scalar_first=True)
        rob_anchor_pos_w = rob_anchor_pos_w
        rob_anchor_ori_w = R.from_quat(rob_anchor_quat_w, scalar_first=True)


        rel_anchor_ori = rob_anchor_ori_w * ref_anchor_ori_w.inv()
        yaw = R.as_euler(rel_anchor_ori, "zyx")[0]
        yaw_ori = R.from_euler("z", yaw)


        tar_anchor_pos_w = np.array([
            rob_anchor_pos_w[0], rob_anchor_pos_w[1], ref_anchor_pos_w[2]
        ])

        tar_pose = []
        for i, pose in enumerate(ref_pose):
            pos, quat = pose[:3], pose[3:7]
            ori = R.from_quat(quat, scalar_first=True)

            tar_pos_w = tar_anchor_pos_w + yaw_ori.apply( pos - ref_anchor_pos_w )
            tar_ori_w = yaw_ori * ori
            target_quat_w = tar_ori_w.as_quat(scalar_first=True)

            tar_pose.append(np.concatenate([tar_pos_w, target_quat_w]))

        return np.stack(tar_pose)


    def get_reward(self, action):
        
        ref_anchor_pos_w = self.ref_poses_hist[self.ref_idx, 0, :3]
        ref_anchor_ori_w = R.from_quat(self.ref_poses_hist[self.ref_idx, 0, 3:7], scalar_first=True)
        ref_anchor_ori_w_inv = ref_anchor_ori_w.inv()

        rob_anchor_pos_w = self.data.xpos[TrackingConfig.anchor_body_id].copy()
        rob_anchor_ori_w = R.from_quat(self.data.xquat[TrackingConfig.anchor_body_id].copy(), scalar_first=True)
        rob_anchor_ori_w_inv = rob_anchor_ori_w.inv()

        tar_pose_w = self.anchor_transform(
            ref_pose=self.ref_poses_hist[self.ref_idx],
            rob_anchor_pos_w=self.data.xpos[TrackingConfig.anchor_body_id].copy(),
            rob_anchor_quat_w=self.data.xquat[TrackingConfig.anchor_body_id].copy()
        )

        #
        # Check termination: height of robot and reference anchor
        #
        bad_anchor_z = np.abs(ref_anchor_pos_w[-1] - rob_anchor_pos_w[-1]) > 0.25

        #
        # Check termination: projected gravity 
        #
        bad_anchor_ori = np.abs(
            ref_anchor_ori_w_inv.apply(self.gravity)[-1] - rob_anchor_ori_w_inv.apply(self.gravity)[-1]
        ) > 0.8

        #
        # Check termination: height of end effectors
        #
        bad_ee_z = False
        for idx, bid in zip(TrackingConfig.end_effector_body_idx, TrackingConfig.end_effector_body_ids):
            z1 = tar_pose_w[idx][2]
            z2 = self.data.xpos[bid].copy()[2]
        
            if np.abs(z1-z2) > 0.25:
                bad_ee_z = True
                break
        
        #
        # Termination
        #
        if bad_anchor_z or bad_anchor_ori or bad_ee_z:
            return True, 0


        #
        # Tracking reward
        # 
        pos_err = []
        ori_err = []
        lin_vel_err = []
        ang_vel_err = []
        vel_holder = np.zeros([6])

        for idx, bid in enumerate(TrackingConfig.tracked_body_ids):

            # position
            pos_err.append(
                np.sum(
                    (tar_pose_w[idx, :3] - self.data.xpos[bid]) ** 2
                )
            )

            # orientation
            body_ori = R.from_quat(self.data.xquat[bid].copy(), scalar_first=True)
            tar_ori = R.from_quat(tar_pose_w[idx, 3:7], scalar_first=True)
            ori_err.append(
                np.sum(
                    R.as_rotvec(body_ori.inv() * tar_ori) ** 2
                )
            )

            # linear and angular velocity
            mujoco.mj_objectVelocity(
                self.model,
                self.data,
                mujoco.mjtObj.mjOBJ_BODY,
                bid,
                vel_holder,
                0,  # world frame velocity
            )
            lin_vel_err.append(
                np.sum(
                    (self.ref_vels_hist[self.ref_idx, idx, 0:3] - vel_holder[3:6])**2
                )
            )

            ang_vel_err.append(
                np.sum(
                    (self.ref_vels_hist[self.ref_idx, idx, 3:6] - vel_holder[0:3])**2
                )
            )

        r_track = (
            np.exp(-np.mean(pos_err) / 0.05**2)
            + np.exp(-np.mean(ori_err) / 0.25**2)
            + np.exp(-np.mean(lin_vel_err) / 0.3**2)
            + np.exp(-np.mean(ang_vel_err) / (0.5*np.pi)**2)
        )

        # print("------------------------------------")
        # print(np.mean(pos_err), np.exp(-np.mean(pos_err) / 0.05**2))
        # print(np.mean(ori_err), np.exp(-np.mean(ori_err) / 0.25**2))
        # print(np.mean(lin_vel_err), np.exp(-np.mean(lin_vel_err) / 0.3**2))
        # print(np.mean(ang_vel_err), np.exp(-np.mean(ang_vel_err) / (np.pi*0.5)**2))
        # print("------------------------------------")

        
        r_track += 0.5 * np.exp( 
            np.sum(-(ref_anchor_pos_w - rob_anchor_pos_w)**2 / 0.3**2)
        )

        r_track += 0.5 * np.exp( 
            np.sum(
                -(
                    R.as_rotvec(rob_anchor_ori_w_inv * ref_anchor_ori_w)
                )**2 / 0.25**2
            )
        )

        #
        # Action reward (penalty)
        #
        r_action = -0.05 * np.sum((action - self.previous_action)**2)


        #
        # Soft joint limit reward (penalty)
        #
        q = self.data.qpos[7:]
        r_joint_lim = -1 * np.sum(
            np.maximum(self.soft_joint_lim_lower - q, 0.)
            + 
            np.maximum(q - self.soft_joint_lim_upper, 0.)
        )

    
        #
        # Contact reward (penalty)
        # Ignored for now.
        #

        reward = r_track + r_action + r_joint_lim
        return False, reward * 2 * self.dt


    def step(self, action):
        # Note that even though i set self.action_space to be [-1,1], clipping is not applied.
        # Thus we can apply action beyond [-1,1]
        # action = np.clip(action, -1., 1.)
        q_target = self.home_qpos[7:] + self.action_scale * action
        self.data.ctrl[:] = q_target
        mujoco.mj_step(self.model, self.data, self.sim_step)

        terminated, reward = self.get_reward(action)
        truncated = False

        if self.num_step >= self.max_episode_steps or self.ref_idx >= len(self.ref_q_hist)-2:
            truncated = True

        self.previous_action = action.copy()
        self.ref_idx += 1
        self.num_step += 1


        a_obs, c_obs = self.get_obs()
        return a_obs, c_obs, reward, terminated, truncated


if __name__ == '__main__':
    motion = "dance1_subject2"
    path = os.path.join(Constants.INTERPOLATED_PATH, motion+".mat")
    mat = loadmat(path)
    ref_q_hist = mat["q_hist"]
    ref_qdot_hist = mat["qdot_hist"]
    ref_poses_hist = mat["poses_hist"]
    ref_vels_hist = mat["vels_hist"]


    env = TrackMDP(
        ref_q_hist, ref_qdot_hist, ref_poses_hist, ref_vels_hist
    )

    a_obs, c_obs = env.reset()
    print(a_obs.shape, c_obs.shape)
    print(env.action_scale)


    yaw = R.from_euler("zyx", [0.63, 0, 0.])
    frame_idx = 10

    # actual pose
    env.data.qpos[:] = env.ref_q_hist[frame_idx]
    quat = R.from_quat(env.data.qpos[3:7].copy(), scalar_first=True)
    env.data.qpos[3:7] = (yaw * quat).as_quat(scalar_first=True)
    env.data.qpos[0:3] += [0.1, 0.1, 0.0]
    mujoco.mj_forward(env.model, env.data)

    # desired pose
    desired_pose = env.anchor_transform(
        env.ref_poses_hist[frame_idx], 
        env.data.xpos[TrackingConfig.anchor_body_id],
        env.data.xquat[TrackingConfig.anchor_body_id]
    )
    print(desired_pose.shape)

    for i, bid in enumerate(TrackingConfig.tracked_body_ids):

        pos_mj = env.data.xpos[bid].copy()
        quat_mj = env.data.xquat[bid].copy()

        pos_at = desired_pose[i, :3]
        quat_at = desired_pose[i, 3:7]

        # a sanity check of the anchor transform, the difference should be small
        print(
            np.linalg.norm(pos_mj - pos_at),
            np.linalg.norm(quat_mj - quat_at)
        )


    # print([RobotConfig.JOINT_PARAMS[joint_name]["STIFFNESS"] for joint_name in RobotConfig.joint_names[1:]])
    # print([RobotConfig.JOINT_PARAMS[joint_name]["DAMPING"] for joint_name in RobotConfig.joint_names[1:]])

