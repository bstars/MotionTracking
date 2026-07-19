import numpy as np
import mujoco
import mujoco.viewer
import time
import os
from scipy.io import savemat, loadmat
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R, Slerp
import imageio.v2 as imageio


from configurations import Constants, RobotConfig, TrackingConfig

def visualize_frame(q_hist, fps, quaternion_convention="wxyz", return_gif=False):
    """
    :param q_hist: np.array, [num_frame, 36]
    :param fps: int
    """

    dt = 1.0 / fps
    model = mujoco.MjModel.from_xml_path(Constants.ROBOT_PATH)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=480, width=640)
    gif = []


    with mujoco.viewer.launch_passive(model, data) as viewer:
        for idx in range(len(q_hist)):
            frame = q_hist[idx]
            quat = frame[3:7]
            data.qpos[0:3] = frame[0:3] # anchor pos
            if quaternion_convention == "wxyz":
                data.qpos[3:7] = quat
            elif quaternion_convention == "xyzw":
                data.qpos[3:7] = [quat[3], quat[0], quat[1], quat[2]]
            else:
                raise AttributeError
            data.qpos[7:] = frame[7:] # joints

            mujoco.mj_forward(model, data)

            viewer.cam.lookat[:] = data.qpos[0:3]
            viewer.sync()

            if return_gif:
                renderer.update_scene(data, camera=viewer.cam)
                rgb_image = renderer.render()
                gif.append(rgb_image)

            time.sleep(dt)
    if return_gif:
        return np.stack(gif)

    
    


def interpolate_frames(input_q_hist, input_fps:int, output_fps:int):
    """
    Interpolate a motion clip to a higher fps.
    Note that input_frames (LAFAN1 dataset) use [x,y,z,w] convention for quaternion.
    But the output use [w,x,y,z] convention for quaternion
    
    :param input_q_hist: np.array, [num_input_frames, 36]
    :param input_fps: int
    :param output_fps: int
    :return output_q_hist: np.array, [num_output_frames, 36]
    :return output_qdot_hist: np.array, [num_output_frames, 35]
    """

    input_dt = 1.0 / input_fps
    output_dt = 1.0 / output_fps

    duration = (len(input_q_hist) - 1) * input_dt
    input_ts = np.arange(0, duration, input_dt)
    output_ts = np.arange(0, duration, output_dt)

    output_q_hist = []
    output_qdot_hist = []

    for i in range(0, len(output_ts)):
        
        # find the neighbor frames
        phase = output_ts[i] / duration + 1e-7
        index_0 = np.floor( (phase * (len(input_q_hist)-1)) ).astype(np.long)
        index_1 = min( index_0 + 1, len(input_q_hist) - 1 )
        dt = output_ts[i] - input_ts[index_0]

        # extract anchor position, anchor quaternion and joint angles
        # the data use quaternion convention [x,y,z,w], but we use [w, x, y, z]
        frame_0 = input_q_hist[index_0]
        frame_1 = input_q_hist[index_1]
        p0 = frame_0[:3]; p1 = frame_1[:3]
        R0 = R.from_quat(frame_0[3:7]); R1 = R.from_quat(frame_1[3:7])
        j0 = frame_0[7:]; j1 = frame_1[7:]

        # compute linear velocitys of anchor and joint angles
        # then linear interpolate
        p_vel = (p1 - p0) / input_dt; p = p0 + p_vel * dt
        j_vel = (j1 - j0) / input_dt; j = j0 + j_vel * dt

        # compute angular velocity of anchor and interpolate using quaternion
        R_delta = R0.inv() * R1
        rotvec = R_delta.as_rotvec()
        angular_vel = rotvec / input_dt
        R_interp = R0 * R.from_rotvec(angular_vel * dt)
        q = R_interp.as_quat(scalar_first=True)

        # record the interpolation and velocity
        output_q_hist.append(
            np.concatenate([p, q, j])
        )
        output_qdot_hist.append(
            np.concatenate([p_vel, angular_vel, j_vel])
        )
        

    return np.stack(output_q_hist), np.stack(output_qdot_hist)


def extract_from_mujoco(model, data):
    """
    Extract the body poses of the frame in data

    :return poses: np.array, [num_tracked_body, 7]
        Every 7 elements express the position and quaternion (in world frame) of a body

    :return vels: np.array, [num_tracked_body, 6]
        Every 6 elements express the linear velocity (in body frame) and angular velocity (in body frame) of a body
    """
    poses = []
    vels = []

    vel_holder = np.zeros([6])

    # poses for bodies
    for body_id in TrackingConfig.tracked_body_ids:
        body_pos = data.xpos[body_id].copy()
        body_quat = data.xquat[body_id].copy()

        mujoco.mj_objectVelocity(
            model,
            data,
            mujoco.mjtObj.mjOBJ_BODY,
            body_id,
            vel_holder,
            0, # world-frame velocity
        )
        poses.append(np.concatenate([body_pos, body_quat]))
        vels.append(np.concatenate([vel_holder[3:6], vel_holder[0:3]]))

    poses= np.stack(poses)
    vels = np.stack(vels)
    return poses, vels



def extract_from_frames(q_hist, qdot_hist):
    """
    For each frame, extract the position (world frame), quaternion (world frame), linear velocity (body frame), angular velocity (body frame) for each tracked body
    This will be used later for computing anchor transform and rewards 
    
    :param q_hist: np.array, [num_frames, 36]
    :param qdot_hist: np.array, [num_frames, 35]

    :return poses: np.array, [num_frames, num_tracked_body, 7]
    :return vels: np.array, [num_frames, num_tracked_body, 6]
    """

    model = mujoco.MjModel.from_xml_path(Constants.ROBOT_PATH)
    data = mujoco.MjData(model)

    poses_hist = []
    vels_hist = []

    for qpos, qvel in zip(q_hist, qdot_hist):

        data.qpos[:] = qpos
        data.qvel[:] = qvel

        # compute forward kinematics using mujoco
        mujoco.mj_forward(model, data)

        poses, vels = extract_from_mujoco(model, data)
        poses_hist.append(poses)
        vels_hist.append(vels)

    return np.stack(poses_hist), np.stack(vels_hist)

if __name__ == '__main__':
    """
    visualize an original motion clip
    """
    # motion_file = os.path.join(Constants.LAFAN1_PATH, "walk1_subject1.csv")
    # q_hist = np.genfromtxt(motion_file, delimiter=",")[:200]
    # print(q_hist.shape)
    # visualize_frame(q_hist, fps=Constants.LAFAN1_FPS, quaternion_convention="xyzw")


    """
    visualize an interpolated motion clip
    """
    # motion_file = os.path.join(Constants.LAFAN1_PATH, "walk1_subject1.csv")
    # q_hist = np.genfromtxt(motion_file, delimiter=",")[:200]
    # interpolated_q_hist, interpolated_qdot_hist = interpolate_frames(q_hist, Constants.LAFAN1_FPS, Constants.INTERPOLATE_FPS)
    # print(interpolated_q_hist.shape, interpolated_qdot_hist.shape)
    # visualize_frame(interpolated_q_hist, fps=Constants.INTERPOLATE_FPS, quaternion_convention="wxyz")


    """
    extract poses and velocities from interpolated motion clip
    """
    # motion_file = os.path.join(Constants.LAFAN1_PATH, "walk1_subject1.csv")
    # q_hist = np.genfromtxt(motion_file, delimiter=",")[:200]
    # interpolated_q_hist, interpolated_qdot_hist = interpolate_frames(q_hist, Constants.LAFAN1_FPS, Constants.INTERPOLATE_FPS)
    # interpolated_poses_hist, interpolated_vels_hist = extract_from_frames(interpolated_q_hist, interpolated_qdot_hist)
    # print(interpolated_q_hist.shape, interpolated_qdot_hist.shape)
    # print(interpolated_poses_hist.shape, interpolated_vels_hist.shape)

    """
    Interpolate all motions
    """
    # os.makedirs(Constants.INTERPOLATED_PATH, exist_ok=True)
    # folder = Constants.LAFAN1_PATH
    # for name in tqdm(os.listdir(folder), desc="Interpolating motions"):
    #     path = os.path.join(folder, name)
    #     if os.path.isfile(path):

    #         input_q_hist = np.genfromtxt(path, delimiter=",")
    #         interpolated_q_hist, interpolated_qdot_hist = interpolate_frames(input_q_hist, Constants.LAFAN1_FPS, Constants.INTERPOLATE_FPS)

    #         interpolated_poses_hist, interpolated_vels_hist = extract_from_frames(interpolated_q_hist, interpolated_qdot_hist)
    
    #         savemat(
    #             os.path.join(Constants.INTERPOLATED_PATH, name)[:-4] + ".mat",
    #             {
    #                 "q_hist":interpolated_q_hist,
    #                 "qdot_hist":interpolated_qdot_hist,
    #                 "poses_hist":interpolated_poses_hist,
    #                 "vels_hist":interpolated_vels_hist
    #             }
    #         )

    #         print( os.path.join(Constants.INTERPOLATED_PATH, name)[:-4] + ".mat" )

    

    # motion = "fight1_subject2"
    motion = "walk1_subject1"
    interpolated = True
    max_len = 500
    save_gif = True
    if interpolated:
        path = os.path.join(Constants.INTERPOLATED_PATH, motion+".mat")
        mat = loadmat(path)
        q_hist, qdot_hist, poses_hist, vels_hist = mat["q_hist"], mat["qdot_hist"], mat["poses_hist"], mat["vels_hist"]
        print(q_hist.shape, qdot_hist.shape, poses_hist.shape, vels_hist.shape)
        fps = Constants.INTERPOLATE_FPS
        quaternion_convention="wxyz"

    else:
        path = os.path.join(Constants.LAFAN1_PATH, motion+".csv")
        q_hist = np.genfromtxt(path, delimiter=",")
        fps = Constants.LAFAN1_FPS
        quaternion_convention="xyzw"

    gif = visualize_frame(q_hist[:max_len], fps, quaternion_convention, return_gif=save_gif)
    if save_gif:
        imageio.mimsave(
            motion + "_target.gif",
            gif[::3], # only for visualization
            duration=1 / Constants.INTERPOLATE_FPS * 5 * 1000,
            loop=0
        )
