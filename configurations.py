import mujoco

class Constants:
    ROBOT_PATH = "./data/g1/scene.xml"
    LAFAN1_PATH = "./data/LAFAN1/g1"
    INTERPOLATED_PATH = "./data/LAFAN1_interpolated"

    LAFAN1_FPS = 30.0
    INTERPOLATE_FPS = 50.0

class TrackingConfig:

    """
    This class defines the anchor body and the body we want to track
    See III.A in BeyondMimic paper (short version) and the official implementaion
    https://github.com/HybridRobotics/whole_body_tracking/blob/main/source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/flat_env_cfg.py#L16

    Note that I reordered the list body_names compared to their code to make torso_link the first, just to make the index easier for extracting anchor pose.
    
    Note that in the xml file of G1 model, some joint is composed with multiple joints. 
    For example, the arm and hand is connected by left_wrist_roll_joint, left_wrist_pitch_joint and left_wrist_yaw_link
    we track the body connected to left_wrist_yaw_link, which is the hand (end-effector).
    """

    anchor_body_name = "torso_link"
    tracked_body_names = [
        "torso_link",

        "pelvis",
        "left_hip_roll_link",
        "left_knee_link",
        "left_ankle_roll_link", # end-effector
        "right_hip_roll_link",
        "right_knee_link",
        "right_ankle_roll_link", # end-effector
        "left_shoulder_roll_link",
        "left_elbow_link",
        "left_wrist_yaw_link", # end-effector
        "right_shoulder_roll_link",
        "right_elbow_link",
        "right_wrist_yaw_link" # end-effector
    ]
    end_effector_body_names = [
        "left_ankle_roll_link",
        "right_ankle_roll_link",
        "left_wrist_yaw_link",
        "right_wrist_yaw_link"
    ]

    """
    These body ids are obtained from the main function in this file
    """
    anchor_body_id = 16
    tracked_body_ids = [
        16, 
        1, 3, 5, 7, 9, 11, 13, 18, 20, 23, 25, 27, 30
    ]
    end_effector_body_ids = [7, 13, 23, 30]
    end_effector_body_idx = [4, 7, 10, 13] # the index of end-effectors in tracked bodies



class RobotConfig:
    # this contains the joint names of G1 robot 
    # and their corresponding index in each motion frame (and mujoco qpos)
    joint_names = [
        # pelvis position (3) + quaternion (4)
        # index [0:7]
        "floating_base_joint",

        # left leg (6)
        # index [7:13]
        "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", 
        "left_knee_joint", 
        "left_ankle_pitch_joint", "left_ankle_roll_joint",

        # right leg (6)
        # index [13:19]
        "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
        "right_knee_joint", 
        "right_ankle_pitch_joint", "right_ankle_roll_joint",

        # waist (3)
        # index [19:22]
        "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",

        # left arm (7)
        # index [22:29]
        "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
        "left_elbow_joint",
        "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",

        # right arm (7)
        # index [29:36]
        "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
        "right_elbow_joint", 
        "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint"
    ]

    """
    Kp/Kd computation from Sec III.C in BeyondMimic paper (short version) and the official implementaion
    https://github.com/HybridRobotics/whole_body_tracking/blob/main/source/whole_body_tracking/whole_body_tracking/robots/g1.py#L7
    """
    ARMATURE_5020 = 0.003609725
    ARMATURE_7520_14 = 0.010177520
    ARMATURE_7520_22 = 0.025101925
    ARMATURE_4010 = 0.00425

    NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 10Hz
    DAMPING_RATIO = 2.0

    STIFFNESS_5020 = ARMATURE_5020 * NATURAL_FREQ**2
    STIFFNESS_7520_14 = ARMATURE_7520_14 * NATURAL_FREQ**2
    STIFFNESS_7520_22 = ARMATURE_7520_22 * NATURAL_FREQ**2
    STIFFNESS_4010 = ARMATURE_4010 * NATURAL_FREQ**2

    DAMPING_5020 = 2.0 * DAMPING_RATIO * ARMATURE_5020 * NATURAL_FREQ
    DAMPING_7520_14 = 2.0 * DAMPING_RATIO * ARMATURE_7520_14 * NATURAL_FREQ
    DAMPING_7520_22 = 2.0 * DAMPING_RATIO * ARMATURE_7520_22 * NATURAL_FREQ
    DAMPING_4010 = 2.0 * DAMPING_RATIO * ARMATURE_4010 * NATURAL_FREQ

    
    JOINT_PARAMS = {
        "left_hip_pitch_joint": {
            "STIFFNESS": STIFFNESS_7520_14,
            "DAMPING": DAMPING_7520_14,
            "ARMATURE": ARMATURE_7520_14,
            "TORQUE_LIM": 88.0,
            "VEL_LIM": 32.0,
        },
        "left_hip_roll_joint": {
            "STIFFNESS": STIFFNESS_7520_22,
            "DAMPING": DAMPING_7520_22,
            "ARMATURE": ARMATURE_7520_22,
            "TORQUE_LIM": 139.0,
            "VEL_LIM": 20.0,
        },
        "left_hip_yaw_joint": {
            "STIFFNESS": STIFFNESS_7520_14,
            "DAMPING": DAMPING_7520_14,
            "ARMATURE": ARMATURE_7520_14,
            "TORQUE_LIM": 88.0,
            "VEL_LIM": 32.0,
        },
        "left_knee_joint": {
            "STIFFNESS": STIFFNESS_7520_22,
            "DAMPING": DAMPING_7520_22,
            "ARMATURE": ARMATURE_7520_22,
            "TORQUE_LIM": 139.0,
            "VEL_LIM": 20.0,
        },
        "left_ankle_pitch_joint": {
            "STIFFNESS": 2.0 * STIFFNESS_5020,
            "DAMPING": 2.0 * DAMPING_5020,
            "ARMATURE": 2.0 * ARMATURE_5020,
            "TORQUE_LIM": 50.0,
            "VEL_LIM": 37.0,
        },
        "left_ankle_roll_joint": {
            "STIFFNESS": 2.0 * STIFFNESS_5020,
            "DAMPING": 2.0 * DAMPING_5020,
            "ARMATURE": 2.0 * ARMATURE_5020,
            "TORQUE_LIM": 50.0,
            "VEL_LIM": 37.0,
        },
        "right_hip_pitch_joint": {
            "STIFFNESS": STIFFNESS_7520_14,
            "DAMPING": DAMPING_7520_14,
            "ARMATURE": ARMATURE_7520_14,
            "TORQUE_LIM": 88.0,
            "VEL_LIM": 32.0,
        },
        "right_hip_roll_joint": {
            "STIFFNESS": STIFFNESS_7520_22,
            "DAMPING": DAMPING_7520_22,
            "ARMATURE": ARMATURE_7520_22,
            "TORQUE_LIM": 139.0,
            "VEL_LIM": 20.0,
        },
        "right_hip_yaw_joint": {
            "STIFFNESS": STIFFNESS_7520_14,
            "DAMPING": DAMPING_7520_14,
            "ARMATURE": ARMATURE_7520_14,
            "TORQUE_LIM": 88.0,
            "VEL_LIM": 32.0,
        },
        "right_knee_joint": {
            "STIFFNESS": STIFFNESS_7520_22,
            "DAMPING": DAMPING_7520_22,
            "ARMATURE": ARMATURE_7520_22,
            "TORQUE_LIM": 139.0,
            "VEL_LIM": 20.0,
        },
        "right_ankle_pitch_joint": {
            "STIFFNESS": 2.0 * STIFFNESS_5020,
            "DAMPING": 2.0 * DAMPING_5020,
            "ARMATURE": 2.0 * ARMATURE_5020,
            "TORQUE_LIM": 50.0,
            "VEL_LIM": 37.0,
        },
        "right_ankle_roll_joint": {
            "STIFFNESS": 2.0 * STIFFNESS_5020,
            "DAMPING": 2.0 * DAMPING_5020,
            "ARMATURE": 2.0 * ARMATURE_5020,
            "TORQUE_LIM": 50.0,
            "VEL_LIM": 37.0,
        },
        "waist_yaw_joint": {
            "STIFFNESS": STIFFNESS_7520_14,
            "DAMPING": DAMPING_7520_14,
            "ARMATURE": ARMATURE_7520_14,
            "TORQUE_LIM": 88.0,
            "VEL_LIM": 32.0,
        },
        "waist_roll_joint": {
            "STIFFNESS": 2.0 * STIFFNESS_5020,
            "DAMPING": 2.0 * DAMPING_5020,
            "ARMATURE": 2.0 * ARMATURE_5020,
            "TORQUE_LIM": 50.0,
            "VEL_LIM": 37.0,
        },
        "waist_pitch_joint": {
            "STIFFNESS": 2.0 * STIFFNESS_5020,
            "DAMPING": 2.0 * DAMPING_5020,
            "ARMATURE": 2.0 * ARMATURE_5020,
            "TORQUE_LIM": 50.0,
            "VEL_LIM": 37.0,
        },
        "left_shoulder_pitch_joint": {
            "STIFFNESS": STIFFNESS_5020,
            "DAMPING": DAMPING_5020,
            "ARMATURE": ARMATURE_5020,
            "TORQUE_LIM": 25.0,
            "VEL_LIM": 37.0,
        },
        "left_shoulder_roll_joint": {
            "STIFFNESS": STIFFNESS_5020,
            "DAMPING": DAMPING_5020,
            "ARMATURE": ARMATURE_5020,
            "TORQUE_LIM": 25.0,
            "VEL_LIM": 37.0,
        },
        "left_shoulder_yaw_joint": {
            "STIFFNESS": STIFFNESS_5020,
            "DAMPING": DAMPING_5020,
            "ARMATURE": ARMATURE_5020,
            "TORQUE_LIM": 25.0,
            "VEL_LIM": 37.0,
        },
        "left_elbow_joint": {
            "STIFFNESS": STIFFNESS_5020,
            "DAMPING": DAMPING_5020,
            "ARMATURE": ARMATURE_5020,
            "TORQUE_LIM": 25.0,
            "VEL_LIM": 37.0,
        },
        "left_wrist_roll_joint": {
            "STIFFNESS": STIFFNESS_5020,
            "DAMPING": DAMPING_5020,
            "ARMATURE": ARMATURE_5020,
            "TORQUE_LIM": 25.0,
            "VEL_LIM": 37.0,
        },
        "left_wrist_pitch_joint": {
            "STIFFNESS": STIFFNESS_4010,
            "DAMPING": DAMPING_4010,
            "ARMATURE": ARMATURE_4010,
            "TORQUE_LIM": 5.0,
            "VEL_LIM": 22.0,
        },
        "left_wrist_yaw_joint": {
            "STIFFNESS": STIFFNESS_4010,
            "DAMPING": DAMPING_4010,
            "ARMATURE": ARMATURE_4010,
            "TORQUE_LIM": 5.0,
            "VEL_LIM": 22.0,
        },
        "right_shoulder_pitch_joint": {
            "STIFFNESS": STIFFNESS_5020,
            "DAMPING": DAMPING_5020,
            "ARMATURE": ARMATURE_5020,
            "TORQUE_LIM": 25.0,
            "VEL_LIM": 37.0,
        },
        "right_shoulder_roll_joint": {
            "STIFFNESS": STIFFNESS_5020,
            "DAMPING": DAMPING_5020,
            "ARMATURE": ARMATURE_5020,
            "TORQUE_LIM": 25.0,
            "VEL_LIM": 37.0,
        },
        "right_shoulder_yaw_joint": {
            "STIFFNESS": STIFFNESS_5020,
            "DAMPING": DAMPING_5020,
            "ARMATURE": ARMATURE_5020,
            "TORQUE_LIM": 25.0,
            "VEL_LIM": 37.0,
        },
        "right_elbow_joint": {
            "STIFFNESS": STIFFNESS_5020,
            "DAMPING": DAMPING_5020,
            "ARMATURE": ARMATURE_5020,
            "TORQUE_LIM": 25.0,
            "VEL_LIM": 37.0,
        },
        "right_wrist_roll_joint": {
            "STIFFNESS": STIFFNESS_5020,
            "DAMPING": DAMPING_5020,
            "ARMATURE": ARMATURE_5020,
            "TORQUE_LIM": 25.0,
            "VEL_LIM": 37.0,
        },
        "right_wrist_pitch_joint": {
            "STIFFNESS": STIFFNESS_4010,
            "DAMPING": DAMPING_4010,
            "ARMATURE": ARMATURE_4010,
            "TORQUE_LIM": 5.0,
            "VEL_LIM": 22.0,
        },
        "right_wrist_yaw_joint": {
            "STIFFNESS": STIFFNESS_4010,
            "DAMPING": DAMPING_4010,
            "ARMATURE": ARMATURE_4010,
            "TORQUE_LIM": 5.0,
            "VEL_LIM": 22.0,
        },
    }


if __name__ == '__main__':
    
    """
    Get the body ids in TrackingConfig. Only run once
    """
    import mujoco
    model = mujoco.MjModel.from_xml_path(Constants.ROBOT_PATH)
    data = mujoco.MjData(model)


    print("--------------- Tracked Bodies ---------------")
    for body_name in TrackingConfig.tracked_body_names:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        print(body_name, body_id)

    print("--------------- End-Effectors ---------------")
    for body_name in TrackingConfig.end_effector_body_names:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        print(body_name, body_id)
