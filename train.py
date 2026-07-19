import numpy as np
import sys
import gymnasium as gym
from gymnasium import Env, spaces
import mujoco
import os
from scipy.io import loadmat, savemat
from scipy.spatial.transform import Rotation as R
import torch
import time
import imageio.v2 as imageio

from configurations import Constants, TrackingConfig, RobotConfig
from mdp import TrackMDP
from ppo import PPO_Clip


actor_dim=[512, 128, 64]
critic_dim=[512, 128, 64]


def train(motion, max_len, ent_coef, steps, from_last=False):
    path = os.path.join(Constants.INTERPOLATED_PATH, motion+".mat")
    mat = loadmat(path)
    ref_q_hist = mat["q_hist"][:max_len]
    ref_qdot_hist = mat["qdot_hist"][:max_len]
    ref_poses_hist = mat["poses_hist"][:max_len]
    ref_vels_hist = mat["vels_hist"][:max_len]


    tic = time.time()
    envs = [
        TrackMDP(ref_q_hist, ref_qdot_hist, ref_poses_hist, ref_vels_hist, actor_obs_noise=True, train=True, max_time=4)
        for _ in range(64)
    ]
    toc = time.time()
    print(toc-tic)

    ppo = PPO_Clip(
        envs, 
        actor_dim=actor_dim,
        critic_dim=critic_dim,
        actor_state_dim=envs[0].actor_state_dim,
        critic_state_dim=envs[0].critic_state_dim,
        init_log_std=-0.5,
        learning_rate=2e-4,
        n_steps=128,
        batch_size=512,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=ent_coef,
        vf_coef=0.5,
        max_grad_norm=1.,
        target_kl=0.05
    )

    if from_last:
        ppo.load(motion + ".pth")

    ppo.train(steps, print_every=1)

    ppo.save(motion + ".pth")


def visualize(motion, max_len, save_gif=False):
    path = os.path.join(Constants.INTERPOLATED_PATH, motion+".mat")
    mat = loadmat(path)
    ref_q_hist = mat["q_hist"][:max_len]
    ref_qdot_hist = mat["qdot_hist"][:max_len]
    ref_poses_hist = mat["poses_hist"][:max_len]
    ref_vels_hist = mat["vels_hist"][:max_len]
    env = TrackMDP(ref_q_hist, ref_qdot_hist, ref_poses_hist, ref_vels_hist, actor_obs_noise=False, train=False, max_time=10)

    ppo = PPO_Clip(
        [env], 
        actor_dim=actor_dim,
        critic_dim=critic_dim,
        actor_state_dim=env.actor_state_dim,
        critic_state_dim=env.critic_state_dim,
        init_log_std=-0.5,
        learning_rate=2e-4,
        n_steps=128,
        batch_size=2048,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.,
        vf_coef=0.5,
        max_grad_norm=1.,
        target_kl=0.1
    )

    ppo.load(motion + ".pth")

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        a_state, _ = env.reset()
        renderer = mujoco.Renderer(env.model, height=480, width=640)
        gif = []

        reward_total = 0.
        while viewer.is_running():
            viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_WIREFRAME] = 0

            state_th = torch.from_numpy(a_state)
            state_th = ppo.actor_stat.normalize(state_th)[None,:]
            normal, action_th, _ = ppo.pi(state_th)

            # action = env.action_space.sample()
            # action = action_th[0].detach().numpy()
            action = normal.mean[0].detach().numpy()
            # print(action)

            a_state, _, reward, terminated, truncated = env.step(action)

            reward_total += reward

            if terminated or truncated:
                print(terminated, truncated)
                break

            viewer.cam.lookat[:] = env.data.qpos[0:3]
            
            viewer.sync()
            if save_gif:
                renderer.update_scene(env.data, camera=viewer.cam)
                rgb_image = renderer.render()
                gif.append(rgb_image)

            time.sleep(env.dt)
    if save_gif:
        gif = np.stack(gif)
        imageio.mimsave(
            motion + "_learned.gif",
            gif[::3],
            duration=1 / Constants.INTERPOLATE_FPS * 5 * 1000,
            loop=0
        )


if __name__ == '__main__':
    motion = "walk1_subject1"
    # motion = "fight1_subject2"
    max_len = 500 # for resource limit, we only track first 500 frames
    mode = sys.argv[1]
    if mode == "train":

        """
        walk1_subject1
        """ 
        # train(motion, max_len=max_len, ent_coef=5e-3, steps=2e6, from_last=False) 
        # train(motion, max_len=max_len, ent_coef=5e-3, steps=3e6, from_last=True) 
        # train(motion, max_len=max_len, ent_coef=5e-3, steps=5e6, from_last=True) 

        # train(motion, max_len=max_len, ent_coef=5e-3, steps=5e6, from_last=True)
        train(motion, max_len=max_len, ent_coef=5e-3, steps=5e6, from_last=True)

        """
        fight1_subject2
        """
        # train(motion, max_len=max_len, ent_coef=5e-3, steps=2e6, from_last=False) 
        # train(motion, max_len=max_len, ent_coef=5e-3, steps=3e6, from_last=True) 
        # train(motion, max_len=max_len, ent_coef=5e-3, steps=2e6, from_last=True) 
        # train(motion, max_len=max_len, ent_coef=5e-3, steps=3e6, from_last=True) 

        # train(motion, max_len=max_len, ent_coef=5e-3, steps=2e6, from_last=True) 
        # train(motion, max_len=max_len, ent_coef=5e-3, steps=2e6, from_last=True) 
        # train(motion, max_len=max_len, ent_coef=5e-3, steps=2e6, from_last=True) 
        # train(motion, max_len=max_len, ent_coef=5e-3, steps=2e6, from_last=True) 
        # train(motion, max_len=max_len, ent_coef=5e-3, steps=2e6, from_last=True) 

        # train(motion, max_len=max_len, ent_coef=5e-3, steps=2e6, from_last=True) 
        # train(motion, max_len=max_len, ent_coef=5e-3, steps=3e6, from_last=True) 
        # train(motion, max_len=max_len, ent_coef=5e-3, steps=5e6, from_last=True) 


        # train(motion, max_len=max_len, ent_coef=5e-3, steps=3e6, from_last=True) 
        # train(motion, max_len=max_len, ent_coef=5e-3, steps=2e6, from_last=True) 
        # train(motion, max_len=max_len, ent_coef=5e-3, steps=2e6, from_last=True) 
        # train(motion, max_len=max_len, ent_coef=5e-3, steps=3e6, from_last=True) 

        # train(motion, max_len=max_len, ent_coef=5e-3, steps=5e6, from_last=True) 
        # train(motion, max_len=max_len, ent_coef=5e-3, steps=5e6, from_last=True) 
        



        

        pass


    elif mode == "test":
        visualize(
            motion, 
            max_len, 
            save_gif=False
        )