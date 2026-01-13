import sys

import gymnasium as gym
import numpy as np
from ppo_model import PPOAgent
from grpo_model import GRPOAgent
from utils import plot_learning_curve
import gymnasium_robotics
gym.register_envs(gymnasium_robotics)

from collections import OrderedDict
import torch
from torch.utils.tensorboard import SummaryWriter
import time, os

# ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# sys.path.append(ROOT_DIR)

from imitation_learning_pusher.bc_network import BCnetwork

def _flatten_obs(obs):
    if isinstance(obs, dict):
        o = np.asarray(obs["observation"], dtype=np.float32)
        a = np.asarray(obs["achieved_goal"], dtype=np.float32)
        d = np.asarray(obs["desired_goal"], dtype=np.float32)
        return np.concatenate([o, a, d], axis=-1).astype(np.float32)
    return np.asarray(obs, dtype=np.float32)



if __name__ == "__main__":
    run_dir = os.path.join("runs", time.strftime("%Y%m%d-%H%M%S"))
    writer = SummaryWriter(run_dir)
    env_name = 'FetchPickAndPlace-v4'
    env = gym.make(env_name)
    seed = 0
    obs, _ = env.reset(seed = seed)
    flat = _flatten_obs(obs)
    update_every = 1000
    batch_size = 64
    n_epochs = 3
    alpha = 1e-4
    policy_clip = 0.2

    success_history= []
    ep_len_history =[]
    success_len_history = []
    fail_len_history = []

    agent = PPOAgent(n_actions=env.action_space.shape[0], batch_size=batch_size, N=update_every, alpha=alpha,
                     policy_clip=policy_clip, n_epochs=n_epochs, input_dims=(flat.shape[0],))
    bc_state = torch.load(
        "../imitation_learning_pusher/models/fetch_pick_and_place/expert-v0.pth",
        map_location="cpu"
    )
    agent.load_bc_weights(bc_state)

    n_episode = 5000

    total_timesteps = 60000 # end
    global_episode = 0 # recording

    while global_episode < n_episode:
        obs, info = env.reset(seed = global_episode + 1)
        obs = _flatten_obs(obs)

        done = False
        ep_success = 0
        ep_len = 0

        while not done:
            action, _, _ = agent.choose_action(obs)
            obs_, _, terminated, truncated, info = env.step(action)
            ep_len += 1
            obs = _flatten_obs(obs_)
            is_success = int(info.get("is_success", 0))
            ep_success = max(ep_success, is_success)
            done = terminated or truncated or (is_success == 1)

            ep_success = max(ep_success, int(info.get("is_success", 0)))

        global_episode += 1
        success_history.append(ep_success)
        ep_len_history.append(ep_len)
        if ep_success == 1:
            success_len_history.append(ep_len)
        else:
            fail_len_history.append(ep_len)

        success_rate = np.mean(success_history[-100:])

        writer.add_scalar("charts/success_rate_100", success_rate, global_episode)

        avg_ep_len_100 = np.mean(ep_len_history[-100:])
        writer.add_scalar("charts/avg_ep_len_100", avg_ep_len_100, global_episode)

        if len(success_len_history) > 0:
            avg_success_len_100 = np.mean(success_len_history[-100:])
            writer.add_scalar("charts/avg_success_len_100", avg_success_len_100, global_episode)

        if len(fail_len_history) > 0:
            avg_fail_len_100 = np.mean(fail_len_history[-100:])
            writer.add_scalar("charts/avg_fail_len_100", avg_fail_len_100, global_episode)
        print(f"iter:{global_episode}")
