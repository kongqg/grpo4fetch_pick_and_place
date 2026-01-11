import os
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



def _flatten_obs(obs):
    if isinstance(obs, dict):
        o = np.asarray(obs["observation"], dtype=np.float32)
        a = np.asarray(obs["achieved_goal"], dtype=np.float32)
        d = np.asarray(obs["desired_goal"], dtype=np.float32)
        return np.concatenate([o, a, d], axis=-1).astype(np.float32)
    return np.asarray(obs, dtype=np.float32)




if __name__ == '__main__':
    run_dir = os.path.join("runs", time.strftime("%Y%m%d-%H%M%S"))
    writer = SummaryWriter(run_dir)
    env_name = 'FetchPickAndPlace-v4'
    env = gym.make(env_name)
    print("max_episode_steps =", env.spec.max_episode_steps)
    update_every = 1000
    batch_size = 64
    n_epochs = 3
    alpha = 1e-4
    policy_clip = 0.2
    obs, _ = env.reset()
    flat = _flatten_obs(obs)
    algo = "grpo"
    if algo == "grpo":
        agent = GRPOAgent(n_actions=env.action_space.shape[0], batch_size=batch_size, N=update_every, alpha=alpha, policy_clip=policy_clip, n_epochs=n_epochs, input_dims=(flat.shape[0],))
    elif algo == "ppo":
        agent = PPOAgent(n_actions=env.action_space.shape[0], batch_size=batch_size, N=update_every, alpha=alpha,
                          policy_clip=policy_clip, n_epochs=n_epochs, input_dims=(flat.shape[0],))
    bc_state = torch.load(
        "../imitation-learning-pusher/models/fetch_pick_and_place/expert-v0.pth",
        map_location="cpu"
    )
    agent.load_bc_weights(bc_state)
    n_episodes = 5000

    if not os.path.exists('plots'):
        os.makedirs('plots')
    figure_file = f'plots/{env_name}.png'

    global_step = 0
    learn_iters = 0

    score_history = []
    success_history = []  # 每个episode一个0/1
    ep_len_history = []  # 每个episode的长度（步数）
    success_len_history = []  # 只记录成功episode的长度
    fail_len_history = []  # 只记录失败episode的长度（可选）

    best_score = -1e9


    total_timesteps = 60000 # compare by env steps
    group_size = 32 # trajectories per "problem" in GRPO

    episode = 0
    group_id = 0

    while global_step < total_timesteps:
        # =========================
        # GRPO pipeline
        # =========================
        if algo == "grpo":
            seed = group_id  # same seed => (roughly) same initial state/goal
            group_id += 1

            # 1) collect n_rollouts full trajectories under the same "problem"
            for k in range(group_size):
                obs, info = env.reset(seed=seed)
                observation = _flatten_obs(obs)

                done = False
                score = 0.0
                ep_success = 0
                ep_len = 0

                while (not done) and (global_step < total_timesteps):
                    action, prob, val = agent.choose_action(observation)

                    obs_, reward, terminated, truncated, info = env.step(action)
                    ep_len += 1
                    observation_ = _flatten_obs(obs_)

                    is_success = int(info.get("is_success", 0))
                    ep_success = max(ep_success, is_success)

                    done = terminated or truncated or (is_success == 1)
                    score += reward
                    ep_success = max(ep_success, int(info.get("is_success", 0)))

                    # hard stop on step budget (so PPO/GRPO compare with identical env steps)
                    if global_step + 1 >= total_timesteps:
                        done = True

                    agent.remember(observation, action, prob, val, reward, done)

                    observation = observation_
                    global_step += 1

                # logging per rollout-episode
                score_history.append(score)
                success_history.append(ep_success)
                ep_len_history.append(ep_len)
                if ep_success == 1:
                    success_len_history.append(ep_len)
                else:
                    fail_len_history.append(ep_len)

                avg_score = np.mean(score_history[-100:])
                success_rate = np.mean(success_history[-100:])

                writer.add_scalar("charts/episode_score", score, episode)
                writer.add_scalar("charts/avg_score_100", avg_score, episode)
                writer.add_scalar("charts/success_rate_100", success_rate, episode)

                avg_ep_len_100 = np.mean(ep_len_history[-100:])
                writer.add_scalar("charts/avg_ep_len_100", avg_ep_len_100, episode)

                if len(success_len_history) > 0:
                    avg_success_len_100 = np.mean(success_len_history[-100:])
                    writer.add_scalar("charts/avg_success_len_100", avg_success_len_100, episode)

                if len(fail_len_history) > 0:
                    avg_fail_len_100 = np.mean(fail_len_history[-100:])
                    writer.add_scalar("charts/avg_fail_len_100", avg_fail_len_100, episode)

                print(f"[GRPO] episode {episode} score {score:.1f} avg_score {avg_score:.1f} "
                      f"success_rate {success_rate:.2f} global_step {global_step} learn_iters {learn_iters}")

                if success_rate > best_score:
                    best_score = success_rate
                    agent.save_models()

                episode += 1
                if global_step >= total_timesteps:
                    break

            # 2) update policy once per group
            stats = agent.learn_grpo(debug=True)
            if stats is not None:
                for k, v in stats.items():
                    writer.add_scalar(k, v, global_step)
            writer.flush()
            learn_iters += 1

        # =========================
        # PPO pipeline (your original)
        # =========================
        else:
            obs, info = env.reset()
            observation = _flatten_obs(obs)

            done = False
            score = 0.0
            ep_success = 0
            ep_len = 0

            while (not done) and (global_step < total_timesteps):
                action, prob, val = agent.choose_action(observation)

                obs_, reward, terminated, truncated, info = env.step(action)
                ep_len += 1
                observation_ = _flatten_obs(obs_)

                is_success = int(info.get("is_success", 0))
                ep_success = max(ep_success, is_success)

                done = terminated or truncated or (is_success == 1)
                score += reward
                ep_success = max(ep_success, int(info.get("is_success", 0)))

                # hard stop on step budget
                if global_step + 1 >= total_timesteps:
                    done = True

                agent.remember(observation, action, prob, val, reward, done)

                observation = observation_
                global_step += 1

                if global_step % update_every == 0:
                    stats = agent.learn(debug=True)
                    if stats is not None:
                        for k, v in stats.items():
                            writer.add_scalar(k, v, global_step)
                    writer.flush()
                    learn_iters += 1

            score_history.append(score)
            success_history.append(ep_success)

            avg_score = np.mean(score_history[-100:])
            success_rate = np.mean(success_history[-100:])

            writer.add_scalar("charts/episode_score", score, episode)
            writer.add_scalar("charts/avg_score_100", avg_score, episode)
            writer.add_scalar("charts/success_rate_100", success_rate, episode)

            ep_len_history.append(ep_len)
            if ep_success == 1:
                success_len_history.append(ep_len)
            else:
                fail_len_history.append(ep_len)

            avg_ep_len_100 = np.mean(ep_len_history[-100:])
            writer.add_scalar("charts/avg_ep_len_100", avg_ep_len_100, episode)

            if len(success_len_history) > 0:
                avg_success_len_100 = np.mean(success_len_history[-100:])
                writer.add_scalar("charts/avg_success_len_100", avg_success_len_100, episode)

            if len(fail_len_history) > 0:
                avg_fail_len_100 = np.mean(fail_len_history[-100:])
                writer.add_scalar("charts/avg_fail_len_100", avg_fail_len_100, episode)

            print(f"[PPO] episode {episode} score {score:.1f} avg_score {avg_score:.1f} "
                  f"success_rate {success_rate:.2f} global_step {global_step} learn_iters {learn_iters}")

            if success_rate > best_score:
                best_score = success_rate
                agent.save_models()

            episode += 1
    x = [i+1 for i in range(len(score_history))]
    plot_learning_curve(x, score_history, figure_file)

