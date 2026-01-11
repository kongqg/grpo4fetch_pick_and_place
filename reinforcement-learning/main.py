import os
import gymnasium as gym
import numpy as np
from ppo_model import Agent
from utils import plot_learning_curve
import gymnasium_robotics
gym.register_envs(gymnasium_robotics)

from collections import OrderedDict
import torch



def _flatten_obs(obs):
    if isinstance(obs, dict):
        o = np.asarray(obs["observation"], dtype=np.float32)
        a = np.asarray(obs["achieved_goal"], dtype=np.float32)
        d = np.asarray(obs["desired_goal"], dtype=np.float32)
        return np.concatenate([o, a, d], axis=-1).astype(np.float32)
    return np.asarray(obs, dtype=np.float32)




if __name__ == '__main__':
    env_name = 'FetchPickAndPlaceDense-v4'
    env = gym.make(env_name)
    update_every = 2000
    batch_size = 64
    n_epochs = 10
    alpha = 3e-4
    obs, _ = env.reset()
    flat = _flatten_obs(obs)
    agent = Agent(n_actions=env.action_space.shape[0], batch_size=batch_size, N=update_every, alpha=alpha, policy_clip=0.3, n_epochs=n_epochs, input_dims=(flat.shape[0],))
    bc_state = torch.load(
        "../imitation-learning-pusher/models/fetch_pick_and_place/expert-v0.pth",
        map_location="cpu"
    )
    agent.load_bc_weights(bc_state)
    n_games = 10000

    if not os.path.exists('plots'):
        os.makedirs('plots')
    figure_file = f'plots/{env_name}.png'

    best_score = 0
    score_history = []

    learn_iters = 0
    best_score = -float('inf')
    n_steps = 0

    for i in range(n_games):
        obs, _ = env.reset()
        observation = _flatten_obs(obs)
        done = False
        score = 0
        while not done:
            action, prob, val = agent.choose_action(observation)
            obs_, reward, terminated, truncated, info = env.step(action)
            observation_ = _flatten_obs(obs_)
            done = terminated or truncated
            n_steps += 1
            score += reward
            agent.remember(observation, action, prob, val, reward, done)
            if n_steps % update_every  == 0:
                agent.learn(debug=True)
                learn_iters += 1
            observation = observation_

        score_history.append(score)
        avg_score = np.mean(score_history[-100:])
        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()

        print('episode', i, 'score %.1f' % score, 'avg score %.1f' % avg_score, 'time_steps', n_steps, 'learning_steps', learn_iters)

    x = [i+1 for i in range(len(score_history))]
    plot_learning_curve(x, score_history, figure_file)

