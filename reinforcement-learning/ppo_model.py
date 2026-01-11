import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Tuple

from torch.distributions import Normal
from torch.distributions.categorical import Categorical

class PPOMemory:
    """
        A class to handle memory for the PPO reinforcement learning algorithm.

    Attributes:
        batch_size (int): The size of each batch for training.
        states (List): List of stored states.
        probs (List): List of action probabilities from the policy.
        vals (List): List of value function estimates.
        actions (List): List of stored actions.
        rewards (List): List of rewards received for taking actions.
        dones (List): List of boolean flags indicating episode termination.
    """
    def __init__(self, batch_size: int):
        """
        Initialize the memory buffer and set batch size.

        Args:
            batch_size (int): The size of each training batch.
        """
        self.states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []

        self.batch_size: int = batch_size

    def generate_batches(self):
        """
        Generate randomized batches for training.

        Returns:
            Tuple containing:
                - states (np.ndarray): Array of stored states.
                - actions (np.ndarray): Array of stored actions.
                - probs (np.ndarray): Array of action probabilities.
                - vals (np.ndarray): Array of value function estimates.
                - rewards (np.ndarray): Array of rewards.
                - dones (np.ndarray): Array of done flags.
                - batches (List[np.ndarray]): List of indices for each batch.
        """
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        idx = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(idx) # Add randomness for Random SGD
        batches = [idx[i:i+self.batch_size] for i in batch_start]

        return np.array(self.states),\
                np.array(self.actions),\
                np.array(self.probs),\
                np.array(self.vals),\
                np.array(self.rewards),\
                np.array(self.dones),\
                batches

    def store_memory(self, state, action, probs, vals, reward, done):
        """
        Store an experience in the memory buffer.

        Args:
            state (any): The observed state.
            action (any): The action taken.
            probs (any): The probability of the action from the policy.
            vals (any): The estimated value of the state.
            reward (float): The reward received.
            done (bool): Flag indicating episode termination.
        """
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self):
        """
        Clear all stored memory from the buffer.
        """
        self.states = []
        self.probs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.vals = []

class ActorNetwork(nn.Module):
    def __init__(self, n_actions, input_dims, alpha, fc1_dims=256, fc2_dims=256, chkpt_dir='checkpoints/ppo'):
        super().__init__()
        self.checkpoint_file = os.path.join(chkpt_dir, 'actor_torch_ppo')

        self.net = nn.Sequential(
            nn.Linear(*input_dims, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
        )
        self.mu = nn.Linear(fc2_dims, n_actions)

        # 用一个可学习的 log_std（每个动作维度一个）
        self.log_std = nn.Parameter(torch.ones(n_actions) * -1.5)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        h = self.net(state)
        mu = self.mu(h)  # (B, n_actions)
        log_std = torch.clamp(self.log_std, -5, 2)
        std = torch.exp(log_std).expand_as(mu) # (B, n_actions)
        return mu, std

    def save_checkpoint(self):
        os.makedirs(os.path.dirname(self.checkpoint_file), exist_ok=True)
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file, map_location=self.device))


class CriticNetwork(nn.Module):
    def __init__(self, input_dims, alpha, fc1_dims=256, fc2_dims=256, chkpt_dir='checkpoints/ppo'):
        super().__init__()
        self.checkpoint_file = os.path.join(chkpt_dir, 'critic_torch_ppo')

        self.critic = nn.Sequential(
            nn.Linear(*input_dims, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, 1),
        )

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        return self.critic(state)

    def save_checkpoint(self):
        os.makedirs(os.path.dirname(self.checkpoint_file), exist_ok=True)
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file, map_location=self.device))


class Agent:
    def __init__(self, n_actions, input_dims, gamma=0.99, alpha=0.0003, gae_lambda=0.95,
                 policy_clip=0.2, batch_size=64, N=2048, n_epochs=10):
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda

        self.actor = ActorNetwork(n_actions, input_dims, alpha)
        self.critic = CriticNetwork(input_dims, alpha)
        self.memory = PPOMemory(batch_size)

    def load_bc_weights(self, bc_state):
        """
        Load Behavior Cloning (BC) weights into PPO actor network.
        Copies BC MLP weights into PPO ActorNetwork (net.0, net.2, mu).

        Args:
            bc_state (dict): state_dict from BCnetwork (keys like network.fc1.weight)
        """

        # --- 兼容：有些保存会包一层 ---
        if isinstance(bc_state, dict) and "model" in bc_state:
            bc_state = bc_state["model"]
        if isinstance(bc_state, dict) and "state_dict" in bc_state:
            bc_state = bc_state["state_dict"]

        actor_state = self.actor.state_dict()

        # === Sanity check: BC keys 必须存在 ===
        required_bc_keys = [
            "network.fc1.weight",
            "network.fc1.bias",
            "network.fc2.weight",
            "network.fc2.bias",
            "network.fc3.weight",
            "network.fc3.bias",
        ]
        for k in required_bc_keys:
            if k not in bc_state:
                raise KeyError(f"[BC LOAD ERROR] Missing key in bc_state: {k}")

        # === PPO actor 对应的 keys（你 PPO 里是 self.net + self.mu）===
        mapping = {
            "network.fc1.weight": "net.0.weight",
            "network.fc1.bias": "net.0.bias",
            "network.fc2.weight": "net.2.weight",
            "network.fc2.bias": "net.2.bias",
            "network.fc3.weight": "mu.weight",
            "network.fc3.bias": "mu.bias",
        }

        # === 强校验：key 必须存在且 shape 必须一致（否则直接报错，避免“假加载”）===
        for bc_k, ppo_k in mapping.items():
            if ppo_k not in actor_state:
                raise KeyError(f"[BC LOAD ERROR] PPO actor missing key: {ppo_k}")

            if actor_state[ppo_k].shape != bc_state[bc_k].shape:
                raise RuntimeError(
                    f"[BC LOAD ERROR] Shape mismatch for {ppo_k}: "
                    f"PPO {tuple(actor_state[ppo_k].shape)} vs BC {tuple(bc_state[bc_k].shape)}"
                )

            actor_state[ppo_k] = bc_state[bc_k]

        # === 加载回 actor ===
        self.actor.load_state_dict(actor_state)

        print("[INFO] BC weights loaded into PPO actor: net.0/net.2/mu (mu network only).")

    def remember(self, state, action, log_prob, val, reward, done):
        self.memory.store_memory(state, action, log_prob, val, reward, done)

    def save_models(self):
        print('... saving models')
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()

    def load_models(self):
        print("... loading model")
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()

    @staticmethod
    def _tanh_squash_log_prob(dist: Normal, raw_action: torch.Tensor, squashed_action: torch.Tensor):
        """
        对 tanh squashing 做 log_prob 修正：
        log π(a) = log N(u|μ,σ) - sum log(1 - tanh(u)^2)
        用tanh 限制在 -1，1之间，那原始的概率密度就变了，所以要变回去
        """
        # base log prob (B, n_actions)
        logp_u = dist.log_prob(raw_action)
        # sum over action dims -> (B,)
        logp_u = logp_u.sum(dim=-1)

        # correction term: sum log(1 - a^2 + eps)
        eps = 1e-6
        correction = torch.log(1.0 - squashed_action.pow(2) + eps).sum(dim=-1)

        return logp_u - correction  # (B,)

    def choose_action(self, observation):
        state = torch.tensor([observation], dtype=torch.float32).to(self.actor.device)

        with torch.no_grad():
            mu, std = self.actor(state)
            dist = Normal(mu, std)

            raw_action = dist.rsample()              # (1, n_actions)
            action = torch.tanh(raw_action)          # [-1,1], (1, n_actions)

            value = self.critic(state).squeeze(-1)   # (1,)
            log_prob = self._tanh_squash_log_prob(dist, raw_action, action)  # (1,)

        action_np = action.squeeze(0).cpu().numpy().astype(np.float32)  # (n_actions,)
        log_prob_item = log_prob.item()
        value_item = value.item()

        return action_np, log_prob_item, value_item

    def learn(self, debug: bool = False, debug_every: int = 1):
        device = self.actor.device

        # ===== 1) 取出 buffer（只取一次，不要每个 epoch 都重新 generate）=====
        state_arr, action_arr, old_logp_arr, vals_arr, reward_arr, done_arr, _ = self.memory.generate_batches()

        rewards = reward_arr.astype(np.float32)
        dones = done_arr.astype(np.float32)
        values = vals_arr.astype(np.float32)
        T = len(rewards)

        adv = np.zeros(T, dtype=np.float32)
        last_gae = 0.0

        for t in reversed(range(T)):
            next_value = values[t + 1] if (t + 1) < T else 0.0
            next_nonterminal = 1.0 - dones[t]  # done=True -> 0

            delta = rewards[t] + self.gamma * next_value * next_nonterminal - values[t]
            last_gae = delta + self.gamma * self.gae_lambda * next_nonterminal * last_gae
            adv[t] = last_gae

        # 这里的 returns 是 critic 的监督目标（不要用标准化后的 adv）
        returns = adv + values

        # PPO 通常要标准化 advantage（更稳）
        adv_mean = adv.mean()
        adv_std = adv.std() + 1e-8
        adv_norm = (adv - adv_mean) / adv_std

        # ===== 3) 开始 PPO 更新 =====
        n_states = len(state_arr)
        batch_size = self.memory.batch_size

        for epoch in range(self.n_epochs):
            # 每个 epoch 重新 shuffle index（等价于你之前每次 generate_batches 的随机性）
            idx = np.arange(n_states, dtype=np.int64)
            np.random.shuffle(idx)
            batches = [idx[i:i + batch_size] for i in range(0, n_states, batch_size)]

            # debug 聚合
            dbg_actor_losses = []
            dbg_critic_losses = []
            dbg_kls = []
            dbg_clip_fracs = []
            dbg_ratio_max = []
            dbg_ratio_min = []
            dbg_std_mean = []

            for batch in batches:
                states = torch.tensor(state_arr[batch], dtype=torch.float32, device=device)
                actions = torch.tensor(action_arr[batch], dtype=torch.float32, device=device)  # (B, n_actions)
                old_logp = torch.tensor(old_logp_arr[batch], dtype=torch.float32, device=device)  # (B,)

                adv_b = torch.tensor(adv_norm[batch], dtype=torch.float32,
                                     device=device)  # 用标准化 advantage 做 policy loss
                ret_b = torch.tensor(returns[batch], dtype=torch.float32, device=device)  # 用未标准化 returns 做 value loss

                # ===== critic =====
                critic_value = self.critic(states).squeeze(-1)  # (B,)

                # ===== actor: 重新算 new_logp =====
                mu, std = self.actor(states)
                dist = Normal(mu, std)

                # actions 是环境动作（tanh 后），需要 atanh 反解成 raw_action 来做 squash 修正
                eps = 1e-6
                a = torch.clamp(actions, -1 + eps, 1 - eps)
                raw_action = 0.5 * (torch.log1p(a) - torch.log1p(-a))  # atanh(a)

                new_logp = self._tanh_squash_log_prob(dist, raw_action, actions)  # (B,)

                # ===== PPO ratio & clipped objective =====
                prob_ratio = torch.exp(new_logp - old_logp)  # (B,)

                surr1 = prob_ratio * adv_b
                surr2 = torch.clamp(prob_ratio, 1 - self.policy_clip, 1 + self.policy_clip) * adv_b
                actor_loss = -torch.min(surr1, surr2).mean()

                # value loss
                critic_loss = (ret_b - critic_value).pow(2).mean()

                entropy = dist.entropy().sum(dim=-1).mean()
                total_loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy

                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()

                # 可选：梯度裁剪（更稳）
                # torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                # torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)

                self.actor.optimizer.step()
                self.critic.optimizer.step()

                # ===== debug 收集 =====
                if debug:
                    with torch.no_grad():
                        clip_frac = ((prob_ratio > 1 + self.policy_clip) | (
                                    prob_ratio < 1 - self.policy_clip)).float().mean().item()
                        approx_kl = (old_logp - new_logp).mean().item()  # 粗略KL
                        dbg_actor_losses.append(actor_loss.item())
                        dbg_critic_losses.append(critic_loss.item())
                        dbg_kls.append(approx_kl)
                        dbg_clip_fracs.append(clip_frac)
                        dbg_ratio_max.append(prob_ratio.max().item())
                        dbg_ratio_min.append(prob_ratio.min().item())
                        dbg_std_mean.append(std.mean().item())

            # 每隔 debug_every 个 epoch 打印一次（避免刷屏）
            if debug and ((epoch % debug_every) == 0):
                print(
                    f"[PPO dbg][epoch {epoch + 1}/{self.n_epochs}] "
                    f"actor_loss={np.mean(dbg_actor_losses):.4f} "
                    f"critic_loss={np.mean(dbg_critic_losses):.4f} "
                    f"kl~={np.mean(dbg_kls):.4f} "
                    f"clip%={np.mean(dbg_clip_fracs) * 100:.1f}% "
                    f"ratio[min,max]=[{np.min(dbg_ratio_min):.3f},{np.max(dbg_ratio_max):.3f}] "
                    f"std_mean={np.mean(dbg_std_mean):.3f} "
                    f"adv_raw(mu/std)={adv_mean:.3f}/{adv_std:.3f}"
                )

        # ===== 4) 清空 buffer =====
        self.memory.clear_memory()

