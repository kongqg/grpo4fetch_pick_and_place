import os
from collections import defaultdict

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


def group_then_split_episodes(traj):
    groups = defaultdict(list)        # group_id -> List[episode]
    cur_ep = defaultdict(list)        # group_id -> current episode steps (临时累积)

    for step in traj:
        obs, action, logp, value, reward, done, gid = step

        cur_ep[gid].append(step)

        if done:                      # done=True 代表这一条 episode 结束
            groups[gid].append(cur_ep[gid])
            cur_ep[gid] = []          # 清空，准备下一条 episode

    # 如果最后有没以 done 结束的残留（一般不该有），你可以选择丢掉或补上
    # for gid, ep in cur_ep.items():
    #     if ep: groups[gid].append(ep)

    return groups

class GRPOAgent:
    def __init__(self, n_actions, input_dims, gamma=0.99, alpha=1e-4, gae_lambda=0.95,
                 policy_clip=0.2, batch_size=64, N=2048, n_epochs=3):
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda

        self.actor = ActorNetwork(n_actions, input_dims, alpha)
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

    def load_models(self):
        print("... loading model")
        self.actor.load_checkpoint()

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

            # ===== 加这一段：数值防线（定位爆炸源头）=====
            if (not torch.isfinite(mu).all()) or (not torch.isfinite(std).all()):
                print("[ERR] non-finite mu/std", mu, std)
                raise RuntimeError("non-finite policy params")
            if (std <= 0).any():
                print("[ERR] non-positive std", std)
                raise RuntimeError("non-positive std")
            dist = Normal(mu, std)

            raw_action = dist.rsample()              # (1, n_actions)
            action = torch.tanh(raw_action)          # [-1,1], (1, n_actions)

            log_prob = self._tanh_squash_log_prob(dist, raw_action, action)  # (1,)

        action_np = action.squeeze(0).cpu().numpy().astype(np.float32)  # (n_actions,)
        log_prob_item = log_prob.item()

        return action_np, log_prob_item

    def learn_grpo(self, trajectory):
        """
        GRPO-style policy update (no value network):
        - Collect a *group* of full trajectories into memory
        - Compute each trajectory return G_i, baseline = mean(G_i)
        - Advantage for every step in trajectory i: A_t = (G_i - baseline)
        - Do ONE full-batch PPO clipped policy update using this advantage
        """
        device = self.actor.device
        target_kl = 0.02

        """
        groups[gid] = [
          episode_0 = [step0, step1, ...],
          episode_1 = [...],
          ...
        ]
        """
        ep_infos = []  # list of (gid, start, end, G)
        start = 0
        G = 0.0
        cur_gid = trajectory[0][6]

        for i, step in enumerate(trajectory):
            reward = float(step[2])
            done = bool(step[4])
            gid = step[6]

            if i == start:
                cur_gid = gid
                G = 0.0

            G += reward

            if done:
                ep_infos.append((cur_gid, start, i, G))
                start = i + 1

        # ------------------------------------------------------------
        # 2) compute per-group baseline (mean return in that group)
        # ------------------------------------------------------------
        group_returns = defaultdict(list)
        for gid, s, e, Gi in ep_infos:
            group_returns[gid].append(Gi)

        group_baseline = {
            gid: float(np.mean(episode_returns))
            for gid, episode_returns in group_returns.items()
        }

        # advantage: per step uses (G_i - baseline)
        adv = np.zeros(len(trajectory), dtype=np.float32)
        for gid, s, e, Gi in ep_infos:
            adv[s:e + 1] = float(Gi - group_baseline[gid])
        # # normalize advantage
        # adv_mean = float(adv.mean())
        # adv_std = float(adv.std() + 1e-8)
        # adv_norm = (adv - adv_mean) / adv_std

        # -------- full batch tensors --------
        state_arr = np.asarray([step[0] for step in trajectory], dtype=np.float32)
        action_arr = np.asarray([step[1] for step in trajectory], dtype=np.float32)
        old_logp_arr = np.asarray([step[5] for step in trajectory], dtype=np.float32)

        states = torch.tensor(state_arr, dtype=torch.float32, device=device)
        actions = torch.tensor(action_arr, dtype=torch.float32, device=device)
        old_logp = torch.tensor(old_logp_arr, dtype=torch.float32, device=device)
        adv_b = torch.tensor(adv, dtype=torch.float32, device=device)

        mu, std = self.actor(states)
        dist = Normal(mu, std)

        # actions are tanh-squashed -> invert to raw_action for correct log_prob
        eps = 1e-6
        a = torch.clamp(actions, -1 + eps, 1 - eps)
        raw_action = 0.5 * (torch.log1p(a) - torch.log1p(-a))  # atanh(a)

        new_logp = self._tanh_squash_log_prob(dist, raw_action, actions)

        prob_ratio = torch.exp(new_logp - old_logp)
        surr1 = prob_ratio * adv_b
        surr2 = torch.clamp(prob_ratio, 1 - self.policy_clip, 1 + self.policy_clip) * adv_b
        actor_loss = -torch.min(surr1, surr2).mean()

        entropy = dist.entropy().sum(dim=-1).mean()
        total_loss = actor_loss - 0.01 * entropy

        self.actor.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        self.actor.optimizer.step()

        with torch.no_grad():
            mu2, std2 = self.actor(states)
            dist2 = Normal(mu2, std2)
            new_logp_post = self._tanh_squash_log_prob(dist2, raw_action, actions)

            ratio_post = torch.exp(new_logp_post - old_logp)
            approx_kl_post = (old_logp - new_logp_post).mean().item()
            clip_frac_post = (
                        (ratio_post > 1 + self.policy_clip) | (ratio_post < 1 - self.policy_clip)).float().mean().item()
            ratio_max_post = ratio_post.max().item()
            ratio_min_post = ratio_post.min().item()
            std_mean_post = std2.mean().item()

        # debug metrics (full-batch)
        stats = {}
        with torch.no_grad():
            clip_frac = (
                        (prob_ratio > 1 + self.policy_clip) | (prob_ratio < 1 - self.policy_clip)).float().mean().item()
            approx_kl = (old_logp - new_logp).mean().item()
            ratio_max = prob_ratio.max().item()
            ratio_min = prob_ratio.min().item()
            std_mean = std.mean().item()
            ent = entropy.item()

            baseline_vals = np.array(list(group_baseline.values()), dtype=np.float32)
            baseline_mean = float(baseline_vals.mean())
            baseline_std_ = float(baseline_vals.std()) if baseline_vals.size > 1 else 0.0

            if approx_kl > 1.5 * target_kl:
                print(f"[GRPO] kl too large in single-step update: kl={approx_kl:.4f} (target {target_kl})")

            print(
                f"[GRPO dbg] "
                f"actor_loss={actor_loss.item():.6f} "
                f"kl~={approx_kl_post:.6f} "
                f"clip%={clip_frac_post * 100:.1f}% "
                f"ratio[min,max]=[{ratio_min_post:.3f},{ratio_max_post:.3f}] "
                f"std_mean={std_mean_post:.3f} "
                f"entropy={ent:.3f} "
                f"n_groups={len(group_returns)} n_traj={len(ep_infos)} "
                f"baseline(mu/std)={baseline_mean:.3f}/{baseline_std_:.3f} "
            )

            stats = {
                "loss/actor": float(actor_loss.item()),
                "debug/kl": float(approx_kl),
                "debug/clip_frac": float(clip_frac),
                "debug/ratio_max": float(ratio_max),
                "debug/std_mean": float(std_mean),
                "grpo/n_groups": float(len(group_returns)),
                "grpo/n_traj": float(len(ep_infos)),
                "grpo/baseline_mean": float(baseline_mean),
                "grpo/baseline_std": float(baseline_std_),
            }

        self.memory.clear_memory()
        return stats


