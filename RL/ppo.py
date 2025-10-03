import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from policy import GaussianPolicy, CategoricalPolicy, ValueNetwork
import gymnasium as gym


class PPO:
    def __init__(self,
                 env_name="CartPole-v1",
                 hidden_sizes=64,
                 batch_size=5000,
                 gamma=0.99,
                 lam=0.95,
                 lr=3e-4,
                 clip_ratio=0.2,
                 train_pi_iters=1,
                 train_v_iters=1,
                 target_kl=0.01,
                 out_activation=None,
                 policy_type="Categorical"):

        self.env = gym.make(env_name)
        self.obs_dim = self.env.observation_space.shape[0]
        self.gamma, self.lam = gamma, lam
        self.clip_ratio = clip_ratio
        self.train_pi_iters = train_pi_iters
        self.train_v_iters = train_v_iters
        self.target_kl = target_kl
        self.batch_size = batch_size

        print(f"policy_type: {policy_type}")
        if policy_type == "Categorical":
            self.act_dim = self.env.action_space.n
            self.policy = CategoricalPolicy(self.obs_dim, hidden_sizes, self.act_dim, out_activation)
        else:
            self.act_dim = self.env.action_space.shape[0]
            self.policy = GaussianPolicy(self.obs_dim, hidden_sizes, self.act_dim, out_activation)

        self.value = ValueNetwork(self.obs_dim, hidden_sizes, out_activation)

        self.policy_optimizer = Adam(self.policy.parameters(), lr=lr)
        self.value_optimizer = Adam(self.value.parameters(), lr=lr)

    def collect_trajectory(self):
        batch_obs, batch_acts, batch_logp, batch_rews, batch_vals, batch_rtgs = [], [], [], [], [], []
        batch_lens = []

        obs, _ = self.env.reset()
        ep_rews = []

        while True:
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
            act = self.policy.get_action(obs_tensor)
            act_tensor = torch.as_tensor(act, dtype=torch.float32)
            logp = self.policy.get_log_prob(obs_tensor, act_tensor)
            val = self.value(obs_tensor)

            next_obs, rew, terminated, truncated, _ = self.env.step(act)
            done = terminated or truncated

            # Store
            batch_obs.append(obs)
            batch_acts.append(act)
            batch_logp.append(logp.detach())
            batch_vals.append(val.item())
            ep_rews.append(rew)

            obs = next_obs

            if done:
                # compute rtgs
                rtgs = self.compute_rtgs(ep_rews)
                batch_rtgs.extend(rtgs)
                batch_rews.extend(ep_rews)
                batch_lens.append(len(ep_rews))

                obs, _ = self.env.reset()
                ep_rews = []

                if len(batch_obs) >= self.batch_size:
                    break

        return batch_obs, batch_acts, batch_logp, batch_vals, batch_rtgs, batch_rews, batch_lens

    def compute_rtgs(self, rews):
        rtgs = np.zeros(len(rews), dtype=np.float32)
        rtgs[-1] = rews[-1]
        for i in reversed(range(len(rews)-1)):
            rtgs[i] = rews[i] + self.gamma * rtgs[i+1]
        return rtgs

    def compute_advantages(self, rewards, values, last_val=0.0):
        """
        Compute advantages using Generalized Advantage Estimation (GAE).

        rewards: (T,)
        values:  (T,)
        last_val: scalar for bootstrap value (e.g., V(s_T))

        Returns:
            advs: (T,) array of advantages
        """
        T = len(rewards)
        advs = np.zeros(T, dtype=float)

        # extend values with last value for bootstrap
        values_ext = np.append(values, last_val)
        last_gae = 0.0

        for t in reversed(range(T)):
            delta = rewards[t] + self.gamma * values_ext[t + 1] - values_ext[t]
            last_gae = delta + self.gamma * self.lam * last_gae
            advs[t] = last_gae

        return advs


    def train_one_epoch(self):
        obs, acts, logp_old, vals, rtgs, rews, lens = self.collect_trajectory()

        obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
        acts_tensor = torch.as_tensor(acts, dtype=torch.float32)
        logp_old_tensor = torch.as_tensor(logp_old, dtype=torch.float32)
        rtgs_tensor = torch.as_tensor(rtgs, dtype=torch.float32)
        # advs = self.compute_advantages(rews, vals)
        # advs = rtgs - vals # A(t) = R(t) - V(s_t)
        advs_tensor = rtgs_tensor - torch.as_tensor(vals, dtype=torch.float32)

        # normalize advantages
        advs_tensor = (advs_tensor - advs_tensor.mean()) / (advs_tensor.std() + 1e-8)

        # policy update
        for i in range(self.train_pi_iters):
            logp = self.policy.get_log_prob(obs_tensor, acts_tensor)
            ratio = torch.exp(logp - logp_old_tensor)
            clipped_ratio = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
            pi_loss = - torch.min(ratio * advs_tensor, clipped_ratio * advs_tensor).mean()

            self.policy_optimizer.zero_grad()
            pi_loss.backward()
            self.policy_optimizer.step()

            # approx KL
            # kl = (logp_old_tensor - logp).mean().item()
            # if kl > 1.5 * self.target_kl:
                # print(f"Early stopping at step {i} due to reaching max kl.")
                # break

        # value update
        for _ in range(self.train_v_iters):
            self.value_optimizer.zero_grad()
            v_loss = self.value.compute_loss(obs_tensor, rtgs_tensor)
            v_loss.backward()
            self.value_optimizer.step()

        return pi_loss.item(), v_loss.item(), np.mean(rtgs), np.mean(lens)

    def train(self, epochs=50):
        for i in range(epochs):
            p_loss, v_loss, avg_ret, avg_len = self.train_one_epoch()
            print(f"Epoch {i:3d} \t p loss: {p_loss:.3f} \t v loss: {v_loss:.3f}"
                  f"\t avg return: {avg_ret:.3f} "
                  f"\t avg ep_len: {avg_len:.3f}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', '--env', type=str, default='CartPole-v0')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--lr', type=float, default=1e-2)
    args = parser.parse_args()
    print('\nUsing reward-to-go formulation of policy gradient.\n')

    ppo = PPO(env_name=args.env_name, lr=args.lr)
    # vpg = VPG(env_name='Pendulum-v1', policy_type="Gaussian")
    ppo.train()
