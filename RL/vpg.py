
import numpy as np
import torch
from torch.optim import Adam
from policy import GaussianPolicy, CategoricalPolicy
import gymnasium as gym



class VPG:
    def __init__(self,
                 env_name="CartPole-v1",
                 hidden_sizes=32,
                 out_activation=None,
                 policy_type="Categorical",
                 batch_size=5000):
        self.env = gym.make(env_name)
        self.obs_dim = self.env.observation_space.shape[0]
        if policy_type == "Categorical":
            self.act_dim = self.env.action_space.n
            print(f"policy_type: {policy_type}")
            self.policy = CategoricalPolicy(self.obs_dim, hidden_sizes, self.act_dim, out_activation)
        else:
            self.act_dim = self.env.action_space.shape[0]
            self.policy = GaussianPolicy(self.obs_dim, hidden_sizes, self.act_dim, out_activation)

        self.batch_size = batch_size



    def reward_to_go(self, rews: list[float]) -> np.ndarray:
        """
            R_t = \sum_{t' = t ^ n} r_t' , t = n to 0
            R_n = r_n
            R_t = r_t + R_{t+1}
        """
        rtgs = np.zeros(len(rews), dtype=float)
        rtgs[-1] = rews[-1]  # last timestep
        for i in reversed(range(len(rews)-1)):
            rtgs[i] = rews[i] + rtgs[i+1]
        return rtgs


    def train(self, lr=0.01, epochs=50):
        self.optimizer = Adam(self.policy.parameters(), lr=lr)

        for i in range(epochs):
            batch_loss, batch_rets, batch_lens = self.train_one_epoch()
            print(f"epoch: {i:3d} \t loss: {batch_loss:.3f} "
                  f"\t return: {np.mean(batch_rets):.3f} "
                  f"\t ep_len: {np.mean(batch_lens):.3f}")



    def train_one_epoch(self):
        # generate trajectories
        batch_obs, batch_acts, batch_advs, batch_rets, batch_len = self.collect_trajectory()

        # convert to tensors
        obs_tensor = torch.as_tensor(batch_obs, dtype=torch.float32)
        acts_tensor = torch.as_tensor(batch_acts, dtype=torch.float32)
        advs_tensor = torch.as_tensor(batch_advs, dtype=torch.float32)

        # update policy
        self.optimizer.zero_grad()
        batch_loss = self.policy.compute_loss(obs_tensor, acts_tensor, advs_tensor)
        batch_loss.backward()
        self.optimizer.step()

        return batch_loss, batch_rets, batch_len

    def collect_trajectory(self):
        batch_obs, batch_acts, batch_advs = [], [], []
        batch_rets, batch_len = [], []

        obs, _ = self.env.reset()
        ep_rews = []

        while True:
            batch_obs.append(obs.copy())
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
            act = self.policy.get_action(obs_tensor)
            obs, rew, terminated, truncated, _ = self.env.step(act)
            done = terminated or truncated

            batch_acts.append(act)
            ep_rews.append(rew)

            if done:
                ep_ret, ep_len = sum(ep_rews), len(ep_rews)
                batch_rets.append(ep_ret)
                batch_len.append(ep_len)
                batch_advs.extend(self.reward_to_go(ep_rews))  # reward-to-go aligns with obs/acts

                # reset episode
                obs, _ = self.env.reset()
                done = False
                ep_rews = []

                # stop collecting only after finishing the current episode
                if len(batch_obs) >= self.batch_size:
                    break

        # truncate to batch_size exactly (optional)
        batch_obs = batch_obs[:self.batch_size]
        batch_acts = batch_acts[:self.batch_size]
        batch_advs = batch_advs[:self.batch_size]

        return batch_obs, batch_acts, batch_advs, batch_rets, batch_len


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', '--env', type=str, default='CartPole-v0')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--lr', type=float, default=1e-2)
    args = parser.parse_args()
    print('\nUsing reward-to-go formulation of policy gradient.\n')

    vpg = VPG(env_name=args.env_name)
    # vpg = VPG(env_name='Pendulum-v1', policy_type="Gaussian")
    vpg.train(lr=args.lr)
