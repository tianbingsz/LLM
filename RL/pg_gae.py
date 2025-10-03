import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from policy import GaussianPolicy, CategoricalPolicy, ValueNetwork
import gymnasium as gym


class ActorCritic:
    def __init__(self,
                 env_name="CartPole-v1",
                 hidden_sizes=64,
                 batch_size=5000,
                 gamma=0.99,
                 lam=0.95,
                 lr=0.01,
                 out_activation=None,
                 policy_type="Categorical"):
        self.env = gym.make(env_name)
        self.obs_dim = self.env.observation_space.shape[0]

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

        self.batch_size = batch_size
        self.gamma = gamma
        self.lam = lam

    # compute reward-to-go
    def reward_to_go(self, rews):
        rtgs = np.zeros(len(rews), dtype=float)
        rtgs[-1] = rews[-1]
        for i in reversed(range(len(rews)-1)):
            rtgs[i] = rews[i] + self.gamma * rtgs[i+1]

        rtgs = (rtgs - np.mean(rtgs)) / (np.std(rtgs) + 1e-8)
        return rtgs


    # GAE function
    def gae(self, rewards, values):
        """
        rewards: (T,)
        values: (T,)
        returns: (T,) advantages using GAE

        delta(t) = r_t + gamma * V(s_{t+1}) - V(s_t)
        A_t = delta_t + \sum_{t'=t^T} (gamma * lam)^(T-t) A_{t'}
        adv(t) = delta(t) + gamma * lam * adv(t+1) (computed backwards)
        """
        T = len(rewards)
        advs = np.zeros(T, dtype=np.float32)
        values_ext = np.append(values, 0.0) # V(s_{t+1}) for last step
        # delta(t) = r_t + gamma * V(s_{t+1}) - V(s_t)
        delta = rewards + values_ext[1:] - values_ext[:-1]
        last_gae = 0.0
        for t in reversed(range(T)):
            # advs[t] = delta[t] + gamma * lamm * advs[t+1]
            last_gae = delta[t] + self.gamma * self.lam * last_gae
            advs[t] = last_gae

        # advs = (advs - np.mean(advs)) / (np.std(advs) + 1e-8)
        return advs


    # collect batch of trajectories
    def collect_trajectory(self):
        batch_obs, batch_acts, batch_rtgo, batch_advs = [], [], [], []
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

                # compute advantage: R_t - V(s_t)
                obs_tensor_batch = torch.as_tensor(batch_obs[-ep_len:], dtype=torch.float32)
                rtgo = torch.as_tensor(self.reward_to_go(ep_rews), dtype=torch.float32)
                batch_rtgo.extend(rtgo)
                values = self.value(obs_tensor_batch).detach() # (B, )
                # batch_advs.extend(returns - values) # adv(t) = R(t) - V(st)
                # batch_advs.extend(self.gae(rtgo, values))
                batch_advs.extend(self.gae(ep_rews, values))


                obs, _ = self.env.reset()
                ep_rews = []

                if len(batch_obs) >= self.batch_size:
                    break

        batch_obs = batch_obs[:self.batch_size]
        batch_acts = batch_acts[:self.batch_size]
        batch_advs = batch_advs[:self.batch_size]

        return batch_obs, batch_acts, batch_advs, batch_rets, batch_len

    # train one epoch
    def train_one_epoch(self):
        batch_obs, batch_acts, batch_advs, batch_rets, batch_len = self.collect_trajectory()

        obs_tensor = torch.as_tensor(batch_obs, dtype=torch.float32)
        acts_tensor = torch.as_tensor(batch_acts, dtype=torch.float32)
        advs_tensor = torch.as_tensor(batch_advs, dtype=torch.float32)

        # policy update
        self.policy_optimizer.zero_grad()
        policy_loss = self.policy.compute_loss(obs_tensor, acts_tensor, advs_tensor)
        policy_loss.backward()
        self.policy_optimizer.step()

        # value update
        self.value_optimizer.zero_grad()
        returns_tensor = advs_tensor + self.value(obs_tensor).detach()  # R_t = adv + V(s_t)
        value_loss = self.value.compute_loss(obs_tensor, returns_tensor)
        value_loss.backward()
        self.value_optimizer.step()

        return policy_loss.item(),value_loss.item(), batch_rets, batch_len

    # training loop
    def train(self, epochs=50):
        for i in range(epochs):
            p_loss, v_loss, rets, lens = self.train_one_epoch()
            print(f"Epoch {i:3d} \t p loss: {p_loss:.3f} \t v loss: {v_loss:.3f}"
                  f"\t return: {np.mean(rets):.3f} "
                  f"\t ep_len: {np.mean(lens):.3f}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', '--env', type=str, default='CartPole-v0')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--lr', type=float, default=1e-2)
    args = parser.parse_args()
    print('\nUsing reward-to-go formulation of policy gradient.\n')

    ac = ActorCritic(env_name=args.env_name, lr=args.lr)
    # ac = ActorCritic(env_name='Pendulum-v1', lr=args.lr, policy_type="Gaussian")
    ac.train()
