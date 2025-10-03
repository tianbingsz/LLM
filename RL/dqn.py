
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import gymnasium as gym

from policy import MLP


# ========= Replay Buffer =========
class ReplayBuffer:
    """
    Simple FIFO experience replay for DQN (discrete actions).
    """
    def __init__(self, obs_dim, size):
        self.obs1_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.acts_buf = np.zeros(size, dtype=np.int64) # discrete action id
        self.rews_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32) # 1.0 if done else 0.0
        self.ptr, self.size, self.max_size = 0, 0, size


    def store(self, obs, act, rew, next_obs, done):
        self.obs1_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)


    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(
            obs1=self.obs1_buf[idxs],
            obs2=self.obs2_buf[idxs],
            acts=self.acts_buf[idxs],
            rews=self.rews_buf[idxs],
            done=self.done_buf[idxs],
        )




# ========= Q-Network using your MLP =========
class QNetwork(nn.Module):
    """
        Q(s, a)
    """
    def __init__(self, obs_dim, hidden_sizes, act_dim, out_activation=None):
        super().__init__()
        self.q = MLP(obs_dim, hidden_sizes, act_dim, out_activation)


    def get_action(self, obs: torch.Tensor):
        """
            a = argmax_{a'} Q(s, a')
            obs: (b=1, o_dim)
            output: (b=1, 1)
        """
        q_values = self.q(obs) # (1, a_dim)
        action = torch.argmax(q_values, dim=-1)
        return int(action.item())


    def forward(self, obs: torch.Tensor):
        return self.q(obs)  # (b, act_dim)


# ========= DQN Trainer (mirrors your VPG style) =========
class DQN:
    def __init__(
        self,
        env_name="CartPole-v1",
        hidden_sizes=128,              # single hidden width to match your MLP signature
        out_activation=None,           # Q-values should be linear (no activation)
        buffer_size=int(1e6),
        gamma=0.99,
        lr=1e-3,
        batch_size=128,
        min_replay_history=20000,      # warmup before training
        update_period=4,               # train every N env steps
        target_update_period=8000,     # hard target sync period
        seed=0,
    ):
        self.env = gym.make(env_name)

        # Seeding
        np.random.seed(seed)
        torch.manual_seed(seed)
        try:
            self.env.reset(seed=seed)
        except TypeError:
            pass  # older gym versions

        # Spaces
        self.obs_dim = self.env.observation_space.shape[0]
        assert hasattr(self.env.action_space, "n"), "DQN requires discrete action space."
        self.act_dim = self.env.action_space.n

        # Q networks
        self.q = QNetwork(self.obs_dim, hidden_sizes, self.act_dim, out_activation)
        self.q_targ = QNetwork(self.obs_dim, hidden_sizes, self.act_dim, out_activation)
        self.q_targ.load_state_dict(self.q.state_dict())

        # Optimizer
        self.optimizer = Adam(self.q.parameters(), lr=lr)

        # Replay buffer
        self.replay = ReplayBuffer(obs_dim=self.obs_dim, size=buffer_size)

        # Hyperparams
        self.gamma = gamma
        self.lr = lr
        self.batch_size = batch_size
        self.min_replay_history = min_replay_history
        self.update_period = update_period
        self.target_update_period = target_update_period

        # Bookkeeping
        self.total_steps = 0

        # Print param counts
        q_params = sum(p.numel() for p in self.q.parameters() if p.requires_grad)
        print(f"DQN initialized — parameters: {q_params}")


    def compute_loss(self, obs1: torch.Tensor, obs2: torch.Tensor, acts: torch.Tensor, rews: torch.Tensor, done):
        """
            y = r(s,a,s') + gamma * Q_{target}(s', a')
            L = MSE(y, Q(s, a))
        """
        # Q(s,a)
        q_values = self.q(obs1) # (b, A)
        q_sa = q_values[torch.arange(len(acts)), acts] # (b, )

        # a = max_a' Q_target(s', a')
        with torch.no_grad():
            q_next = self.q_targ(obs2)
            q_next_max = torch.max(q_next, dim=-1)[0] # (b, )
            backup = rews + self.gamma * (1.0 - done) * q_next_max

        # Huber loss
        loss = F.smooth_l1_loss(q_sa, backup)

        return loss, q_sa



    def _train_step(self):
        # sample a batch
        batch = self.replay.sample_batch(self.batch_size)
        obs1 = torch.as_tensor(batch["obs1"], dtype=torch.float32) # (b, o_d)
        obs2 = torch.as_tensor(batch["obs2"], dtype=torch.float32) # (b, o_d)
        acts = torch.as_tensor(batch["acts"], dtype=torch.long) # (b, 1)
        rews = torch.as_tensor(batch["rews"], dtype=torch.float32) # (b, )
        done = torch.as_tensor(batch["done"], dtype=torch.float32)


        self.optimizer.zero_grad()
        loss, q_sa = self.compute_loss(obs1, obs2, acts, rews, done)
        loss.backward()
        self.optimizer.step()

        return loss.item(), q_sa.detach().numpy()

    def train(self, steps_per_epoch=5000, epochs=100, max_ep_len=1000, log_freq=1000):
        obs, _ = self.env.reset()
        ep_ret, ep_len, done = 0.0, 0, False

        total_steps = steps_per_epoch * epochs
        for t in range(total_steps):
            # Act
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
            a = self.q.get_action(obs_tensor)

            # Step env
            next_obs, r, terminated, truncated, _ = self.env.step(a)
            done = terminated or truncated
            ep_ret += r
            ep_len += 1

            # Store
            self.replay.store(obs, a, r, next_obs, done and (ep_len < max_ep_len))
            obs = next_obs

            # Episode end handling
            if done or (ep_len >= max_ep_len):
                obs, _ = self.env.reset()
                ep_ret, ep_len, done = 0.0, 0, False

            # Learner updates
            if (self.replay.size > self.min_replay_history) and (self.total_steps % self.update_period == 0):
                loss, q_vals = self._train_step()
                if self.total_steps % log_freq == 0:
                    print(f"[step {self.total_steps:7d}] loss={loss:.4f} q_mean={np.mean(q_vals):.3f} buffer={self.replay.size}")

            # Target sync
            if self.total_steps % self.target_update_period == 0:
                self.q_targ.load_state_dict(self.q.state_dict())


            self.total_steps += 1


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', '--env', type=str, default='CartPole-v1')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--lr', type=float, default=1e-2)
    args = parser.parse_args()
    print('\nUsing reward-to-go formulation of policy gradient.\n')

    ql = DQN(env_name=args.env_name, lr=args.lr)
    ql.train()
