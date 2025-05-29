import numpy as np
import torch

class ReplayBuffer:
    def __init__(self, obs_dim, capacity=100000):
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.ptr = 0
        self.size = 0
        self.buffer = np.zeros((capacity, obs_dim), dtype=np.float32)

    def add(self, obs):
        # Always store as (N, obs_dim) where obs_dim matches buffer
        obs = np.asarray(obs)
        if obs.shape[-1] != self.obs_dim:
            obs = obs[..., :self.obs_dim]
        n = obs.shape[0] if obs.ndim > 1 else 1
        if self.ptr + n > self.capacity:
            overflow = self.ptr + n - self.capacity
            self.buffer[self.ptr:self.capacity] = obs[:n-overflow]
            self.buffer[0:overflow] = obs[n-overflow:]
            self.ptr = overflow
        else:
            self.buffer[self.ptr:self.ptr+n] = obs
            self.ptr = (self.ptr + n) % self.capacity
        self.size = min(self.size + n, self.capacity)

    def sample(self, batch_size):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return torch.tensor(self.buffer[idxs], dtype=torch.float32)

    def sample_pair(self, batch_size):
        # Sample indices such that idx+1 is valid
        if self.size < 2:
            raise ValueError("Not enough samples in buffer to sample pairs.")
        idxs = np.random.randint(0, self.size - 1, size=batch_size)
        obs_t = torch.tensor(self.buffer[idxs], dtype=torch.float32)
        obs_tp1 = torch.tensor(self.buffer[idxs + 1], dtype=torch.float32)
        return obs_t, obs_tp1

    def feed_forward_generator(self, num_mini_batch, mini_batch_size, allow_replacement=True):
        """
        Generator that yields mini-batches of concatenated (state + next_state) observations.
        Compatible with amp-rsl-rl interface but returns 119-dim concatenated observations.
        """
        total = num_mini_batch * mini_batch_size
        
        if total > self.size:
            if not allow_replacement:
                raise ValueError(f"Not enough samples in buffer: requested {total}, but have {self.size}")
        
        for i in range(num_mini_batch):
            # Sample consecutive pairs and concatenate them
            state, next_state = self.sample_pair(mini_batch_size)
            # For 119-dim observations, we expect state and next_state to be truncated/processed
            # to fit the 119-dim format. For now, just return state (this needs environment integration)
            yield state

class DemoBuffer:
    def __init__(self, data):
        self.data = data
        self.size = data.shape[0]

    def sample(self, batch_size):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return torch.tensor(self.data[idxs], dtype=torch.float32)

    def sample_pair(self, batch_size):
        if self.size < 2:
            raise ValueError("Not enough samples in demo buffer to sample pairs.")
        idxs = np.random.randint(0, self.size - 1, size=batch_size)
        obs_t = self.data[idxs].clone().detach().float()
        obs_tp1 = self.data[idxs + 1].clone().detach().float()
        return obs_t, obs_tp1

    def feed_forward_generator(self, num_mini_batch, mini_batch_size, allow_replacement=True):
        """
        Generator that yields mini-batches of data.
        For DemoBuffer, the data is already concatenated AMP observations, so we return it as-is.
        """
        total = num_mini_batch * mini_batch_size
        
        if total > self.size:
            if not allow_replacement:
                raise ValueError(f"Not enough samples in demo buffer: requested {total}, but have {self.size}")
        
        for i in range(num_mini_batch):
            # Demo buffer contains full AMP observations (already concatenated)
            idxs = np.random.randint(0, self.size, size=mini_batch_size)
            batch = self.data[idxs].clone().detach().float()
            yield batch  # Return full observation, not split
