from collections import namedtuple
from components.episode_buffer import ReplayBuffer, OfflineBuffer
import numpy as np
import torch

RewardBatch = namedtuple('Batch', 'trajectories conditions returns')
Batch = namedtuple('Batch', 'trajectories conditions')
ValueBatch = namedtuple('ValueBatch', 'trajectories conditions values')

class SequenceDataset(torch.utils.data.Dataset):
    def __init__(self, arg_path, horizon=20, termination_penalty = 0, use_padding = True, discount=0.99, include_returns = False):
        # self.preprocess_fn = get_preprocess_fn(preprocess_fns, env)
        # self.env = env = load_environment(env)
        self.horizon = horizon
        self.termination_panalty = termination_penalty
        self.buffer = OfflineBuffer(arg_path, self.termination_panalty)
        self.max_path_length = self.buffer.get_max_path_length()
        self.use_padding = use_padding
        self.discount = discount
        self.discounts = self.discount ** np.arange(self.max_path_length)[:, None]
        self.observation_dim = self.buffer.get_observation_dim()
        self.action_dim = self.buffer.get_actions_dim()
        self.path_lengths = self.buffer.path_lengths
        self.include_returns = include_returns

        self.indices = self.make_indices(self.buffer.path_lengths, horizon)
        

        
        self.path_lengths = self.buffer.path_lengths

        print(self.buffer)

    def make_indices(self, path_lengths, horizon):
        '''
            makes indices for sampling from dataset;
            each index maps to a datapoint
        '''
        indices = []
        for i, path_length in enumerate(path_lengths):
            max_start = min(path_length.item() - 1, self.max_path_length - horizon)
            if not self.use_padding:
                max_start = min(max_start, path_length - horizon)
            for start in range(max_start):
                end = start + horizon
                indices.append((i, start, end))
        indices = np.array(indices)
        return indices
    
    def get_conditions(self, observations):
        '''
            condition on current observation for planning
        '''
        return {0: observations[0]}
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx, eps=1e-4):
        path_ind, start, end = self.indices[idx]

        observations = self.buffer.buffer['obs'][path_ind, start:end]
        actions = self.buffer.buffer['actions'][path_ind, start:end]

        conditions = self.get_conditions(observations)
        trajectories = np.concatenate([actions, observations], axis=-1)
        
        if self.include_returns:
            rewards = self.buffer.buffer['reward'][path_ind, start:]
            discounts = self.discounts[:len(rewards)]
            returns = (discounts * rewards).sum()
            returns = np.array([returns], dtype=np.float32)
            batch = RewardBatch(trajectories, conditions, returns)
        else:
            batch = Batch(trajectories, conditions)



        return batch