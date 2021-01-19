import torch
from torch import autograd
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset
import gym
from logging import ERROR


def flatten_list_dicts(list_dicts):
    return {k: torch.cat([d[k] for d in list_dicts], dim=0) for k in list_dicts[-1].keys()}


class TransitionDataset(Dataset):
    def __init__(self, transitions):
        super().__init__()
        self.states, self.actions, self.rewards, self.terminals = transitions['states'], transitions['actions'].detach(), transitions['rewards'], transitions['terminals']  # Detach actions

    # Allows string-based access for entire data of one type, or int-based access for single transition
    def __getitem__(self, idx):
        if isinstance(idx, str):
            if idx == 'states':
                return self.states
            elif idx == 'actions':
                return self.actions
            elif idx == 'terminals':
                return self.terminals
        else:
            return dict(states=self.states[idx], actions=self.actions[idx], rewards=self.rewards[idx], next_states=self.states[idx + 1], terminals=self.terminals[idx])

    def __len__(self):
        return self.terminals.size(0) - 1  # Need to return state and next state


gym.logger.set_level(ERROR)  # Ignore warnings from Gym logger


class CartPoleEnv():
    def __init__(self):
        self.env = gym.make('CartPole-v1')

    def reset(self):
        state = self.env.reset()
        return torch.tensor(state, dtype=torch.float32).unsqueeze(dim=0)  # Add batch dimension to state

    def step(self, action):
        state, reward, terminal, _ = self.env.step(action[0].detach().numpy())  # Remove batch dimension from action
        return torch.tensor(state, dtype=torch.float32).unsqueeze(
            dim=0), reward, terminal  # Add batch dimension to state

    def seed(self, seed):
        return self.env.seed(seed)

    def close(self):
        self.env.close()

    @property
    def observation_space(self):
        return self.env.observation_space

    @property
    def action_space(self):
        return self.env.action_space
