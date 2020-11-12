import torch
from torch.optim import Adam
import gym
from copy import deepcopy
from collections import deque
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.distributions import Distribution
from torch.distributions.normal import Normal
import torch.nn.functional as F
from policy import fcnn_policy
"""Model defined here too, and follows Kaixhin structure rather than Spinningup,
since it is easier to understand"""
class TanhNormal(Distribution):
    """Copied from Kaixhi"""
    def __init__(self, loc, scale):
        super().__init__()
        self.normal = Normal(loc, scale)

    def sample(self):
        return torch.tanh(self.normal.sample())

    # samples with re-parametrization trick (differentiable)
    def rsample(self):
        return torch.tanh(self.normal.rsample())

    # Calculates log probability of value using the change-of-variables technique (uses log1p = log(1 + x) for extra numerical stability)
    def log_prob(self, value):
        inv_value = (torch.log1p(value) - torch.log1p(-value)) / 2  # artanh(y)
        return self.normal.log_prob(inv_value) - torch.log1p(-value.pow(2) + 1e-6)  # log p(f^-1(y)) + log |det(J(f^-1(y)))|

    @property
    def mean(self):
        return torch.tanh(self.normal.mean)


LOG_MIN = -20
LOG_MAX = 2


class SoftActor(nn.module):
    def __init__(self, observation_dim, action_dim, hidden_layer_size=(256,256),
                 activation_function=nn.Tanh, output_activation=nn.Identity):
        super().__init__()
        """The network have output size of action_dim*2 because output of this network are 
        divided in to the mu and log_std component of a gaussian distribution"""
        self.network = fcnn_policy([observation_dim]+list(hidden_layer_size)+[2*action_dim],
                                   activation_function=activation_function, output_activation=output_activation)

    def forward(self, state: torch.Tensor):
        mu, log_std = self.network(state).chunk(2,dim=1)
        log_std = torch.clamp(log_std, LOG_MIN, LOG_MAX) #to make it not too random/deterministic
        normal = TanhNormal(mu, log_std.exp())
        return normal


class Critic(nn.Module):
    def __init__(self, observation_dim, action_dim=None, hidden_layer_size=(256, 256),
                 activation_function=nn.Tanh, output_activation=nn.Identity):
        super().__init__()
        network_architecture = [observation_dim + action_dim if action_dim is not None else 0] + list(hidden_layer_size) + [1]
        self.network = fcnn_policy(network_architecture,
                                   activation_function=activation_function, output_activation= output_activation)

    def forward(self, state, action=None):
        if action is None:
            value = self.network(state)
        else:
            value = self.network(torch.cat([state, action], dim=1))
        return value.squeeze(dim=1)


class SoftActorCritic(nn.Module):

    def __init__(self, observation_dim, action_dim, hidden_layer_size=(256, 256),
                 activation_function=nn.Tanh, output_activation=nn.Identity):
        self.actor = SoftActor(observation_dim=observation_dim, action_dim=action_dim,
                               hidden_layer_size=hidden_layer_size, activation_function=activation_function,
                               output_activation=output_activation)
        self.critic_v = Critic(observation_dim=observation_dim, action_dim=None,
                               hidden_layer_size=hidden_layer_size, activation_function=activation_function,
                               output_activation=output_activation)

        self.critic_q1 = Critic(observation_dim=observation_dim, action_dim=action_dim,
                                hidden_layer_size=hidden_layer_size, activation_function=activation_function,
                                output_activation=output_activation)

        self.critic_q2 = Critic(observation_dim=observation_dim, action_dim=action_dim,
                                hidden_layer_size=hidden_layer_size, activation_function=activation_function,
                                output_activation=output_activation)


def sac(policy=SoftActorCritic, epoch=10000, gamma=0.99, lam=0.97,
        actor_lr=1e-3, critic_lr=1e-3, alpha=0.2, buffer_size=1000,
        initial_exploration=1000):
    envname = 'MountainCarContinuous-v0'
    env = gym.make(envname)
    observation_space = env.observation_space.shape[0]
    action_space = env.action_space.shape[0]
    # Scaling of action
    scaling_factor = env.action_space.high[0] - env.action_space.low[0]
    scaling_const = env.action_space.low[0]
    agent = policy(observation_space, action_space)
    target_agent = deepcopy(agent)

    actor_optimizer = Adam(agent.pi.parameters(), lr=actor_lr)
    critic_q_optimizer = Adam(list(agent.critic_q1.parameters())+list(agent.critic_q2.parameters()), lr=critic_lr)
    critic_v_optimizer = Adam(agent.critic_v.parameters(), lr=critic_lr)
    replay_buffer = deque(maxlen=buffer_size)

    def update(episodes):
        states = torch.as_tensor([ep[0] for ep in episodes], dtype=torch.float32)
        actions = torch.as_tensor([ep[1] for ep in episodes], dtype=torch.float32)
        rewards = torch.as_tensor([ep[2] for ep in episodes], dtype=torch.float32)
        next_states = torch.as_tensor([ep[3] for ep in episodes], dtype=torch.float32)
        is_done = torch.as_tensor([ep[-1] for ep in episodes], dtype=torch.float32)

        target_q = rewards + gamma * (1-is_done) * agent.critic_v(next_states)
        pi = agent.actor(states)
        sample_actions = pi.rsample()
        entropy_component = pi.log_prob(sample_actions).sum(dim=1)
        critic_q = torch.min(agent.critic_q2(states, sample_actions.detach()), agent.critic_q1(states, sample_actions.detach()))
        target_v = critic_q - alpha * entropy_component

        # Update Q
        critic_q_loss = (1/2) * ((agent.critic_q1 - target_q)**2).mean() + \
                        (1/2) * ((agent.critic_q2 - target_q)**2).mean()
        critic_q_optimizer.zero_grad()
        critic_q_loss.backwards()
        critic_q_optimizer.step()

        # Update V

        critic_v_loss = (1/2) * ((agent.critic_v(states) - target_v)**2).mean()
        critic_v_optimizer.zero_grad()
        critic_v_loss.backwards()
        critic_v_optimizer.step()

        # Update Policy
        target_pi = torch.min(agent.critic_q2(states, sample_actions), agent.critic_q1(states, sample_actions))
        loss_pi = (alpha * entropy_component - target_pi).mean()
        actor_optimizer.zero_grad()
        loss_pi.backwards()


if __name__ == '__main__':
    ag = sac()
