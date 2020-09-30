import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
import gym

"""This file contains the Neural network policies. The reason for this being in an independent file is so that
    All the different Policy Gradient methods can be used with the same policy network, for performance comparison.
    Why? Because one might want to understand the effect of policy network/depth/size on the network."""


def fcnn_policy(layer_sizes, activation_function=nn.Tanh, output_activation=nn.Identity):
    """
    This is a basic fully connected neural network, used as an default for the basic model
    @param layer_sizes: a list of integers that specify each hidden layers size
    @param activation_function: Specify the activation function for the hidden layers.
    @param output_activation: The activation function for the output layer.
    @return: nn.Sequential
    """
    nn_layers = []
    for j in range(len(layer_sizes)-1):
        activation = activation_function if j < len(layer_sizes)-2 else output_activation
        nn_layers += [nn.Linear(layer_sizes[j], layer_sizes[j+1]), activation()]
    return nn.Sequential(*nn_layers) # Star "unpacks" the list, in to args for the function


class BasicPolicy(nn.Module):

    def __init__(self, observation_space, action_space, hidden_layers=(30, 20),  network=fcnn_policy):
        super().__init__()
        self.policy = network([observation_space]+list(hidden_layers)+[action_space])

    def get_policy(self, obs):
        logits = self.policy(obs)
        return Categorical(logits=logits)

    def forward(self, obs: torch.Tensor, act=None):
        pi: Categorical = self.get_policy(obs)

        log_act = None
        if act is not None:
            log_act = pi.log_prob(act)
        return pi, log_act

    def step(self, obs):
        with torch.no_grad():
            pi = self.get_policy(obs)
            a = pi.sample()
            logp_a = pi.log_prob(a)

        return a.numpy(), logp_a.numpy()

    def act(self, obs):
        return self.step(obs)[0]


class ActorCriticPolicy(nn.Module):

    def __init__(self, observation_space, action_space, hidden_layer=(30, 20), actor=BasicPolicy, critic=fcnn_policy):
        super().__init__()
        # The policy, "actor"
        self.pi = actor(observation_space.shape[0], action_space.n)
        # The critic
        self.v = critic([observation_space.shape[0]]+list(hidden_layer)+[1])

    def step(self, obs):
        with torch.no_grad():
            a, log_p = self.pi.step(obs)
            v = self.v(obs)
            return a, log_p, v.numpy()

    def act(self, obs):
        return self.step(obs)[0]


if __name__ == "__main__":
    env = gym.make('MountainCar-v0')
    obs_space = env.observation_space.shape[0]
    act_space = env.action_space.n
    state = env.reset()
    state_as_torch = torch.as_tensor(state, dtype=torch.float32)
    acp = ActorCriticPolicy(obs_space, act_space)
    bp = BasicPolicy(obs_space, act_space)

    action, _, value = acp.step(state_as_torch)
    action2, _ = bp.step(state_as_torch)
    print("ActorCritic: ", action)
    print("Actor: ", action2)
    print("Critic (value func for state): ", value)
    env.step(action)
    env.step(action2)