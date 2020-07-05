import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.optim import Adam
import gym
from SpinningupRL.RLbasics.targets import monte_carlo_target
import matplotlib.pyplot as plt

def nn_policy(layer_sizes, activation_function=nn.Tanh, output_activation=nn.Identity):
    nn_layers = []
    for j in range(len(layer_sizes)-1):
        activation = activation_function if j < len(layer_sizes)-2 else output_activation
        nn_layers += [nn.Linear(layer_sizes[j], layer_sizes[j+1]), activation()]
    return nn.Sequential(*nn_layers) # Star "unpacks" the list, in to args for the function


class REINFORCE:

    def __init__(self, learning_rate=1e-3, hidden_layer=[30, 20]):
        self.lr = learning_rate
        self.hl = hidden_layer
        self.policy = None
        self.policy_gradient = None
        self.policy_log_gradient = None #score function
        self.weight = None
        self.baseline = None # normal REINFORCE jut use the Gt
        self.gamma = 0.99
        self.env = None
        self.optimizer = None # type: Adam

    def set_env(self, environment):
        self.env = environment
        observation_space = env.observation_space.shape[0]
        action_space = env.action_space.n
        self.policy = nn_policy(layer_sizes=[observation_space] + self.hl + [action_space])

    def get_policy(self, obs):
        return Categorical(logits=self.policy(obs))

    def get_action(self, obs):
        return self.get_policy(obs).sample().item()

    def compute_loss(self, obs, act, weights):
        log_probability = self.get_policy(obs).log_prob(act)
        return -(log_probability * weights).mean()

    def train_episode(self, episode):
        states = [ep[0] for ep in episode]
        actions = [ep[1] for ep in episode]
        weights = monte_carlo_target(episode)
        self.optimizer.zero_grad()
        batch_loss = self.compute_loss(obs=torch.as_tensor(states, dtype=torch.float32),
                          act=torch.as_tensor(actions, dtype=torch.int32),
                          weights=torch.as_tensor(weights, dtype=torch.float32))
        batch_loss.backward()
        self.optimizer.step()

    def train(self, env, trajectory=1000):
        episode_sum_reward = []
        self.optimizer = Adam(self.policy.parameters(), lr=self.lr)
        percent = 0
        plt.ion()
        fig, ax = plt.subplots()
        ax.set_title(env.spec._env_name+" total reward per episode")
        ax.set_ylabel("reward")
        ax.set_xlabel("episodes")
        for i in range(trajectory):
            episode = []
            done = False
            state = env.reset()
            action = self.get_action(torch.as_tensor(state, dtype=torch.float32))
            next_state, reward, done, _ = env.step(action)
            episode.append((state, action, reward, next_state))
            state = next_state
            if i % (trajectory/100) == 0:
                print(percent, "% done")
                percent+=1
            while not done:
                if i % (trajectory/5) == 0:
                    env.render()
                action = self.get_action(torch.as_tensor(state, dtype=torch.float32))
                next_state, reward, done, _ = env.step(action)
                episode.append((state, action, reward, next_state))
                state = next_state
            self.train_episode(episode)
            sum_reward = sum([ep[2] for ep in episode])
            episode_sum_reward.append(sum_reward)
            #if i % (trajectory / 10) == 0:
            #    ax.plot([*range(len(episode_sum_reward))], episode_sum_reward)
            #    fig.canvas.draw()
            #    fig.canvas.flush_events()
            #    #plt.show()
            #    plt.pause(0.05)
        ax.plot([*range(len(episode_sum_reward))], episode_sum_reward)
        plt.show()




if __name__ == "__main__":
    agent = REINFORCE()
    env = gym.make('MountainCar-v0')
    agent.set_env(env)
    agent.policy.load_state_dict(torch.load("./MCNN"))
    for i in range(10):
        agent.train(env)
        torch.save(agent.policy.state_dict(), "./MCNN")
