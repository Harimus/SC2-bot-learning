import SpinningupRL.policy
import torch
from util import TransitionDataset, flatten_list_dicts, CartPoleEnv
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.optim import Adam
import torch.nn as nn
import matplotlib.pyplot as plt

# Performs a behavioural cloning update
def behavioural_cloning_update(agent, expert_trajectories, agent_optimiser, batch_size):
    expert_dataloader = DataLoader(expert_trajectories, batch_size=batch_size, shuffle=True, drop_last=True)
    loss = nn.NLLLoss()
    for expert_transition in expert_dataloader:
        expert_state, expert_action = expert_transition['states'], expert_transition['actions']

        agent_optimiser.zero_grad()
        pi, log_prob = agent.pi(expert_state, expert_action)  # Maximum likelihood objective
        #behavioural_cloning_loss = (-log_prob).mean()
        behavioural_cloning_loss =loss(pi.logits, expert_action)
        behavioural_cloning_loss.backward()
        agent_optimiser.step()
    return behavioural_cloning_loss.item()

fig, ax = plt.subplots()
def BC(episodes=1000, actor_lr=1e-3, batch_size=128):

    test_interval = episodes / 100
    env = CartPoleEnv()
    action_space, observation_space = env.action_space.n, env.observation_space.shape[0]
    agent = SpinningupRL.policy.ActorCriticPolicy(observation_space, action_space)
    actor_optimizer = Adam(agent.pi.parameters())
    expert_trajectories = TransitionDataset(flatten_list_dicts(torch.load('expert_trajectories.pth')))
    pbar = tqdm(range(1, episodes+1), unit_scale=1, smoothing=0)
    tot_reward_plot = []
    plot_step = []
    total_reward = 0
    for i in pbar:
        current_loss = behavioural_cloning_update(agent, expert_trajectories, actor_optimizer, batch_size)
        if i % test_interval == 0:
            done = False
            state = env.reset()
            total_reward = 0
            while not done:
                action = agent.act(state)
                next_state, reward, done = env.step(torch.as_tensor(action))
                total_reward += reward
                state = next_state
            plot_step.append(i)
            tot_reward_plot.append(total_reward)
        pbar.set_description("Step: %i | Reward: %f | Loss: %f" % (i, total_reward, current_loss))






if __name__ == "__main__":
    agent = BC()
