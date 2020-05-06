import random
from Easy21.BaseAgent import BaseAgent
from Easy21.easy21 import play, get_state_space_np, get_state_space


class MonteCarloAgent(BaseAgent):
    def __init__(self):
        super().__init__()

    def take_action(self, state):
        self.Ns[state] += 1
        return self.epsilon_greedy(state)

    def train(self, episodes):
        # Episode contains [s,a,r,s]
        Gt = 0
        discount = 1
        gamma = discount
        for ep in episodes[::-1]:
            reward = ep[2]
            Gt += gamma * reward
            gamma *= discount
            ep.append(Gt)
        for ep in episodes:
            Gt = ep[-1]
            self.Nsa[tuple([ep[0], ep[1]])] += 1
            self.q_value[tuple([ep[0], ep[1]])] += self.alpha_t(ep[0], ep[1]) * \
                                                   (Gt - self.q_value[tuple([ep[0], ep[1]])])

        return


if __name__ == "__main__":

    num_iter = 1000000
    mc_agent = MonteCarloAgent()
    mc_agent.set_state_space(get_state_space())
    mc_agent.set_action_space(['hit', 'stick'])
    X, Y = get_state_space_np()
    percentage = num_iter / 100
    p = 1
    print_timing = num_iter / 2
    for i in range(1, num_iter + 1):
        episodes = play(mc_agent.take_action)
        mc_agent.train(episodes)
        if i % print_timing == 0:
            mc_agent.plot_vf(X, Y)
        if i % percentage == 0:
            print(p, "% done...")
            p += 1
    mc_agent.save_qvalue("MC_Qval.pkl")
