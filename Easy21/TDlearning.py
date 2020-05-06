from Easy21.BaseAgent import BaseAgent
from Easy21 import easy21
import random
from Easy21.MonteCarlo import MonteCarloAgent
from collections import defaultdict, Counter
import matplotlib.pyplot as plt


class TDlearning(BaseAgent):
    def __init__(self, td_l=1.0):
        super().__init__()
        self.td_lambda = td_l

    def set_lambda(self, l):
        self.td_lambda = l

    def online_training(self, init_state, step=easy21.step):
        state = init_state
        interrupt = False
        eligibility_trace = defaultdict(float)
        action = self.take_action(easy21.state_as_tuple(state))

        while not interrupt:
            self.Nsa[tuple([easy21.state_as_tuple(state), action])] += 1
            next_state, reward = step(state, action)

            q_hat = 0
            next_action = ""
            if next_state:
                next_action = self.take_action(easy21.state_as_tuple(next_state))
                q_hat = self.q_value[tuple([easy21.state_as_tuple(next_state), next_action])]
            delta = reward + self.gamma * q_hat - self.q_value[tuple([easy21.state_as_tuple(state), action])]
            eligibility_trace[tuple([easy21.state_as_tuple(state), action])] += 1
            for (st, ac), value in eligibility_trace.items():
                self.q_value[st, ac] += self.alpha_t(st, ac) * delta * value
                eligibility_trace[st, ac] *= self.gamma * self.td_lambda
            action = next_action
            state = next_state
            if not next_state:
                interrupt = True  # terminal state

    def take_action(self, state):
            self.Ns[state] += 1
            return self.epsilon_greedy(state)

    def msq_error(self, q_star: defaultdict):
        msqe = 0
        total_num = 0
        for key, value in q_star.items():
            msqe += (self.q_value[key] - value) ** 2
            total_num += 1
        return msqe/total_num



if __name__ == "__main__":
    mc_agent = MonteCarloAgent()
    mc_agent.load_qvalue("MC_Qval.pkl")
    Q_star = mc_agent.get_qvalue()
    td = [0.1*a for a in range(0, 11)]
    td_lambda = [TDlearning(t) for t in td]
    for TD in td_lambda:
        TD.set_state_space(easy21.get_state_space())
        TD.set_action_space(['hit', 'stick'])
    num_iter = 10000
    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2)

    ax.set_title("0.1 to 0.9 lambda")
    ax.set_ylabel("MSE")
    ax.set_xlabel("lambda")

    ax2.set_title("MSE for lambda 0 and 1")
    ax2.set_ylabel("MSE")
    ax2.set_xlabel("Episode")
    MSQE_tot = []
    for i in range(1, num_iter+1):

        for TD in td_lambda:
            start_state = easy21.initialize_game()
            TD.online_training(start_state, easy21.step)

        if i % 10 == 0:
            MSQE = [TD.msq_error(Q_star) for TD in td_lambda]
            if not MSQE_tot:
                MSQE_tot = [[] for MS in MSQE]

            [MSQE_tot[j].append(MSQE[j]) for j in range(len(MSQE))]
            #ax2.scatter([i] * len(MSQE), MSQE_tot)
            ax.plot(td, MSQE)
            ax2.scatter([i, i], [MSQE[0], MSQE[-1]], color=['r', 'b'])
            plt.show()
            plt.pause(0.01)

    plt.savefig("TDresult.png")

