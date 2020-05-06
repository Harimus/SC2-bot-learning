from collections import defaultdict, Counter
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
import pickle


class BaseAgent:
    def __init__(self):
        """value_function: key = tuple([player state, dealer state])"""
        self.value_function = defaultdict(float)
        self.q_value = defaultdict(float)
        self.Ns = Counter()  # increases with play
        self.Nsa = Counter()  # increases with train
        self.N0 = 100
        self.state_space = dict()
        self.action_space = []
        self.gamma = 1
    def reset(self):
        self.value_function = defaultdict(float)
        self.q_value = defaultdict(float)
        self.Ns = Counter()
        self.Nsa = Counter()

    def epsilon_t(self, state):
        return self.N0 / (self.N0 + self.Ns[state])

    def alpha_t(self, state, action):
        return 1 /(self.Nsa[tuple([state, action])])

    def alpha_t_key(self, stateaction):
        return 1 /(self.Nsa[stateaction])

    def set_state_space(self, state_space):
        """dict of int with key int. Assuming 3D space for now"""
        self.state_space = state_space

    def set_action_space(self, action_space):
        """"""
        self.action_space = action_space

    def save_qvalue(self, filename):
        with open(filename, 'wb') as file:
            pickle.dump(self.q_value, file, pickle.HIGHEST_PROTOCOL)

    def load_qvalue(self, filename):
        with open(filename, 'rb') as file:
            self.q_value = pickle.load(file)

    def get_qvalue(self):
        return self.q_value

    def epsilon_greedy(self, state):
        choice = random.choices(['greedy', 'random'],
                                [self.epsilon_t(state), 1 - self.epsilon_t(state)])[0]
        if choice == 'greedy':
            best_action = ''
            best_val = float('-inf')
            for action in self.action_space:
                q_val = self.q_value[tuple([state, action])]
                if q_val > best_val:
                    best_val = q_val
                    best_action = action
            return best_action
        elif choice == 'random':
            return random.choice(self.action_space)
        else:
            print('fail')

    def plot_vf(self, X, Y):
        Z = np.zeros(X.shape)
        i = 0
        for key, value in self.state_space.items():
            j = 0
            for val in value:
                maxval = float('-inf')
                for action in self.action_space:
                    action_val = self.q_value[tuple([tuple([key, val]), action])]
                    if action_val > maxval:
                        maxval = action_val

                Z[i, j] = maxval
                j += 1
            i += 1
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X=X, Y=Y, Z=Z)
        print(self.q_value)
        plt.show()

