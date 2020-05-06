import random
import math
import numpy as np
import copy
# Environment


def draw(init=False):
    """Sample with replacement: Drawing one card (say red 10) does not mean the same card can't be drawn again.
    Output is a dict with element:
    * integer sampled by an uniform distribution (1 to 10): number
    * string with either 'red' or 'black' (sampled 1/3 & 2/3 probability): color """
    random.seed()
    card = dict()
    card['number'] = random.choices([*range(1, 11)], [0.1]*10)[0]
    if init:
        card['color'] = 'black'
    else:
        card['color'] = random.choices(['red', 'black'], [1 / 3, 2 / 3])[0]
    return card


def add_card(card):
    return -1 * card['number'] if card['color'] == 'red' else card['number']


def check_card_sum(state_player):
    """return true if busted"""
    if state_player > 21 or state_player < 1:
        return True
    return False


def step(in_state, action):
    """state: dict with element
              dealer = initial card of dealer
              player = sum of players card
        TERMINAL STATE is boolean False.
        action:string of 'hit' or 'stick' """
    state = copy.deepcopy(in_state)
    if action == "stick":
        while 1 <= state['dealer'] < 17:
            # print("Before ",  state['dealer'])
            state['dealer'] += add_card(draw())
            # print("After ", state['dealer'])
        if check_card_sum(state['dealer']):
            return False, 1
        elif state['dealer'] > state['player']:
            return False, -1
        elif state['player'] > state['dealer']:
            return False, 1
        elif state['player'] == state['dealer']:
            return False, 0
    else:
        state['player'] += add_card(draw())
        if check_card_sum(state['player']):
            return False, -1
        return state, 0


def initialize_game():
    state = dict()
    state['player'] = add_card(draw(True))
    state['dealer'] = add_card(draw(True))
    return state


def state_as_tuple(state):
    return tuple([state['player'], state['dealer']])


def play(agent, agent_learn=None):
    state = initialize_game()
    reward = 0
    episodes = []
    while state:
        sars = [state_as_tuple(state)]
        action = agent(state_as_tuple(state))
        sars.append(action)
        state, reward = step(state, action)
        if agent_learn:
            if not state:
                agent_learn(state, reward)
            else:
                agent_learn(state_as_tuple(state), reward)
        sars.append(reward)
        if state:
            sars.append(state_as_tuple(state))
        episodes.append(sars)
    return episodes


def get_state_space():
    state_space = dict()
    for i in range(1, 22):
        state_space[i] = [*range(1, 11)]
    return state_space


def get_state_space_np():
    """return the X element of 3D plot. just X.T for the Y"""
    X = np.array([*range(1, 11)])
    Y = np.array([*range(1, 22)])
    X, Y = np.meshgrid(X, Y)
    return X, Y

if __name__ == '__main__':
    def stickAgent(state):
        return 'stick'
    def hitAgent(state):
        return 'hit'
    print("for stickAgent: ")
    for i in range(0, 5):
        print(play(stickAgent))
    print("for hitAgent: ")
    for i in range(0, 5):
        print(play(hitAgent))
