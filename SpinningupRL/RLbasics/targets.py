from collections import defaultdict, Counter
"""
Standalone libraries for targets used in Value function approximation for Policy Gradient methods.
The input of each function is assumed to be list of (state, action, reward, next_state) tuples pairs
The Target in value function approximation is the substitute of "true value function" that the gradient descent
optimizes toward.

Notions: Episode is a list that contains (state, action, reward, next_state) tuples
"""

def discounted_reward(episode, gamma=0.99):
    """For MonteCarlo, the target is just Gt = Discount reward!"""
    Gt = 0
    target = []
    discount = 1
    for ep in reversed(episode):
        reward = ep[2]
        Gt += discount * reward
        discount *= gamma
        target.append(Gt)
    return target[::-1]


def monte_carlo_target(episode, gamma=0.99):
    return discounted_reward(episode, gamma)


def td_0_target(reward, gamma, valuefunction):
    """input is the reward, gamma, and valuefunctions VALUE of the next_state"""
    return reward + gamma*valuefunction


def td_lambda_target(episode, value_function, td_lambda=0.1, gamma=0.99):
    """forward view td(lambda) target. This computes the target:
        G_t^lambda at time t being the FIRST element (tuple (s,a,r,s')) of episode.
        This is a n-step TD with the n being the length of the input episode."""
    Gt_lambda = 0
    Gt = 0
    discount = 1
    reward_n = 0
    for n in range(len(episode)):
        ep = episode[n]
        state = ep[0]
        reward = ep[2]
        next_state = ep[3]
        #print(reward, gamma, value_function(next_state))
        Gt_n = reward_n + discount * td_0_target(reward, gamma,
                                                 value_function(next_state))
        print("Gt_", n, "= ", str(Gt_n), "td0: ", td_0_target(reward, gamma, value_function(next_state)))
        #print("reward_n ", reward_n, "discount", discount)

        reward_n += discount * reward
        discount *= gamma
        Gt_lambda += td_lambda**n * Gt_n
        print("Gt_lambda at", n, Gt_lambda)
    Gt_lambda *= (1-td_lambda)
    return Gt_lambda


def td_lambda_target_backward(episode, value_function, td_lambda=0.1, gamma=0.99):
    """Similar to td_lambda_target but BACKWARDS TD(lambda). the LAST ELEMENT of input episode is the
    G_t time point the return point towards. Returns the TD-error times eligibility trace for each step (state)
    V(s) <- V(s) + step * TD_error*Eligibility_trace    (last component returned)"""
    eligibility_trace = defaultdict(float)
    td_error_eligibility = []
    for ep in episode:
        state = ep[0]
        action = ep[1]
        reward = ep[2]
        next_state = ep[3]
        delta_t = reward + gamma*value_function(next_state) - value_function(state)
        eligibility_trace[tuple(state)] += 1
        for key, value in eligibility_trace.items():
            eligibility_trace[key] *= td_lambda * gamma
        td_error_eligibility.append(delta_t*eligibility_trace[tuple(state)])
    return td_error_eligibility

class MonteCarlo:
    """ This is the monte-carlo target. it is EVERY-VISIT. It saves the Increment counter
    and therefore if you want to re-use it inside your code you need to call the reset() function of the static class"""
    N_v = Counter()  # Increment counter for state
    N0 = 100
    Gt_ep = []
    @staticmethod
    def setGt(episode, gamma=0.9):
        Gt_ep = discounted_reward(episode, gamma)

    @staticmethod
    def Vtargets(episode, gamma=0.9):
        if not MonteCarlo.Gt_ep:
            MonteCarlo.Gt_ep = discounted_reward(episode, gamma)
        for i in range(len(episode)):
            state = episode[i][0]

    @staticmethod
    def Vtarget(sats, gamma=0.9):
        state = sats[0]



    @staticmethod
    def reset():
        MonteCarlo.N = defaultdict(int)
def reset():
    N = defaultdict(int)


if __name__ == "__main__":
    episode =[(1,1,1,2), (2,2,2,1), (1,2,3,1)]
    print("testing MC....")
    def val(something):
        return something*0.5
    print("Gt = ", td_lambda_target(episode, val) )


    print("testing TD(0)....")
    print("testing TD(lambda) forward....")
    print("testing TD(lambda) backward....")
