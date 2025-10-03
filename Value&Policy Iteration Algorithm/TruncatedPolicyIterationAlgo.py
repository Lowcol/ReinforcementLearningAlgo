import numpy as np

def truncated_policy_iteration(states, actions, P, R, gamma=0.9, theta=1e-6, k=3):
    """
    states: list of states
    action: list of actions
    P: transition probabilities dict P[s][a] = [(prob, next_state), ...]
    R: reward dict R[s][a][s'] = reward
    gamma: discount factor
    theta: convergence threshold
    k: number of policy evaluation iteration
    """
    #initialize value function
    policy = np.zeros(len(states), dtype=int)
    V = np.zeros(len(states))
    
    stable = False
    while not stable:
        #1. Truncated Policy Iteration
        for _ in range(k):
            delta = 0
            for s in states:
                v = V[s]
                a = policy[s]
                V[s] = max(
                    sum(prob*(R[s][a][s_next]+gamma*V[s_next])
                        for prob, s_next in P[s][a])
                    for a in actions
                )
                delta = max(delta, abs(v-V[s]))
            if delta < theta:
                break
                
        # 2. Policy Improvements
        stable = True
        for s in states:
            old_actions = policy[s]
            q_values = []
            for a in actions:
                q = sum(prob*(R[s][a][s_next]+gamma*V[s_next])
                    for prob, s_next in P[s][a])
                q_values.append(q)
            best_action = np.argmax(q_values)
            policy[s] = best_action
            if old_actions!=best_action:
                stable=False
            
    return V, policy
        
# Example MDP
states = [0, 1, 2]
actions = [0, 1]

# Transition probabilities
P = {
    0: {
        0: [(1.0, 0)],   # action 0 stays in 0
        1: [(1.0, 1)],   # action 1 goes to 1
    },
    1: {
        0: [(1.0, 0)],
        1: [(1.0, 2)],
    },
    2: {
        0: [(1.0, 2)],
        1: [(1.0, 2)],
    }
}

# Rewards
R = {
    0: {0: {0: 0}, 1: {1: 5}},
    1: {0: {0: 0}, 1: {2: 10}},
    2: {0: {2: 0}, 1: {2: 0}},
}

V, policy = truncated_policy_iteration(states, actions, P, R, k=2)

print("Optimal Value Function: ", V)
print("Optimal Policy:", policy)
