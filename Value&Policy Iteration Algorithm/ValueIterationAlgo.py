import numpy as np

def value_iteration(states, actions, P, R, gamma=0.9, theta=1e-6):
    """
    states: list of states
    action: list of actions
    P: transition probabilities dict P[s][a] = [(prob, next_state), ...]
    R: reward dict R[s][a][s'] = reward
    gamma: discount factor
    theta: convergence threshold
    """
    #initialize value function
    V = np.zeros(len(states))
    
    while True:
        delta = 0
        for s in states:
            v = V[s]
            # Bellman update
            V[s] = max(
                sum(prob*(R[s][a][s_next]+gamma*V[s_next])
                    for prob, s_next in P[s][a])
                for a in actions
            )
            delta = max(delta, abs(v-V[s]))
        if delta < theta:
            break
            
    # Extract policy
    policy = np.zeros(len(states), dtype=int)
    for s in states:
        q_values = []
        for a in actions:
            q = sum(prob*(R[s][a][s_next]+gamma*V[s_next])
                for prob, s_next in P[s][a])
            q_values.append(q)
        policy[s] = np.argmax(q_values)
        
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

V, policy = value_iteration(states, actions, P, R)

print("Optimal Value Function: ", V)
print("Optimal Policy:", policy)
