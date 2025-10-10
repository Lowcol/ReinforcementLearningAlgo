import numpy as np

def policy_iteration(states, actions, P, R, gamma=0.9, theta=1e-6):
    """
    states: list of states
    action: list of actions
    P: transition probabilities dict P[s][a] = [(prob, next_state), ...]
    R: reward dict R[s][a][s'] = reward
    gamma: discount factor
    theta: convergence threshold
    """
    #Step 0: initialize policy randomly and value function
    
    policy = np.zeros(len(states), dtype=int) #initialize policy
    V = np.zeros(len(states)) #initialize value function
    
    stable = False
    while not stable:
        #1. Policy Evaluation
        while True:
            delta = 0
            for s in states:
                v = V[s]
                V[s] = max(
                    sum(prob*(R[s][a][s_next]+gamma*V[s_next])
                        for prob, s_next in P[s][a])
                    for a in actions
                )
                delta = max(delta, abs(v-V[s]))
            if delta < theta:
                break
 
        # 2. Policy Improvement
        stable = True
        for s in states:
            old_action = policy[s]
            q_values = []
            for a in actions:
                q = sum(prob*(R[s][a][s_next]+gamma*V[s_next])
                    for prob, s_next in P[s][a])
                q_values.append(q)
            best_action = np.argmax(q_values)
            policy[s] = best_action
            if old_action != best_action:
                stable = False
                
    return V, policy
        
# Example MDP
states = [0, 1, 2]
actions = [0, 1]

P = {
    0: {0: [(1.0, 0)], 1: [(1.0, 1)]},
    1: {0: [(1.0, 0)], 1: [(1.0, 2)]},
    2: {0: [(1.0, 2)], 1: [(1.0, 2)]}
}

R = {
    0: {0: {0: 0}, 1: {1: 5}},
    1: {0: {0: 0}, 1: {2: 10}},
    2: {0: {2: 0}, 1: {2: 0}},
}

V, policy = policy_iteration(states, actions, P, R)

print("Optimal Value Function: ", V)
print("Optimal Policy:", policy)
