import numpy as np
from collections import defaultdict
import random
import gymnasium as gym

def mc_epsilon_greedy(env, num_episode, gamma=0.99, epsilon=0.3, alpha=0.5):
    Q = defaultdict(lambda: np.ones(env.action_space.n)*0.1)
    
    for episode in range(num_episode):
        state, _ = env.reset()
        state = tuple(state) if isinstance(state, (np.ndarray, list)) else state
        episode_data = []
        done = False
        
        while not done:
            # ε-greedy action selection
            if random.random() < epsilon:
                action = env.action_space.sample()  # Explore
            else:
                action = np.argmax(Q[state])  # Exploit
                
            next_state, reward, done, truncated, _ = env.step(action)
            next_state = tuple(next_state) if isinstance(next_state, (np.ndarray, list)) else next_state
            
            episode_data.append((state, action, reward))
            state = next_state
            
        # Compute returns and update Q
        G = 0
        for state, action, reward in reversed(episode_data):
            G = gamma * G + reward
            # first-visit MC: update only if first occurrence in episode
            if not any(s == state and a == action for s, a, _ in episode_data[:-1]):
                Q[state][action] += alpha * (G - Q[state][action])
                
    # Derive final policy
    policy = {s: np.argmax(a_vals) for s, a_vals in Q.items()}
    return Q, policy


# ---- RUNNING THE ALGORITHM ----
env = gym.make("FrozenLake-v1", is_slippery=False)
Q, policy = mc_epsilon_greedy(env, num_episode=10000)

print("Learned Policy:")
for s in sorted(policy.keys()):
    print(f"State {s}: Action {policy[s]}")

env.close()
