import random
from collections import defaultdict

class SimpleEnv:
    def __init__(self):
        self.states = [0, 1, 2, 3]  # 3 is terminal state
        self.terminal_state = 3
        self.actions = [0, 1]       # 2 actions
        self.state = None
        self.reset()
        
    def reset(self, start_state=None):
        # exploring start: random starting state
        self.state = start_state if start_state is not None else random.choice(self.states[:-1])
        return self.state
    
    def step(self, action):
        # define simple deterministic transitions
        if self.state == self.terminal_state:
            return self.state, 0, True
        
        if action == 0:
            next_state = max(0, self.state - 1)
        else:
            next_state = min(3, self.state + 1)
        
        reward = 1 if next_state == self.terminal_state else 0
        done = (next_state == self.terminal_state)
        self.state = next_state
        return next_state, reward, done
    

def monte_carlo_es(env, num_episodes=5000, gamma=1.0):
    Q = defaultdict(lambda: {a: 0.0 for a in env.actions})
    policy = {s: random.choice(env.actions) for s in env.states if s != env.terminal_state}
    returns = defaultdict(list)
    
    for episode in range(num_episodes):
        start_state = random.choice(env.states[:-1])
        start_action = random.choice(env.actions)
        
        episode_data = []
        state = env.reset(start_state)
        action = start_action
        done = False
        
        print(f"\n--- Episode {episode+1} ---")
        print(f"Start state: {state}, start action: {action}")
        
        step_count = 0
        while not done:
            step_count += 1
            next_state, reward, done = env.step(action)
            episode_data.append((state, action, reward))
            state = next_state
            if not done:
                action = policy[next_state]
            if step_count > 50:  # safety cutoff to prevent infinite loop
                print("Episode cutoff at 50 steps (loop likely stuck)")
                done = True
        
        print(f"Episode finished after {step_count} steps.")
        
        G = 0
        visited = set()
        for state, action, reward in reversed(episode_data):
            G = gamma * G + reward
            if (state, action) not in visited:
                returns[(state, action)].append(G)
                Q[state][action] = sum(returns[(state, action)]) / len(returns[(state, action)])
                best_action = max(Q[state], key=Q[state].get)
                policy[state] = best_action
                visited.add((state, action))
        
        if (episode + 1) % 100 == 0:
            print(f"\n✅ Completed {episode+1} episodes")

    return Q, policy


env = SimpleEnv()
Q, policy = monte_carlo_es(env, num_episodes=5000)

print("\nLearned Q-values:")
for s in sorted(Q.keys()):
    print(f"State {s}: {Q[s]}")

print("\nLearned Policy:")
for s, a in policy.items():
    print(f"π({s}) = {a}")
