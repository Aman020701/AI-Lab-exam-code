import random

# Define the parameters of the problem
num_stages = 10
reward_scale = 100
p_correct = [0.9 - i * 0.05 for i in range(num_stages)]
rewards = [i * reward_scale for i in range(1, num_stages + 1)]

# Define the state and action space
states = [(s, r) for s in range(1, num_stages + 1) for r in range(0, (s * reward_scale) + 1)]
actions = ['proceed', 'quit']

# Define the transition probabilities
def transition_probs(state, action):
    stage, reward = state
    if action == 'proceed':
        p = p_correct[stage-1]
        if random.random() < p:
            next_state = (stage + 1, reward + rewards[stage-1])
        else:
            next_state = (1, 0)
    else: # action == 'quit'
        next_state = (1, 0)
    return {next_state: 1}

# Define the rewards
def rewards_fun(state, action, next_state):
    stage, reward = next_state
    return reward

# Define the termination conditions
def is_terminal(state):
    return state[0] > num_stages or state[1] == 0

# Monte Carlo simulation
def simulate(policy, num_simulations):
    returns = {state: 0 for state in states}
    counts = {state: 0 for state in states}

    for i in range(num_simulations):
        state = (1, 0)
        history = []
        while not is_terminal(state):
            action = policy(state)
            history.append((state, action))
            probs = transition_probs(state, action)
            next_state = random.choices(list(probs.keys()), weights=probs.values())[0]
            state = next_state
        G = 0
        for t in reversed(range(len(history))):
            state, action = history[t]
            reward = rewards_fun(state, action, state)
            G = reward + G
            returns[state] += G
            counts[state] += 1

    # Compute the expected value for each state
    values = {}
    for state in states:
        if counts[state] > 0:
            values[state] = returns[state] / counts[state]
        else:
            values[state] = 0
    return values

# Define the policy to be evaluated
def policy(state):
    stage, reward = state
    if reward < rewards[stage-1] / 2:
        return 'proceed'
    else:
        return 'quit'

# Run the simulation
num_simulations = 10000
values = simulate(policy, num_simulations)

# Find the optimal decision at each stage
for stage in range(1, num_stages+1):
    max_reward = -1
    optimal_action = None
    for action in actions:
        state = (stage, max(rewards) // 2)
        probs = transition_probs(state, action)
        reward = sum([values[next_state] * prob for next_state, prob in probs.items()])
        if reward > max_reward:
            max_reward = reward
            optimal_action = action
    print(f"At stage {stage}, the optimal action is to {optimal_action} with an expected reward of {max_reward:.2f}.")


# input according to question input
# num_stages = 10
# p_correct = [0.99, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
# rewards = [100, 500, 1000, 5000, 10000, 50000, 100000, 500000, 1000000, 5000000]

