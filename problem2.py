import numpy as np
import gym
from gym import spaces
from gym.utils import seeding

class MyMDP(gym.Env):
    def __init__(self):
        self.states = [0, 1, 2, 3]   # Define the set of states
        self.actions = [0, 1]        # Define the set of actions
        self.transitions = {         # Define the transition probabilities
            0: {0: [0.2, 0], 1: [0.8, 1]},
            1: {0: [0.5, 2], 1: [0.5, 3]},
            2: {0: [1, 0], 1: [0, 0]},
            3: {0: [1, 3], 1: [0, 3]}
        }
        self.rewards = {             # Define the reward function
            0: {0: 0, 1: 0},
            1: {0: 0, 1: 0},
            2: {0: 1, 1: 0},
            3: {0: 0, 1: 10}
        }
        self.discount_factor = 0.9   # Define the discount factor
        self.observation_space = spaces.Discrete(4)  # Define the observation space
        self.action_space = spaces.Discrete(2)       # Define the action space
        self.seed()
        self.state = None
        self.viewer = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        reward = self.rewards[self.state][action]
        prob, next_state = self.transitions[self.state][action]
        done = False if next_state != self.state else True
        self.state = next_state
        return next_state, reward, done, {}

    def reset(self):
        self.state = self.np_random.choice(self.states)
        return self.state

    def render(self, mode='human'):
        pass

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
