"""
Chapter 7: Grid Operations Optimization
Reinforcement Learning for voltage control using a simplified feeder simulation.
"""

import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces
import random

class VoltageControlEnv(gym.Env):
    """
    Custom environment simulating grid voltage with reactive power control.
    State: [voltage, load]
    Action: adjust reactive power (+/-)
    """
    def __init__(self):
        super().__init__()
        self.voltage = 1.0
        self.load = 0.8
        self.reactive_power = 0.0
        self.action_space = spaces.Discrete(3)  # Decrease Q, Hold, Increase Q
        self.observation_space = spaces.Box(low=np.array([0.9, 0.5]), high=np.array([1.1, 1.2]), dtype=np.float32)

    def reset(self, seed=None, options=None):
        self.voltage = 1.0
        self.load = np.random.uniform(0.6, 1.0)
        self.reactive_power = 0.0
        return np.array([self.voltage, self.load], dtype=np.float32), {}

    def step(self, action):
        if action == 0:  # decrease reactive power
            self.reactive_power -= 0.02
        elif action == 2:  # increase reactive power
            self.reactive_power += 0.02

        # Voltage dynamics
        self.voltage = 1.0 - 0.05 * (self.load - 0.8) + 0.04 * self.reactive_power + np.random.normal(0, 0.002)
        reward = -abs(self.voltage - 1.0)  # Penalize deviation from 1.0 pu

        self.load += np.random.normal(0, 0.01)
        self.load = np.clip(self.load, 0.5, 1.2)

        done = False
        return np.array([self.voltage, self.load], dtype=np.float32), reward, done, False, {}

    def render(self):
        pass

def train_rl_agent(episodes=200):
    env = VoltageControlEnv()
    q_table = np.zeros((10, 10, env.action_space.n))  # Discretized voltage/load

    def discretize(obs):
        v_bin = min(int((obs[0] - 0.9) / 0.02), 9)
        l_bin = min(int((obs[1] - 0.5) / 0.07), 9)
        return v_bin, l_bin

    alpha, gamma, epsilon = 0.1, 0.9, 0.1

    for ep in range(episodes):
        state, _ = env.reset()
        v_bin, l_bin = discretize(state)
        total_reward = 0
        for _ in range(50):
            action = np.argmax(q_table[v_bin, l_bin]) if random.random() > epsilon else env.action_space.sample()
            next_state, reward, _, _, _ = env.step(action)
            nv_bin, nl_bin = discretize(next_state)
            q_table[v_bin, l_bin, action] += alpha * (reward + gamma * np.max(q_table[nv_bin, nl_bin]) - q_table[v_bin, l_bin, action])
            v_bin, l_bin = nv_bin, nl_bin
            total_reward += reward
        if ep % 20 == 0:
            print(f"Episode {ep}: Total Reward {total_reward:.2f}")

if __name__ == "__main__":
    train_rl_agent()
