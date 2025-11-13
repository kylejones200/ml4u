"""Chapter 7: Grid Operations Optimization."""

import numpy as np
import matplotlib.pyplot as plt
import yaml
from pathlib import Path
import random

# Optional dependency: gymnasium
HAS_GYM = True
try:
    import gymnasium as gym
    from gymnasium import spaces
except Exception:
    HAS_GYM = False

# Load config
config_path = Path(__file__).parent / "config.yaml"
with open(config_path) as f:
    config = yaml.safe_load(f)

random.seed(config["model"]["random_state"])
np.random.seed(config["model"]["random_state"])


class VoltageControlEnv(gym.Env if HAS_GYM else object):
    """Custom environment simulating grid voltage with reactive power control."""
    
    def __init__(self):
        if HAS_GYM:
            super().__init__()
        self.voltage = config["environment"]["voltage_init"]
        self.load = 0.8
        self.reactive_power = 0.0
        if HAS_GYM:
            self.action_space = spaces.Discrete(3)  # Decrease Q, Hold, Increase Q
            self.observation_space = spaces.Box(
                low=np.array([0.9, config["environment"]["load_min"]]), 
                high=np.array([1.1, config["environment"]["load_max"]]), 
                dtype=np.float32
            )

    def reset(self, seed=None, options=None):
        self.voltage = config["environment"]["voltage_init"]
        self.load = np.random.uniform(
            config["environment"]["load_init_min"],
            config["environment"]["load_init_max"]
        )
        self.reactive_power = 0.0
        return np.array([self.voltage, self.load], dtype=np.float32), {}

    def step(self, action):
        step_size = config["environment"]["reactive_power_step"]
        if action == 0:  # decrease reactive power
            self.reactive_power -= step_size
        elif action == 2:  # increase reactive power
            self.reactive_power += step_size

        # Voltage dynamics
        self.voltage = (1.0 - 0.05 * (self.load - 0.8) + 0.04 * self.reactive_power + 
                       np.random.normal(0, config["environment"]["voltage_noise"]))
        reward = -abs(self.voltage - 1.0)  # Penalize deviation from 1.0 pu

        self.load += np.random.normal(0, config["environment"]["load_noise"])
        self.load = np.clip(self.load, config["environment"]["load_min"], 
                           config["environment"]["load_max"])

        return np.array([self.voltage, self.load], dtype=np.float32), reward, False, False, {}

    def render(self):
        pass


def train_rl_agent():
    """Train Q-learning agent for voltage control."""
    if not HAS_GYM:
        print("gymnasium not available; skipping RL training.")
        return
    
    env = VoltageControlEnv()
    q_table = np.zeros((10, 10, env.action_space.n))  # Discretized voltage/load

    def discretize(obs):
        v_bin = min(int((obs[0] - 0.9) / 0.02), 9)
        l_bin = min(int((obs[1] - 0.5) / 0.07), 9)
        return v_bin, l_bin

    alpha = config["rl"]["alpha"]
    gamma = config["rl"]["gamma"]
    epsilon = config["rl"]["epsilon"]
    episodes = config["rl"]["episodes"]
    steps = config["rl"]["steps_per_episode"]
    print_interval = config["rl"]["print_interval"]

    for ep in range(episodes):
        state, _ = env.reset()
        v_bin, l_bin = discretize(state)
        total_reward = 0
        for _ in range(steps):
            if random.random() > epsilon:
                action = np.argmax(q_table[v_bin, l_bin])
            else:
                action = env.action_space.sample()
            next_state, reward, _, _, _ = env.step(action)
            nv_bin, nl_bin = discretize(next_state)
            q_table[v_bin, l_bin, action] += alpha * (
                reward + gamma * np.max(q_table[nv_bin, nl_bin]) - q_table[v_bin, l_bin, action]
            )
            v_bin, l_bin = nv_bin, nl_bin
            total_reward += reward
        if ep % print_interval == 0:
            print(f"Episode {ep}: Total Reward {total_reward:.2f}")


if __name__ == "__main__":
    train_rl_agent()
