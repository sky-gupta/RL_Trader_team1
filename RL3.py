## This file consists of the code for the environment, setting it up and registering it in gym_examples to make a call play it without stable baselines 3


import gym
import numpy as np
import pandas as pd
import numpy as np
import pygame
from gymnasium import spaces
from gymnasium import Env

class StockTradingEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    
    def __init__(self, df):
        super(StockTradingEnv, self).__init__()
        
        self.df = df
        self.current_step = 0
        
        # Define action and observation space
        self.action_space = spaces.Discrete(3)  # 0: Hold, 1: Buy, 2: Sell
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32)
        
        # Initial investment
        self.initial_balance = 10000
        self.balance = self.initial_balance
        self.shares_held = 0
        self.net_worth = self.initial_balance

    def reset(self):
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = 0
        self.net_worth = self.initial_balance
        return self._next_observation()

    def _next_observation(self):
        obs = self.df.iloc[self.current_step][['Open', 'High', 'Low', 'Close', 'Volume']].values
        return obs

    def step(self, action):
        self.current_step += 1
        
        # Execute action
        if action == 1:  # Buy
            self._buy()
        elif action == 2:  # Sell
            self._sell()

        if self.current_step >= len(self.df) - 1:
            done = True
        else:
            done = False
        
        obs = self._next_observation()
        reward = self.net_worth - self.initial_balance
        info = {}

        return obs, reward, done, info

    def _buy(self):
        available_funds = self.balance
        price = self.df.iloc[self.current_step]['Close']
        shares_bought = available_funds // price
        self.shares_held += shares_bought
        self.balance -= shares_bought * price

    def _sell(self):
        price = self.df.iloc[self.current_step]['Close']
        self.balance += self.shares_held * price
        self.shares_held = 0

    def render(self, mode='human'):
        print(f'Step: {self.current_step}')
        print(f'Balance: {self.balance}')
        print(f'Shares held: {self.shares_held}')
        print(f'Net worth: {self.net_worth}')


from gymnasium.envs.registration import register

register(
     id="gym_examples/stock-v0",
     entry_point="gym_examples.envs:stock",
     max_episode_steps=300,
)

from setuptools import setup

setup(
    name="gym_examples",
    version="0.0.1",
    install_requires=["gymnasium==0.26.0", "pygame==2.1.0"],
)
import gym_examples
from gym_examples.envs.stock import stock
env = gymnasium.make('gym_examples/stock-v0')
