#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 16:25:50 2025

@author: matti
"""
import sys
import gymnasium as gym
from gymnasium import spaces
import numpy as np

sys.path.append("../")
from envs.binance_trading_enviroment import binance_trading_env

class DQNTradingEnv(gym.Env):
    """Custom Environment that follows gymnasium interface"""
    
    def __init__(self, window_length):
        super(DQNTradingEnv, self).__init__()
        
        dataset_length = 1000
        
        #initialise helper enviroment
        self.helper_env = binance_trading_env()
        
        #Load dataset into helper enviroment
        self.helper_env.get_sin_wave_dataset(dataset_length,period = 0.1, noise = 0, bin_size=10)
        
        # Define action and observation space
        # For example, an action space with discrete actions (0 or 1)
        self.action_space = spaces.Discrete(2)
        
        n = window_length + 1
        # For example, an observation space with continuous values
        self.observation_space = spaces.Box(
                low=np.full(n, -np.inf), 
                high=np.full(n, np.inf), 
                dtype=np.float32
                                            )
        
        self.window_length = window_length
        
        #set time to allow for sufficient historical data
        self.helper_env.step_time(window_length)
        
        self.helper_env.money = 2
        self.helper_env.transaction_percentage = 0
        
        self.previous_value = 2
        
    
        # Initialize state and other variables
        self.token_amount_held = 0
        self.state = self.generate_observation()
        self.episode_length = dataset_length - window_length - 1
        self.current_step = 0
        

    def generate_observation(self):
        current_data = self.helper_env.get_prices('SIN',self.window_length,return_data=True)['SIN']
        open_prices = current_data['open']
        observation = np.concat((open_prices,np.array([self.token_amount_held])))
        
        return np.array(observation)
        
        
    def reset(self, seed=None):
        """Reset the environment to the initial state"""
        self.helper_env.close_all_positons()
        self.helper_env.time = self.window_length
        self.state = self.generate_observation()  # Starting state
        self.current_step = 0
        self.token_amount_held = 0
        self.helper_env.money = 2
        self.previous_value = 2
        return self.state, {}  # Return initial state and info dict

    def step(self, action):
        """Execute one time step within the environment"""
        # Apply action (0 or 1)
        if action == 0:
            self.current_value = self.helper_env.get_current_portfolio_value()
            reward = np.log(self.current_value) - np.log(self.previous_value)
            self.helper_env.step_time(1)
            action_type = 0
            
        else: #elif action == 1: 
            #sell if  token is held
            if self.token_amount_held > 0:
                self.token_amount_held = 0
                self.helper_env.close_all_positons()
                self.current_value = self.helper_env.get_current_portfolio_value()
                reward = np.log(self.current_value) - np.log(self.previous_value)
                self.helper_env.step_time(1)
                action_type = -1
            else:
                #buy if token is not held
                self.token_amount_held += self.helper_env.buy_token('SIN', 1, return_data= True)
                self.current_value = self.helper_env.get_current_portfolio_value()
                reward = np.log(self.current_value) - np.log(self.previous_value)
                self.helper_env.step_time(1)    
                action_type = -1
                
        
        self.previous_value = self.current_value
        
        # Increment step count
        self.current_step += 1
        self.state = self.generate_observation()
        
        # Check if episode is done
        done = self.current_step >= self.episode_length
        
        # Info dictionary (optional)
        info = {'step':self.step, 'price': self.state[-2], 'action' : action_type}
        
        return self.state, reward, done, False, info

    def render(self):
        """Render the environment (optional)"""
        print(f"Current State: {self.state}")
        
  

