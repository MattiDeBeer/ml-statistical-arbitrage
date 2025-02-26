#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 18:12:18 2025

@author: matti
"""


import sys
import numpy as np
import torch.nn as nn
from torch import tensor
from torch import cat
import matplotlib.pyplot as plt
from gymnasium import spaces
sys.path.append("../")
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


# Define the custom feature extractor (for continuous input space)
class FullyConnectedFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, features_dim: int = 50):
        super(FullyConnectedFeatureExtractor, self).__init__(observation_space, features_dim)

        # Define a simple MLP to process the continuous observation space
        input_dim = int(np.prod(observation_space.shape))  # Flatten the observation space
        self.net = nn.Sequential(
            nn.Linear(input_dim, 50),
            nn.ReLU(),
            nn.Linear(50, features_dim)  # Output feature vector
        )

    def forward(self, observations):
        # Flatten the observation and pass through the network
        return self.net(observations.flatten(start_dim=1))  # Flatten to match the input dimension

#Define the custom feature extractor (for continuous input space)
class DictFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict, features_dim: int = 50):
        super(DictFeatureExtractor, self).__init__(observation_space, features_dim)
        
        cont_dim = 10
        disc_amount = 2
            
        self.cont_net = nn.Sequential(
             nn.Linear(cont_dim, 10),
             nn.ReLU(),
         )
         
        
        self.disc_net = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(disc_amount*2,4),
            nn.ReLU()
            )
         
        self.combiner_net = nn.Sequential(
           nn.Linear(24,features_dim),
           nn.ReLU()
           )
        
        self.lstm = nn.LSTM(1, 20, 1, batch_first=True)
      
        
        
    def forward(self, observations):

        disc_obs = cat([
               observations['is_bought'].float().reshape(-1, 2),
               observations['previous_action'].float().reshape(-1, 2)
            ], dim=1)
        
        cont_obs = observations['open'].unsqueeze(-1)
        
        lstm_out, (hidden,cell) = self.lstm(cont_obs)
        
        cont_obs_out = hidden[-1]
        
        #cont_obs_out = self.cont_net(cont_obs)
        disc_obs_out = self.disc_net(disc_obs)
    
        Y = cat([cont_obs_out, disc_obs_out],dim = 1)

        Z = self.combiner_net(Y)
        
        return Z


class DqnModelCont:
    def __init__(self, enviromentClass, FeatureExtractorClass = FullyConnectedFeatureExtractor, **kwargs):
        self.enviroment_dv = DummyVecEnv([lambda: enviromentClass()])
        self.enviroment = enviromentClass()
        # Define policy kwargs with custom feature extractor
        policy_kwargs = dict(
            features_extractor_class=FullyConnectedFeatureExtractor,
            features_extractor_kwargs=dict(features_dim=128)  # Adjust feature size
        )
        
        self.model = DQN(
            policy="MlpPolicy",  # Use MLPPolicy for continuous observation space
            env=self.enviroment_dv,
            policy_kwargs=policy_kwargs,
            verbose=2,
            learning_rate=1e-3,
            buffer_size=10000,
            learning_starts=500,
            batch_size=32,
            gamma=0.99,
            target_update_interval=500,
            tensorboard_log="./dqn_tensorboard/"
        )



    def train(self,train_steps):
        self.model.learn(total_timesteps=train_steps)
        
    def save(self):
        # Save the model after training
        self.model.save("saved_models/")
        
    def plot_episode(self):
        done = False
        prices = []
        actions = []
        enviroment = self.enviroment
        obs, _ = enviroment.reset()
        total_reward = 0
        done = False

        # Run a single episode
        while not done:
            action, _states = self.model.predict(obs, deterministic=True)  # Use deterministic for evaluation
            prices.append(enviroment.get_current_price(enviroment.token))
            actions.append(action)
            obs, reward, done, truncated, info  = enviroment.step(action)
            total_reward += reward
            

        # 5. Print the total reward for this episode
        print(f"Total reward for this episode: {total_reward}")
        
        # Create the figure and axis
        plt.figure(figsize=(10, 6))
        
        # Plot the price data
        plt.plot(prices, label="Price", color="blue", linewidth=2)
        
        # Initialize variables to alternate between buy and sell
        is_buy = True  # Start with buy on the first '1'
        
        # List to store the indices for buy and sell
        buy_indices = []
        sell_indices = []
        
        # Iterate through the actions and determine buy/sell based on alternating 1s
        for i in range(len(actions)):
            if actions[i] == 1:
                if is_buy:
                    buy_indices.append(i)  # It's a buy
                else:
                    sell_indices.append(i)  # It's a sell
                is_buy = not is_buy  # Alternate between buy and sell
                
        # Plot Buy (green triangles)
        plt.scatter(buy_indices, [prices[i] for i in  buy_indices], marker='^', color='green', label="Buy", s=100, zorder=5)
        
        # Plot Sell (red triangles)
        plt.scatter(sell_indices, [prices[i] for i in sell_indices] , marker='v', color='red', label="Sell", s=100, zorder=5)
        
        # Labels and title
        plt.title("Price Data with Buy/Sell Actions")
        plt.xlabel("Time")
        plt.ylabel("Price")
        plt.legend()
        
        # Display the plot
        plt.grid(True)
        plt.show()


            
class DqnModelDict:
    def __init__(self, enviromentClass, FeatureExtractorClass = DictFeatureExtractor, **kwargs):
        self.enviroment_dv = DummyVecEnv([lambda: enviromentClass()])
        self.enviroment = enviromentClass()
        # Define policy kwargs with custom feature extractor
        policy_kwargs = dict(
            features_extractor_class=DictFeatureExtractor,
            features_extractor_kwargs=dict(features_dim=10)  # Adjust feature size
        )
        
        self.model = DQN(
            policy="MultiInputPolicy",  # Use MLPPolicy for discrete observation space
            env=self.enviroment_dv,
            policy_kwargs=policy_kwargs,
            verbose=2,
            learning_rate=1e-3,
            buffer_size=10000,
            learning_starts=500,
            batch_size=32,
            gamma=0.99,
            target_update_interval=500,
            tensorboard_log="./dqn_tensorboard/"
        )



    def train(self,train_steps):
        self.model.learn(total_timesteps=train_steps)
        
    def save(self):
        # Save the model after training
        self.model.save("saved_models/")
        
    def plot_episode(self):
        done = False
        prices = []
        actions = []
        enviroment = self.enviroment
        obs, _ = enviroment.reset()
        total_reward = 0
        done = False

        # Run a single episode
        while not done:
            action, _states = self.model.predict(obs, deterministic=True)  # Use deterministic for evaluation
            prices.append(enviroment.get_current_price(enviroment.token))
            actions.append(action)
            obs, reward, done, truncated, info  = enviroment.step(action)
            total_reward += reward
            

        # 5. Print the total reward for this episode
        print(f"Total reward for this episode: {total_reward}")
        
        # Create the figure and axis
        plt.figure(figsize=(10, 6))
        
        # Plot the price data
        plt.plot(prices, label="Price", color="blue", linewidth=2)
        
        # Initialize variables to alternate between buy and sell
        is_buy = True  # Start with buy on the first '1'
        
        # List to store the indices for buy and sell
        buy_indices = []
        sell_indices = []
        
        # Iterate through the actions and determine buy/sell based on alternating 1s
        for i in range(len(actions)):
            if actions[i] == 1:
                if is_buy:
                    buy_indices.append(i)  # It's a buy
                else:
                    sell_indices.append(i)  # It's a sell
                is_buy = not is_buy  # Alternate between buy and sell
                
        # Plot Buy (green triangles)
        plt.scatter(buy_indices, [prices[i] for i in  buy_indices], marker='^', color='green', label="Buy", s=100, zorder=5)
        
        # Plot Sell (red triangles)
        plt.scatter(sell_indices, [prices[i] for i in sell_indices] , marker='v', color='red', label="Sell", s=100, zorder=5)
        
        # Labels and title
        plt.title("Price Data with Buy/Sell Actions")
        plt.xlabel("Time")
        plt.ylabel("Price")
        plt.legend()
        
        # Display the plot
        plt.grid(True)
        plt.show()
    
        
    


