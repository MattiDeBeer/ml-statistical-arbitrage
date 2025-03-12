#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 18:12:18 2025

@author: matti
"""


import sys
import numpy as np
import torch
import torch.nn as nn
from torch import tensor
from torch import cat
import matplotlib.pyplot as plt
from gymnasium import spaces
sys.path.append("../")
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

#Define the custom feature extractor (for dictionary input space)
class DictFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict, **kwargs):

        #Get the feature dimension
        features_dim = kwargs.get("features_dim", 10)
        print(f"output dimension: {features_dim}")

        #Initialize the base class
        super(DictFeatureExtractor, self).__init__(observation_space,features_dim)
        
        print("\nInitializing custom feature extractor \n")

        #Get the compile flag
        compile_flag = kwargs.get("compile_flag", False)
        print(f"compile flag: {compile_flag}")

        #Get the continious observation space keys
        self.continious_keys = kwargs.get("continious_obs", {'open': None}).keys()
        print(f"continious keys: {self.continious_keys}")

        #Get the LSTM hidden size
        lstm_hidden_size = kwargs.get("lstm_hidden_size", 20)
        print(f"lstm hidden size: {lstm_hidden_size}")

        #Get the discrete observation space keys
        self.disc_keys = kwargs.get("discrete_obs", {'is_bought': 2, 'previous_action' : 2}).keys()
        print(f"discrete keys: {self.disc_keys}")

        #Get the discrete output dimension
        self.disc_out_dim = kwargs.get("disc_out_dim", 4)
        print(f"discrete output dimension: {self.disc_out_dim}")

        #Get the continious output dimension
        self.combiner_hidden = kwargs.get("combiner_hidden", [10,10])
        print(f"combiner hidden layers: {self.combiner_hidden}")

        #Get the combiner net hidden dimensions
        self.combiner_hidden = kwargs.get("combiner_hidden", [10,10])
        print(f"combiner hidden layers: {self.combiner_hidden}")

        #Get the disc net hidden dimensions
        self.disc_hidden = kwargs.get("disc_hidden", [2])
        print(f"disc hidden layers: {self.disc_hidden}")

        if kwargs.get("verbose", False):

            print("Custom Feature Extractor parameters \n")
            print(f"continious keys: {self.continious_keys}")
            print(f"lstm hidden size: {lstm_hidden_size}")
            print(f"discrete keys: {self.disc_keys}")
            print(f"discrete output dimension: {self.disc_out_dim}")
            print(f"combiner hidden layers: {self.combiner_hidden}")
            print(f"disc hidden layers: {self.disc_hidden}")
            print(f"compile flag: {compile_flag}")


        self.disc_net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(len(self.disc_keys)*2,self.disc_hidden[0]),
            nn.ReLU()
            )
        for i in range(1,len(self.disc_hidden)):
            self.disc_net.add_module(f"hidden_{i}", nn.Linear(self.disc_hidden[i-1], self.disc_hidden[i]))
            self.disc_net.add_module(f"relu_{i}", nn.ReLU())

        self.disc_net.add_module("output", nn.Linear(self.disc_hidden[-1], self.disc_out_dim))
        self.disc_net.add_module("relu", nn.ReLU())
        
        self.lstm_dict = nn.ModuleDict({})
        for key in self.continious_keys:
            self.lstm_dict[key] = nn.LSTM(1, lstm_hidden_size, batch_first=True)

        self.combiner_net = nn.Sequential(
            nn.Linear(lstm_hidden_size * len(self.continious_keys) + self.disc_out_dim, self.combiner_hidden[0]),
            nn.ReLU(),
        )

        for i in range(1,len(self.combiner_hidden)):
            self.combiner_net.add_module(f"hidden_{i}", nn.Linear(self.combiner_hidden[i-1], self.combiner_hidden[i]))
            self.combiner_net.add_module(f"relu_{i}", nn.ReLU())
        
        self.combiner_net.add_module("output", nn.Linear(self.combiner_hidden[-1], features_dim))
        self.combiner_net.add_module("relu", nn.ReLU())

        if kwargs.get("verbose", False):
            print("\nCombiner Net Architecture: \n")
            print(self.combiner_net)
            print("\nDiscrete Net Architecture: \n")
            print(self.disc_net)
            print("\nLSTM Net Architecture: \n")
            print(self.lstm_dict)


        ### COMPILE FOR BETTER PERFORMANCE ###
        ### Note that this causes errors when run in the an IDE ###
        if compile_flag:
            self.cont_net = torch.compile(self.cont_net)
            self.disc_net = torch.compile(self.disc_net)
            self.combiner_net = torch.compile(self.combiner_net)
            self.lstm = torch.compile(self.lstm)

    def forward(self, observations):

        disc_obs = []
        for key in self.disc_keys:
            disc_obs.append(observations[key])

        disc_obs = cat(disc_obs, dim = -1)
        
        hidden_states = []
        
        for key, lstm in self.lstm_dict.items():
            obs = observations[key].unsqueeze(-1)
            _, (hn, _) = lstm(obs)
            hidden_states.append(hn[-1])  

        lstm_out = torch.cat(hidden_states, dim=-1)  
        disc_obs_out = self.disc_net(disc_obs)

        Y = cat([lstm_out, disc_obs_out],dim = -1)

        Z = self.combiner_net(Y)
        
        return Z

            
class DqnModel:
    def __init__(self, config):

        required_keys = ["enviromentClass","token",]
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Key {key} is missing in the config dict")

        ### Enviroment Configurations ###
        enviromentClass = config.get("enviromentClass")
        episode_length = config.get("episode_length", 1000)
        continious_dim = config.get("continious_dim", 10)
        continious_observation_space = config.get("continious_obs", {'open' : (-np.inf, np.inf)})
        discrete_observation_space = config.get("discrete_obs", {'is_bought' : 2, 'previous_action' : 2})
        token = config.get("token","BTCUSDT")
        verbose = config.get("verbose", False)

        ### Feature Extractor Configurations ###
        fearutes_dim = config.get("features_dim", 10)
        combiner_hidden = config.get("combiner_hidden", [10,10])
        disc_hidden = config.get("disc_hidden", [10,10])
        lstm_hidden_size = config.get("lstm_hidden_size", 20)
        disc_out_dim = config.get("disc_out_dim", 4)
        compile_flag = config.get("compile_flag", False)

        self.FeatureExtractorClass = DictFeatureExtractor

        print(" \nTraining enviroment \n ")
        # Create the enviroment
        self.enviroment_dv = DummyVecEnv([lambda: enviromentClass(
                    episode_length=episode_length,
                    continious_dim=continious_dim,
                    token=token,
                    continious_obs = continious_observation_space,
                    discrete_obs = discrete_observation_space,
                    verbose = verbose

        )])
        
        print(" \nTesting enviroment \n")
        self.enviroment = enviromentClass(episode_length=episode_length,
                                          continious_dim=continious_dim,
                                          token=token,
                                          continious_obs = continious_observation_space,
                                          discrete_obs = discrete_observation_space,
                                          verbose = verbose
                                          )



        # Define policy kwargs with custom feature extractor
        policy_kwargs = dict(
            features_extractor_class=self.FeatureExtractorClass,
            features_extractor_kwargs=dict(features_dim=10,
                                            continious_dim = continious_dim,
                                            continious_obs = continious_observation_space,
                                            discrete_obs = discrete_observation_space,
                                            disc_out_dim = disc_out_dim,
                                            fearutes_dim = fearutes_dim,
                                            combiner_hidden = combiner_hidden,
                                            disc_hidden = disc_hidden,
                                            lstm_hidden_size = lstm_hidden_size,
                                            compile_flag = compile_flag,
                                            verbose = verbose
                                            )  # Adjust feature size
                            )
        
        self.model = DQN(
            policy="MultiInputPolicy",
            env=self.enviroment_dv,
            policy_kwargs=policy_kwargs,
            verbose=2,
            learning_rate=1e-3,
            buffer_size=10000,
            learning_starts=500,
            batch_size=32,
            gamma=0.99,
            target_update_interval=500,
            #tensorboard_log="./dqn_tensorboard/"
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
    
        
    


