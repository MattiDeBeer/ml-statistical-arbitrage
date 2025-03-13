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
import matplotlib.pyplot as plt
from gymnasium import spaces
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

#Define the custom feature extractor (for dictionary input space)
class DictFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict, **kwargs):

        #Get the feature dimension
        features_dim = kwargs.get("features_dim", 10)

        #Initialize the base class
        super(DictFeatureExtractor, self).__init__(observation_space,features_dim)

        #Get the compile flag
        compile_flag = kwargs.get("compile_flag", False)

        #Get the continious observation space keys
        self.timeseries_keys = kwargs.get("timeseries_obs", {}).keys()

        #get the indicator observations
        self.indicator_keys = kwargs.get("indicator_obs", {}).keys()

        #Get the discrete observation space keys
        self.disc_keys = kwargs.get("discrete_obs", {}).keys()

        #get the indicator network layers
        self.indicator_layers = kwargs.get("indicator_layers", [0])

        #Get the LSTM hidden size
        lstm_hidden_size = kwargs.get("lstm_hidden_size", 0)

        #Get the disc net layers
        self.disc_layers = kwargs.get("disc_layers", [2,2])

        #Get the combiner hidden layers
        self.combiner_layers = kwargs.get("combiner_layers", [10,10])

        if kwargs.get("verbose", False):  
            print("\nCustom Fearute Extractor Parameters \n")
            print(f"Number of Feature dimensions: {features_dim}")
            print(f"Compile flag: {compile_flag}")
            print(f"Timeseries keys: {self.timeseries_keys}")
            print(f"Indicator keys: {self.indicator_keys}")
            print(f"Discrete keys: {self.disc_keys}")
            print(f"Indicator network layers: {self.indicator_layers}")
            print(f"LSTM hidden size: {lstm_hidden_size}")
            print(f"Discrete network layers: {self.disc_layers}")
            print(f"Combiner network hidden layers: {self.combiner_layers}")

        assert (len (self.indicator_keys) != None) and (len(self.disc_keys) != None) and (len(self.timeseries_keys) != None), "You must provide at least on observation key"
        
        if len(self.indicator_keys) != 0:
            self.indicator_net = nn.Sequential(
                nn.Linear(len(self.disc_keys), self.indicator_layers[0]),
                nn.ReLU()
            )
            for i in range(1,len(self.indicator_layers)):
                self.indicator_net.add_module(f"layer_{i}", nn.Linear(self.indicator_layers[i-1],self.indicator_layers[i]))
                self.indicator_net.add_module(f"relu_{i}", nn.ReLU())
        else:
            self.indicator_layers = [0]
            self.indicator_keys = {}
            self.indicator_net = lambda x : x

        if len(self.disc_keys) != 0:
            self.disc_net = nn.Sequential(
                nn.Flatten(),
                nn.Linear(len(self.disc_keys)*2,self.disc_layers[0]),
                nn.ReLU()
                )
            for i in range(1,len(self.disc_layers)):
                self.disc_net.add_module(f"hidden_{i}", nn.Linear(self.disc_layers[i-1], self.disc_layers[i]))
                self.disc_net.add_module(f"relu_{i}", nn.ReLU())
        else:
            self.disc_layers = [0]
            self.disc_keys = {}
            self.disc_net = lambda x : x
        
        if len(self.timeseries_keys) != 0:
            self.lstm_dict = nn.ModuleDict({})
            for key in self.timeseries_keys:
                self.lstm_dict[key] = nn.LSTM(1, lstm_hidden_size, batch_first=True)
        else:
            self.lstm_hidden_size = [0]
            self.lstm_keys = {}
            self.lstm_dict={}

        self.combiner_net = nn.Sequential(
                nn.Linear(lstm_hidden_size * len(self.timeseries_keys) + self.disc_layers[-1] + self.indicator_layers[-1], self.combiner_layers[0]),
                nn.ReLU(),
            )

        for i in range(1,len(self.combiner_layers)):
            self.combiner_net.add_module(f"hidden_{i}", nn.Linear(self.combiner_layers[i-1], self.combiner_layers[i]))
            self.combiner_net.add_module(f"relu_{i}", nn.ReLU())
        
        self.combiner_net.add_module("output", nn.Linear(self.combiner_layers[-1], features_dim))
        self.combiner_net.add_module("relu", nn.ReLU())

        if kwargs.get("verbose", False):
            print("\nCombiner Net Architecture: \n")
            print(self.combiner_net)

            if len(self.disc_keys) != 0:
                print("\nDiscrete Net Architecture: \n")
                print(self.disc_net)
            if len(self.timeseries_keys) !=  0:
                print("\nLSTM Net Architecture: \n")
                print(self.lstm_dict)
            if len(self.indicator_keys) != 0:
                print("\n Indicator Net Architecture: \n")
                print(self.indicator_net)


        ### COMPILE FOR BETTER PERFORMANCE ###
        ### Note that this causes errors when run in the an IDE ###
        if compile_flag:
            self.cont_net = torch.compile(self.cont_net)
            self.disc_net = torch.compile(self.disc_net)
            self.combiner_net = torch.compile(self.combiner_net)
            self.lstm = torch.compile(self.lstm)

    def forward(self, observations):

        disc_obs = [torch.tensor([])]
        for key in self.disc_keys:
            disc_obs.append(observations[key])
        disc_obs = torch.cat(disc_obs, dim = -1)

        indicator_obs = [torch.tensor([])]
        for key in self.indicator_keys:
            indicator_obs.append[observations[key]]
        indicator_obs = torch.cat(indicator_obs, dim = -1)

        hidden_states = [torch.tensor([])]
        for key, lstm in self.lstm_dict.items():
            obs = observations[key].unsqueeze(-1)
            _, (hn, _) = lstm(obs)
            hidden_states.append(hn[-1])  

        lstm_out = torch.cat(hidden_states, dim=-1)  
        disc_obs_out = self.disc_net(disc_obs)
        indicator_obs_out = self.indicator_net(indicator_obs)

        Y = torch.cat([lstm_out, disc_obs_out, indicator_obs_out],dim = -1)
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
        timeseries_observation_space = config.get("timeseries_obs", {})
        discrete_observation_space = config.get("discrete_obs", {})
        indicator_observation_space = config.get("indicator_obs", {})
        token = config.get("token","BTCUSDT")
        verbose = config.get("verbose", False)
        transaction_precentage = config.get('transaction_precentage', 0.01)
        token_pair = config.get('token_pair',None)

        ### Feature Extractor Configurations ###
        fearutes_dim = config.get("features_dim", 10)
        combiner_layers = config.get("combiner_layers", [10,10])
        disc_layers = config.get("disc_layers", [10,10])
        indicator_layers = config.get("indicator_layers", [2,2])
        lstm_hidden_size = config.get("lstm_hidden_size", 20)
        compile_flag = config.get("compile_flag", False)

        ### DQN Model Parameters ###
        learning_rate = config.get("learning_rate", 1e-3)
        buffer_size = config.get("buffer_size", 10000)
        learning_starts = config.get("learning_starts", 500)
        batch_size = config.get("batch_size", 32)
        gamma = config.get("gamma", 0.99)
        target_update_interval = config.get("target_update_interval", 500)
        exploration_initial_eps = config.get("exploration_initial_eps", 1.0)
        exploration_final_eps = config.get("exploration_final_eps", 0.05)
        exploration_fraction = config.get("exploration_fraction", 0.5)

        if verbose:
            print("\nDQN Config Parameters\n")
            print(f"learning_rate: {learning_rate}")
            print(f"buffer_size: {buffer_size}")
            print(f"learning_starts: {learning_starts}")
            print(f"batch_size: {batch_size}")
            print(f"gamma: {gamma}")
            print(f"target_update_interval: {target_update_interval}")
            print(f"exploration_initial_eps: {exploration_initial_eps}")
            print(f"exploration_final_eps: {exploration_final_eps}")
            print(f"exploration_fraction: {exploration_fraction}")

        self.FeatureExtractorClass = DictFeatureExtractor

        if verbose:
            print(" \nTraining enviroment \n ")

        # Create the enviroment
        self.enviroment_dv = DummyVecEnv([lambda: enviromentClass(
                    episode_length=episode_length,
                    token=token,
                    indicator_obs = indicator_observation_space,
                    timeseries_obs = timeseries_observation_space,
                    discrete_obs = discrete_observation_space,
                    verbose = verbose,
                    transaction_precentage = transaction_precentage,
                    token_pair = token_pair

        )])
        
        if verbose:
            print(" \nTesting enviroment \n")

        self.enviroment = enviromentClass(episode_length=episode_length,
                                        token=token,
                                        indicator_obs = indicator_observation_space,
                                        timeseries_obs = timeseries_observation_space,
                                        discrete_obs = discrete_observation_space,
                                        verbose = verbose,
                                        transaction_precentage = transaction_precentage,
                                        token_pair = token_pair

        )



        # Define policy kwargs with custom feature extractor
        policy_kwargs = dict(
            features_extractor_class=self.FeatureExtractorClass,
            features_extractor_kwargs=dict(features_dim=10,
                                            timeseries_obs = timeseries_observation_space,
                                            discrete_obs = discrete_observation_space,
                                            indicator_obs = indicator_observation_space,
                                            fearutes_dim = fearutes_dim,
                                            combiner_layers = combiner_layers,
                                            indicator_layers = indicator_layers,
                                            disc_layers = disc_layers,
                                            lstm_hidden_size = lstm_hidden_size,
                                            compile_flag = compile_flag,
                                            verbose = verbose
                                            )
                            )
        
        self.model = DQN(
            policy="MultiInputPolicy",
            env=self.enviroment_dv,
            policy_kwargs=policy_kwargs,
            verbose=2,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            gamma=gamma,
            target_update_interval=target_update_interval,
            exploration_initial_eps=exploration_initial_eps,  # Start with full exploration
            exploration_final_eps=exploration_final_eps,   # Minimum exploration
            exploration_fraction=exploration_fraction,
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
    
        
    


