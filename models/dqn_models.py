#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 18:12:18 2025

@author: matti
"""
import matplotlib.pyplot as plt
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from collections import defaultdict
from torch.nn import ReLU
from torch.cuda import is_available
import sys
from tqdm import tqdm
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
from collections import deque
import os

class EpisodeRewardLoggerCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose=verbose)
        self.current_episode_reward = 0.0
        self.episode_count = 0 
        self.num_steps_in_episode = 0

    def _on_step(self) -> bool:
        # 1) Accumulate the step’s reward
        if "rewards" in self.locals:
            reward = self.locals["rewards"][0]
            self.current_episode_reward += reward
            self.num_steps_in_episode +=1
            

        # 2) Check if the episode has ended
        if "dones" in self.locals:
            done_array = self.locals["dones"]
            if done_array[0]:
                self.episode_count +=1

                # log total reward for episode to TB
                self.logger.record("episode/total_reward", self.current_episode_reward)
                
                #log average reward to TB
                if self.num_steps_in_episode > 0:
                    avg_reward = self.current_episode_reward / self.num_steps_in_episode
                    self.logger.record("episode/average_reward", avg_reward)
                    
                #Dump to TB using episode counter
                self.logger.dump(self.episode_count)
                
                # Reset for the next episode
                self.current_episode_reward = 0.0
                self.num_steps_in_episode = 0

        return True

class DqnModel:
    def __init__(self, config):
        required_keys = ["enviromentClass","token","feature_extractor_class"]
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
        FeatureExtractorClass = config.get("feature_extractor_class", None)
        fearutes_dim = config.get("features_dim", 10)
        combiner_layers = config.get("combiner_layers", [10,10])
        disc_layers = config.get("disc_layers", [10,10])
        indicator_layers = config.get("indicator_layers", [2,2])
        lstm_hidden_size = config.get("lstm_hidden_size", 20)
        compile_flag = config.get("compile_flag", False)
        verbose_level = config.get('verbose_level', 1)

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
        q_net_layers = config.get("q_net_layers", [])

        #set to GPU if available
        device = "cuda" if is_available() else "cpu"

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

        #Build the test enviroment
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
            net_arch = q_net_layers,
            activation_fn=ReLU,
            features_extractor_class=FeatureExtractorClass,
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
        
        #Create the DQN model
        self.model = DQN(
            policy="MultiInputPolicy",
            env=self.enviroment_dv,
            policy_kwargs=policy_kwargs,
            verbose=verbose_level,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            gamma=gamma,
            target_update_interval=target_update_interval,
            exploration_initial_eps=exploration_initial_eps,  # Start with full exploration
            exploration_final_eps=exploration_final_eps,   # Minimum exploration
            exploration_fraction=exploration_fraction,
            device = device
            #tensorboard_log="./dqn_tensorboard/"
        )

        #Print model parameters if verbose is set
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
            print(f"Device: {device}")
            print("\nQ Newtork\n")
            print(self.model.q_net)
            if not input("\nPlease confirm that this is the desired model parameters (y/n): ").strip().lower() == "y":
                print("\nAborting training")
                sys.exit()

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
    
class PairsDqnModel:
    def __init__(self, config):

        self.config = config
        
        required_keys = ["enviromentClass","token_pair","feature_extractor_class","dataset_file"]
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Key {key} is missing in the config dict")

        ### Enviroment Configurations ###
        enviromentClass = config.get("enviromentClass")
        self.episode_length = config.get("episode_length", 1000)
        timeseries_observation_space = config.get("timeseries_obs", {})
        discrete_observation_space = config.get("discrete_obs", {})
        indicator_observation_space = config.get("indicator_obs", {})
        verbose = config.get("verbose", False)
        transaction_precentage = config.get('transaction_precentage', 0.01)
        token_pair = config.get('token_pair',None)
        z_score_context_length = config.get('z_score_context_length',15)
        coint_context_length = config.get('coint_context_length', 15)
        GPU_AVAILABLE = config.get('GPU_available', False)
        dataset_file = config.get('dataset_file' )
        test_dataset = config.get('test_dataset')

        ### Feature Extractor Configurations ###
        FeatureExtractorClass = config.get("feature_extractor_class", None)
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
        q_net_layers = config.get("q_net_layers", [])
        verbose_level = config.get('verbose_level', 1)
        self.log = config.get('log', False)

        #save configs
        run_id = config.get("run_id", "0000")
        tensorboard_locaiton = config.get("tensorboard_log_file", "./dqn_tensorboard")
        self.model_save_location = config.get("model_save_folder", "saved_models")
        self.model_save_location = self.model_save_location + "/" + str(run_id) + "/"
        
        if self.log:
            tb_log = tensorboard_locaiton+"/"+str(run_id)+"/"
        else:
            tb_log = None
            
        #Initialize some variables that are useful for later plotting
        self.token_pair = token_pair
        self.timeseries_keys = list(timeseries_observation_space.keys())
        self.indicator_keys = list(indicator_observation_space.keys())
        self.discrete_keys = list(discrete_observation_space.keys())

        #set to GPU if available
        device = "cuda" if is_available() else "cpu"
        
        if verbose:
            print(" \nTraining enviroment \n ")

        # Create the enviroment
        self.enviroment_dv = DummyVecEnv([lambda: enviromentClass(
                    episode_length=self.episode_length,
                    indicator_obs = indicator_observation_space,
                    timeseries_obs = timeseries_observation_space,
                    discrete_obs = discrete_observation_space,
                    verbose = verbose,
                    transaction_precentage = transaction_precentage,
                    token_pair = token_pair,
                    z_score_context_length = z_score_context_length,
                    coint_context_length = coint_context_length,
                    GPU_available = GPU_AVAILABLE,
                    dataset_file = dataset_file

        )])
        
        if verbose:
            print(" \nTesting enviroment \n")

        #Create the testing enviroment
        self.enviroment = enviromentClass(episode_length=self.episode_length,
                                        indicator_obs = indicator_observation_space,
                                        timeseries_obs = timeseries_observation_space,
                                        discrete_obs = discrete_observation_space,
                                        verbose = verbose,
                                        transaction_precentage = transaction_precentage,
                                        token_pair = token_pair,
                                        z_score_context_length = z_score_context_length,
                                        coint_context_length = coint_context_length,
                                        GPU_available = GPU_AVAILABLE,
                                        dataset_file = test_dataset

        )



        # Define policy kwargs with custom feature extractor
        policy_kwargs = dict(
            net_arch = q_net_layers,
            activation_fn=ReLU,
            features_extractor_class=FeatureExtractorClass,
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
                                            verbose = verbose,
                                            token_pair = token_pair
                                            )
                            )
        
        self.model = DQN(
            policy="MultiInputPolicy",
            env=self.enviroment_dv,
            policy_kwargs=policy_kwargs,
            verbose=verbose_level,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            gamma=gamma,
            target_update_interval=target_update_interval,
            exploration_initial_eps=exploration_initial_eps,  # Start with full exploration
            exploration_final_eps=exploration_final_eps,   # Minimum exploration
            exploration_fraction=exploration_fraction,
            device = device,
            tensorboard_log = tb_log
        )
        
        self.model.learn(0)
        
        if self.log:
            self.model.logger.record('model/q_net_architecture', str(self.model.q_net))
            for key, value in config.items():
                self.model.logger.record(f"model/{key}", str(value))
        
        self.model.logger.dump(step=0)

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
            print(f"Device: {device}")
            print("\n Q Network Architecture: \n")
            print(self.model.q_net)
            if not input("\nPlease confirm that this is the desired model parameters (y/n): ").strip().lower() == "y":
                print("Aborting training")
                sys.exit()


    def train(self,episode_num,eval_frequency = 5, eval_steps=5):
        train_steps = episode_num * self.episode_length
        callbacks = [EpisodeRewardLoggerCallback()]
        for i in tqdm( range (0,episode_num), desc='Training Model', unit ='Episode' ):
            self.model.learn(total_timesteps=self.episode_length, reset_num_timesteps=False,callback=callbacks)
            if i % eval_frequency == 0:
                self.eval_episode(eval_steps)
                self.save(f"episode_{i}")

        
    def save(self, file_name):
        # Save the model
        self.model.save(f"{self.model_save_location}{file_name}")

    def _generate_keyset(self,timeseries_keys,discrete_keys,indicator_keys,token_pair,excluded_keys = []):

        for key in excluded_keys:
            if key in timeseries_keys:
                timeseries_keys.remove(key)
            if key in indicator_keys:
                indicator_keys.remove(key)
            if key in discrete_keys:
                discrete_keys.remove(key)

        lstm_keys = []
        for key in timeseries_keys:
            if key != 'z_score':
                lstm_keys.append(token_pair[0] + '_' + key)
                lstm_keys.append(token_pair[1] + '_' + key)
            else:
                lstm_keys.append('z_score')

        if 'adfuller' in indicator_keys:
            indicator_keys = list(indicator_keys)
            indicator_keys.remove('adfuller')
            indicator_keys.append(token_pair[0] + '_adfuller')
            indicator_keys.append(token_pair[1]+ '_adfuller')

        return lstm_keys, discrete_keys, indicator_keys
    
    def eval_episode(self,num_episodes):
        total_rewards = []
        percentage_changes = []

        for _ in range(num_episodes):
            done = False
            actions = []
            enviroment = self.enviroment
            obs, _ = enviroment.reset()
            start_cash = self.enviroment.money
            total_reward = 0
            done = False

            # Run a single episode
            while not done:
                action, _state = self.model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info  = enviroment.step(action)
                total_reward += reward

                self.enviroment.close_all_positions()
                end_cash = self.enviroment.money

                percentage_change = ((end_cash - start_cash) / start_cash) * 100
                total_rewards.append(total_reward)
                percentage_changes.append(percentage_change)

        avg_reward = np.mean(total_rewards)
        avg_percentage_change = np.mean(percentage_changes)

        print(f"Average reward over {num_episodes} evaluation episodes: {avg_reward:.5f}")
        print(f"Average percentage change in money over {num_episodes} evaluation episodes: {avg_percentage_change:.5f}%")

    def plot_episode(self,excluded_keys = [], action_num = 1):
        if action_num == 1:
            self.plot_episode_1_action(excluded_keys = excluded_keys)
        elif action_num == 2:
            self.plot_episode_2_action(excluded_keys = excluded_keys)

        
    def plot_episode_1_action(self,excluded_keys = []):
        
        timeseries_keys, discrete_keys, indicator_keys = self._generate_keyset(self.timeseries_keys,self.discrete_keys,self.indicator_keys,self.token_pair, excluded_keys=excluded_keys)

        done = False
        actions = []
        enviroment = self.enviroment
        obs, _ = enviroment.reset()
        total_reward = 0
        done = False
        timeseries_observations = defaultdict(list)
        indicator_observations = defaultdict(list)
        price_observations = defaultdict(list)

        # Run a single episode
        while not done:
            action, _state = self.model.predict(obs, deterministic=True)
            price_observations[self.token_pair[0]].append(enviroment.get_current_price(self.token_pair[0]))
            price_observations[self.token_pair[1]].append(enviroment.get_current_price(self.token_pair[1]))
            actions.append(action)
            obs, reward, done, truncated, info  = enviroment.step(action)
            total_reward += reward

            for key in timeseries_keys:
                timeseries_observations[key].append(obs[key][-1])
            
            for key in indicator_keys:
                indicator_observations[key].append(obs[key][0])

        merged_dict = timeseries_observations | indicator_observations | price_observations
        num_plots = len(merged_dict.keys())

        if num_plots == 0:
            raise ValueError("You must provide sufficient keys to make a plot. It's possible you have chosen to exclude too many keys")
        
        is_buy = True
        buy_indices = []
        sell_indices = []
        for i in range(len(actions)):
            if actions[i] == 1:
                if is_buy:
                    buy_indices.append(i)  #Long on arb
                else:
                    sell_indices.append(i)  #Exit Erb
                is_buy = not is_buy  
        
        fig, axes = plt.subplots(num_plots, 1, figsize=(8, num_plots * 3))

        for ax, (key, array) in zip(axes, merged_dict.items()):
            ax.plot(array)  # Line plot
            ax.scatter(buy_indices, [array[i] for i in  buy_indices], marker='^', color='green', label="Buy arb", s=100, zorder=5)
            ax.scatter(sell_indices, [array[i] for i in sell_indices] , marker='v', color='red', label="Exit arb", s=100, zorder=5)
            ax.set_title(key)  # Use dictionary key as title	
            ax.grid(True)
            
        print(f"Total reward for this episode: {total_reward}")

        plt.tight_layout()  # Adjust layout to avoid overlapping
        plt.show()


    def plot_episode_2_action(self,excluded_keys = []):
            
            timeseries_keys, discrete_keys, indicator_keys = self._generate_keyset(self.timeseries_keys,self.discrete_keys,self.indicator_keys,self.token_pair, excluded_keys=excluded_keys)

            done = False
            actions = []
            enviroment = self.enviroment
            obs, _ = enviroment.reset()
            total_reward = 0
            done = False
            timeseries_observations = defaultdict(list)
            indicator_observations = defaultdict(list)
            price_observations = defaultdict(list)

            # Run a single episode
            while not done:
                action, _state = self.model.predict(obs, deterministic=True)
                price_observations[self.token_pair[0]].append(enviroment.get_current_price(self.token_pair[0]))
                price_observations[self.token_pair[1]].append(enviroment.get_current_price(self.token_pair[1]))
                actions.append(action)
                obs, reward, done, truncated, info  = enviroment.step(action)
                total_reward += reward

                for key in timeseries_keys:
                    timeseries_observations[key].append(obs[key][-1])
                
                for key in indicator_keys:
                    indicator_observations[key].append(obs[key][0])

            merged_dict = timeseries_observations | indicator_observations | price_observations
            num_plots = len(merged_dict.keys())

            if num_plots == 0:
                raise ValueError("You must provide sufficient keys to make a plot. It's possible you have chosen to exclude too many keys")
            
            actions = np.array(actions)

            buy_indices = np.where(actions == 1)[0]
            sell_indices = np.where(actions == 2)[0]
            
            fig, axes = plt.subplots(num_plots, 1, figsize=(8, num_plots * 3))

            for ax, (key, array) in zip(axes, merged_dict.items()):
                ax.plot(array)  # Line plot
                ax.scatter(buy_indices, [array[i] for i in  buy_indices], marker='^', color='green', label="Buy arb", s=100, zorder=5)
                ax.scatter(sell_indices, [array[i] for i in sell_indices] , marker='v', color='red', label="Exit arb", s=100, zorder=5)
                ax.set_title(key)  # Use dictionary key as title	
                ax.grid(True)
                
            print(f"Total reward for this episode: {total_reward}")

            plt.tight_layout()  # Adjust layout to avoid overlapping
            plt.show()
