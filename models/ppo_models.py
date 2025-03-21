from stable_baselines3 import PPO
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
import torch
import torch.nn as nn
import torch.optim as optim
from models.statarb_policy import StatarbExperienceGenerator
from torch.utils.data import Dataset, DataLoader
from gymnasium import spaces


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


# Custom dataset for loading experiences
class ExperienceDataset(Dataset):
    def __init__(self, experiences):
        self.experiences = experiences
    
    def __len__(self):
        return len(self.experiences)
    
    def __getitem__(self, idx):
        observation, action = self.experiences[idx]

        for key in observation.keys():
            observation[key] = torch.tensor(observation[key], dtype=torch.float32)

        return observation, torch.tensor(action, dtype=torch.float32)

class PairsPPOModel:

    def __init__(self,config):
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

        self.features_extractor_class = FeatureExtractorClass

        ### DQN Model Parameters ###
        learning_rate = config.get("learning_rate", 1e-3)
        learning_starts = config.get("learning_starts", 500)
        self.batch_size = config.get("batch_size", 32)
        gamma = config.get("gamma", 0.99)
        target_update_interval = config.get("target_update_interval", 500)
        exploration_initial_eps = config.get("exploration_initial_eps", 1.0)
        exploration_final_eps = config.get("exploration_final_eps", 0.05)
        exploration_fraction = config.get("exploration_fraction", 0.5)
        q_net_layers = config.get("q_net_layers", [])
        verbose_level = config.get('verbose_level', 1)
        self.log = config.get('log', False)
        self.algo_env_class = config.get('algo_env', None)
        self.pretrain_env_class = config.get('pretrain_env', None)

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

        if not self.algo_env_class is None:
            self.algo_env = self.algo_env_class(episode_length=self.episode_length,
                                            indicator_obs = indicator_observation_space,
                                            timeseries_obs = timeseries_observation_space,
                                            discrete_obs = discrete_observation_space,
                                            verbose = verbose,
                                            transaction_precentage = transaction_precentage,
                                            token_pair = token_pair,
                                            z_score_context_length = z_score_context_length,
                                            coint_context_length = coint_context_length,
                                            GPU_available = GPU_AVAILABLE,
                                            dataset_file = test_dataset,
                                            use_algo = True

            )

            self.algo_env_dv = DummyVecEnv([lambda: self.algo_env_class(
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
                    dataset_file = dataset_file,
                    use_algo = True
            )])

        if not self.pretrain_env_class is None:

            self.pretrain_env = self.pretrain_env_class(
                        episode_length=self.episode_length,
                        indicator_obs = indicator_observation_space,
                        timeseries_obs = timeseries_observation_space,
                        discrete_obs = discrete_observation_space,
                        token_pair = token_pair,

                )
            
            self.pretrain_env_dv = DummyVecEnv([lambda: self.pretrain_env_class(
                        episode_length=self.episode_length,
                        indicator_obs = indicator_observation_space,
                        timeseries_obs = timeseries_observation_space,
                        discrete_obs = discrete_observation_space,
                        token_pair = token_pair,

                )])

        self.features_extractor_kwargs=dict(features_dim=10,
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

        # Define policy kwargs with custom feature extractor
        policy_kwargs = dict(
            net_arch = q_net_layers,
            activation_fn=ReLU,
            features_extractor_class=FeatureExtractorClass,
            features_extractor_kwargs = self.features_extractor_kwargs
                            )
        
        self.model = PPO(
            policy="MultiInputPolicy",  # Type of policy architecture (can be 'MlpPolicy', 'CnnPolicy', etc.)
            env=self.enviroment_dv,  # The environment that the agent interacts with
            policy_kwargs=policy_kwargs,  # Additional arguments to pass to the policy network (such as network architecture)
            verbose=verbose_level,  # Verbosity level, higher value means more logging
            learning_rate=learning_rate,  # Learning rate for optimizer
            batch_size=self.batch_size,  # Batch size for training
            gamma=gamma,  # Discount factor for future rewards (used in Bellman equation
            device=device,  # Which device to use (e.g., 'cpu' or 'cuda')
            tensorboard_log=tb_log  # Log directory for TensorBoard
        )

    def train(self,episode_num,eval_frequency = 5, eval_steps=5):
        self.model.set_env(self.enviroment_dv)
        callbacks = [EpisodeRewardLoggerCallback()]
        for i in tqdm( range (0,episode_num), desc='Training Model', unit ='Episode', leave=False):
            self.model.learn(total_timesteps=self.episode_length, reset_num_timesteps=False,callback=callbacks)
            if i % eval_frequency == 0:
                self.eval_episode(eval_steps)
                self.save('dataset_PPO_{i}')

    def train_algo(self,episode_num,eval_frequency = 5, eval_steps=5):
        self.model.set_env(self.algo_env_dv)
        callbacks = [EpisodeRewardLoggerCallback()]
        for i in tqdm( range (0,episode_num), desc='Algo Training Model', unit ='Episode', leave=False):
            self.model.learn(total_timesteps=self.episode_length, reset_num_timesteps=False,callback=callbacks)
            if i % eval_frequency == 0:
                self.eval_episode(eval_steps, algo=True)
                self.save('algo_PPO_{i}')


    def mimic_statarb_trading(self, train_steps=200, gradient_steps=50, algo=False, batch_size=500,buffer_size = 1000):

        if algo:
            environment = self.algo_env
        else:
            environment = self.enviroment

        discrete_obs_dict = {key: space.n for key, space in environment.observation_space.spaces.items() if isinstance(space, spaces.Discrete)}

        criterion = nn.functional.cross_entropy

        for i in tqdm(range(0, train_steps), 'Forcing training on statarb experiences', unit='buffer iteration'):

            # Initialize experience generator
            experience_generator = StatarbExperienceGenerator(environment, self.token_pair)
            experiences = experience_generator.generate_experiences(buffer_size)
            buffer = []

            for experience in experiences:
                
                obs, _, action, _ ,_ , _ = experience

                for key in obs.keys():
                    if key in discrete_obs_dict.keys():
                        obs[key] = np.eye(discrete_obs_dict[key])[obs[key]]

                action = np.eye(2)[action]

                buffer.append((obs,action))

            dataset = ExperienceDataset(buffer)
            dataLoader = DataLoader(dataset,batch_size=10)

            loss_accum = 0

            for obs, actions in dataLoader:
             
                features = self.model.policy.features_extractor(obs)
                logits = self.model.policy.mlp_extractor.policy_net(features)
                action_logits = self.model.policy.action_net(logits)

                self.model.policy.optimizer.zero_grad()
               
                loss = criterion(action_logits,actions)

                loss.backward()
                self.model.policy.optimizer.step()

                loss_accum += loss.item()

            print(f"Train loss {loss_accum}")

    def eval_episode(self,num_episodes,algo=False):
        total_rewards = []
        percentage_changes = []

        for _ in range(num_episodes):
            done = False

            if algo:
                enviroment = self.algo_env
            else:
                enviroment = self.enviroment
    
            obs, _ = enviroment.reset()
            start_cash = enviroment.money
            total_reward = 0
            done = False
            actions = []

            # Run a single episode
            while not done:
                action, _state = self.model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info  = enviroment.step(action)
                total_reward += reward
                actions.append(action)

                self.enviroment.close_all_positions()
                end_cash = enviroment.money

                percentage_change = ((end_cash - start_cash) / start_cash) * 100

                total_rewards.append(total_reward)
                percentage_changes.append(percentage_change)

        avg_reward = np.mean(total_rewards)
        avg_percentage_change = np.mean(percentage_changes)

        print(f"Average reward over {num_episodes} evaluation episodes: {avg_reward}")
        print(f"Average percentage change in money over {num_episodes} evaluation episodes: {avg_percentage_change:.5f}%")
            
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

    def save(self, file_name):
        self.model.save(f"{self.model_save_location}{file_name}")