import sys
import os
import gymnasium
from gymnasium import spaces
import numpy as np

sys.path.append(os.path.abspath("../"))
from envs.binance_trading_enviroment import BinanceTradingEnv

class RlTradingEnv(BinanceTradingEnv,gymnasium.Env):
    """Custom Trading Environment following Gymnasium interface"""

    def __init__(self, window_length = 10, dataset_length = 1000):
        super().__init__()
    
        # Load dataset form inherited trading env
        #self.get_sin_wave_dataset(dataset_length, period=0.1, bin_size=10)
        self.dataset_length = dataset_length
        
        #get complex sin wave dataset
        self.get_complex_sin_wave_dataset(dataset_length)
        
        
        # Define action and observation space
        self.action_space = spaces.Discrete(2)  # 0 = Hold, 1 = Buy/Sell
        n = window_length + 3
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(n,), dtype=np.float32)
        
        #define data window length
        self.window_length = window_length
        
        # step forward to allow for a sufficient historical dataset window
        self.step_time(window_length)
        
        self.token = 'SIN'
        
        #set current funds and transaction fees
        self.money = 50
        self.transaction_percentage = 0.01
        
        #set previous portfolio value
        self.previous_value = 50
        
        # Setup action decay
        self.action_decay_constant = np.log(2) / 10
        self.action_decay_premultiplier = 0
        self.steps_since_action = 0
        
        # Initialize state
        self.bought_last_time = 0
        self.token_amount_held = 0
        self.state = self.generate_observation()
        self.episode_length = dataset_length - window_length - 1
        self.current_step = 0
        
    def reward_function(self, current_value, previous_value, action):
        
        reward = np.log(current_value / previous_value)
        
        action_decay = np.array([self.action_decay_premultiplier * np.exp(-1 * self.steps_since_action * self.action_decay_constant)])
        
        
        #ensure reward cannot be too large
        if reward > 100:
            reward = 100
        elif reward < -100:
            reward = 100
        
        return reward

    def generate_observation(self):
        state = {}
        action_decay = np.array([self.action_decay_premultiplier * np.exp(-1 * self.steps_since_action * self.action_decay_constant)])
        current_data = self.get_historical_prices(self.token, self.window_length, return_data=True)[self.token]
        open_prices = current_data['open']
        
        state = np.concatenate((open_prices, np.array([self.token_amount_held]), action_decay, np.array([self.bought_last_time])))
        
        return state
        
    def reset(self, seed=None, options=None):
        """Reset the environment to the initial state"""
        self.close_all_positons()
        self.token_amount_held = 0
        self.time = self.window_length + 1
        self.bought_last_time = 0
        self.steps_since_action = 0
        self.get_complex_sin_wave_dataset(self.dataset_length, random_seed = seed)
        self.state = self.generate_observation()
        self.current_step = 0
        self.token_amount_held = 0
        self.money = 20
        self.previous_value = 20
        return self.state, {}

    def step(self, action):
            
        # Apply action
        if action == 0:
            self.current_value = self.get_current_portfolio_value()
            reward = self.reward_function(self.current_value, self.previous_value, action)
            self.step_time(1)
            self.steps_since_action += 1
            
        else: 
            # Sell if token is held
            if self.token_amount_held > 0:
                self.token_amount_held = 0
                self.close_all_positons()
                self.current_value = self.get_current_portfolio_value()
                reward = self.reward_function(self.current_value, self.previous_value, action)
                self.step_time(1)
                self.steps_since_action = 0
                
            else:
                # Buy if token is not held
                self.token_amount_held += self.buy_token(self.token, 1, return_data=True)
                self.current_value = self.get_current_portfolio_value()
                reward = self.reward_function(self.current_value,self.previous_value, action)
                self.step_time(1)    
                action_type = 1
                self.steps_since_action = 0
                
        self.bought_last_time = action
        self.previous_value = self.current_value
        self.current_step += 1
        self.state = self.generate_observation()
        
        # Check if episode is done
        done = self.current_step >= self.episode_length
        truncated = False  
        
        info = {}
        
        return self.state, reward, done, truncated, info
        

class TestEnv(gymnasium.Env):
    def __init__(self):
        super(TestEnv, self).__init__()

        # Define observation space as a dictionary
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)

        # Define action space as Discrete (2 actions: 0 and 1)
        self.action_space = spaces.Discrete(2)

    def reset(self, seed= None):
        """Reset the environment and return the initial observation."""
        obs = np.zeros(3)
        return obs, {}

    def step(self, action):
        if action == 1:
            reward = 1
        else:
            reward = 0
            
        """Step the environment with a given action."""
        # Random next observation
        next_obs = np.random.rand(3) # Random continuous feature
        
       

        # Random done signal (end of episode)
        done = np.random.choice([True, False])

        # Info (unused but required)
        info = {}

        return next_obs, reward, done, False, info

        
  

