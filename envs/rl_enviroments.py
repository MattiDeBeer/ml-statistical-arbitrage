import sys
import os
import gymnasium
from gymnasium import spaces
import numpy as np

sys.path.append(os.path.abspath("../"))
from envs.binance_trading_enviroment import BinanceTradingEnv

### DEPRICIATED ###
class RlTradingEnvContinious(BinanceTradingEnv,gymnasium.Env):
    """Custom Trading Environment following Gymnasium interface"""

    def __init__(self, window_length = 10, dataset_length = 100):
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
        self.money = 20
        self.transaction_percentage = 0.01
        
        #set previous portfolio value
        self.previous_value = 20
        
        # Initialize state
        self.bought_last_time = 0
        self.token_amount_held = 0
        self.state = self.generate_observation()
        self.episode_length = dataset_length - window_length - 1
        self.current_step = 0
        
    def reward_function(self, current_value, previous_value, action):
        
        reward = np.log(current_value / previous_value)
        
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
        super().reset(seed=seed) 
        self.close_all_positions()
        self.token_amount_held = 0
        self.time = self.window_length + 1
        self.bought_last_time = 0
        self.steps_since_action = 0
        self.get_complex_sin_wave_dataset(self.dataset_length)
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
                self.close_all_positions()
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

### DEPRICIATED ###
class RlTradingEnvDict(BinanceTradingEnv,gymnasium.Env):
    """Custom Trading Environment following Gymnasium interface"""

    def __init__(self, window_length = 10, dataset_length = 100):
        super().__init__()
    
        # Load dataset form inherited trading env
        #self.get_sin_wave_dataset(dataset_length, period=0.1, bin_size=10)
        self.dataset_length = dataset_length
        
        #get complex sin wave dataset
        self.get_complex_sin_wave_dataset(dataset_length)
        
        # Define action and observation space
        self.action_space = spaces.Discrete(2)  # 0 = Hold, 1 = Buy/Sell
        n = window_length
        
        
        self.observation_space = spaces.Dict({
                'open': spaces.Box(low=-np.inf, high=np.inf, shape=(n,)),
                #'close': spaces.Box(low=-np.inf, high=np.inf, shape=(n,)),
                'previous_action' : spaces.Discrete(2),
                'is_bought' : spaces.Discrete(2),
                })

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
        self.is_bought = 0
        self.state = self.generate_observation(0)
        self.episode_length = dataset_length - window_length - 1
        self.current_step = 0
        
    def reward_function(self, current_value, previous_value):
        
        #calculate reward
        reward = np.log(current_value / previous_value)
        
        #ensure reward cannot be too large
        if reward > 100:
            reward = 100
        elif reward < -100:
            reward = 100
        
        return reward

    def generate_observation(self,action):
        state = {}
        
        current_data = self.get_historical_prices(self.token, self.window_length, return_data=True)[self.token]
        
        state['open'] = current_data['open']
        #state['close'] = current_data['close']
        state['previous_action'] = action
        state['is_bought'] = self.is_bought
        
        return state
        
    def reset(self, seed=None, options=None):
        """Reset the environment to the initial state"""
        super().reset(seed=seed) 
        self.close_all_positions()
        self.is_bought = 0
        self.time = self.window_length + 1
        self.get_complex_sin_wave_dataset(self.dataset_length)
        self.state = self.generate_observation(0)
        self.current_step = 0
        self.money = 20
        self.previous_value = 20
        return self.state, {}

    def step(self, action):
            
        # Apply action
        if action == 0:
            self.current_value = self.get_current_portfolio_value()
            reward = self.reward_function(self.current_value, self.previous_value)
            self.step_time(1)
            
        else: 
            # Sell if token is held
            if self.is_bought == 1:
                self.is_bought = 0
                self.close_all_positions()
                self.current_value = self.get_current_portfolio_value()
                reward = self.reward_function(self.current_value, self.previous_value)
                self.step_time(1)
                
            else:
                # Buy if token is not held
                self.is_bought = 1
                self.buy_token(self.token, 1)
                self.current_value = self.get_current_portfolio_value()
                reward = self.reward_function(self.current_value,self.previous_value)
                self.step_time(1)    
                
        self.current_step += 1
        self.state = self.generate_observation(action)
        self.previous_value = self.current_value
        
        # Check if episode is done
        done = self.current_step >= self.episode_length
        truncated = False  
        
        info = {}
        
        return self.state, reward, done, truncated, info







class RlTradingEnvBTC(BinanceTradingEnv,gymnasium.Env):
    """Custom Trading Environment following Gymnasium interface"""

    def __init__(self, window_length = 10, episode_length = 1000):
        super().__init__()
    
        # Load dataset form inherited trading env
        self.episode_length = episode_length
        
        #load dataset
        self.load_token_dataset('dataset_100000_1m.h5')
        
        # Define action and observation space
        self.action_space = spaces.Discrete(2)  # 0 = Hold, 1 = Buy/Sell
        n = window_length
        
        #Define observation space
        self.observation_space = spaces.Dict({
                'open_returns': spaces.Box(low=-np.inf, high=np.inf, shape=(n,)),
                #'close': spaces.Box(low=-np.inf, high=np.inf, shape=(n,)),
                'previous_action' : spaces.Discrete(2),
                'is_bought' : spaces.Discrete(2),
                })

        #define data window length
        self.window_length = window_length
        
        #set token
        self.token = 'BTCUSDT'
        
        #set transcation Percentage
        self.transaction_percentage = 0.01
        
        #call reset function
        self.state, _ = self.reset()
        
    def reward_function(self, current_value, previous_value):
        
        #calculate reward
        reward = np.log(current_value / previous_value)
        
        #ensure reward cannot be too large
        if reward > 100:
            reward = 100
        elif reward < -100:
            reward = 100
        
        return reward

    def generate_observation(self,action):
        state = {}
        
        current_data = self.get_historical_prices(self.token, self.window_length, return_data=True)[self.token]
        
        state['open_returns'] = current_data['log_return_open']
        #state['close'] = current_data['close']
        state['previous_action'] = action
        state['is_bought'] = self.is_bought
        
        return state
        
    def reset(self, seed=None, options=None):
        """Reset the environment to the initial state"""
        super().reset(seed=seed) 
        #reset positions
        self.close_all_positions()
        self.is_bought = 0
        
        #set internal time
        self.time = self.window_length + 1
        
        #load episode
        self.get_token_episode(self.token,self.episode_length)
        
        #generate initial observation
        self.state = self.generate_observation(0)
        
        #reset money values
        self.money = 20
        self.previous_value = 20
        return self.state, {}

    def step(self, action):
        self.previous_value = self.get_current_portfolio_value()
        
        # Apply action
        if action == 0:
            pass
            
        else: 
            # Sell if token is held
            if self.is_bought == 1:
                self.is_bought = 0
                self.close_all_positions()
                
            else:
                # Buy if token is not held
                self.is_bought = 1
                self.buy_token(self.token, 1)
                
        #step internal time
        done = not self.step_time(1)  
        
        #calculate reward
        self.current_value = self.get_current_portfolio_value()
        reward = self.reward_function(self.current_value,self.previous_value)
        
        #generate next observation
        self.state = self.generate_observation(action)
        
        # Return uselesss variables
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

        
  

