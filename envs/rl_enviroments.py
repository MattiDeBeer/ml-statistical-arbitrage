import sys
import os
import gymnasium
from gymnasium import spaces
import numpy as np

sys.path.append(os.path.abspath("../"))
from envs.binance_trading_enviroment import BinanceTradingEnv

class RlTradingEnvSin(BinanceTradingEnv,gymnasium.Env):

    """ 
    A trading enviroment tha inherits from BinanceTradingEnv 
    This enviroment is setup to use algromithmically generates sin wave price data.
    This enviroment follows the gymnasium interface.
    """

    def __init__(self, window_length = 10, episode_length = 1000):
        super().__init__()
        """
        Initialize the enviroment with the following parameters:
        window_length: int - The number of previous data points to include in the observation space
        episode_length: int - The number of data points in the episode
        """
    
        #set the episode length
        self.episode_length = episode_length
    
        # Define action and observation space
        self.action_space = spaces.Discrete(2)  # 0 = Hold, 1 = Buy/Sell
        n = window_length
        
        #Define observation space dictionaty
        self.observation_space = spaces.Dict({
                'open': spaces.Box(low=-np.inf, high=np.inf, shape=(n,)),
                'previous_action' : spaces.Discrete(2),
                'is_bought' : spaces.Discrete(2),
                })

        #define data window length
        self.window_length = window_length
        
        #set token
        self.token = 'SIN'
        
        #set transcation Percentage
        self.transaction_percentage = 0.01
        
        #call reset function
        self.state, _ = self.reset()
        
    def reward_function(self, current_value, previous_value):
        """
        Calculate the reward for the current step
        params:
        current_value: float - The current portfolio value
        previous_value: float - The previous portfolio value
        returns:
        reward: float - The reward for the current step
        """
        
        #calculate reward using log returns
        reward = np.log(current_value / previous_value)
        
        #ensure reward cannot be too large
        if reward > 100:
            reward = 100
        elif reward < -100:
            reward = 100
        
        return reward

    def generate_observation(self,action):
        """
        Generate the observation for the current step.
        The observation is a dictionary containing the following keys:
        open: np.array - The previous open prices for the last window_length steps
        previous_action: int - The action taken in the previous step
        is_bought: int - 1 if the token is held, 0 if not
        params:
        action: int - The action taken in the previous step
        returns:
        state: dict - The observation for the current step
        """
        state = {}
        
        #get current data
        current_data = self.get_historical_prices(self.token, self.window_length)[self.token]
        
        #set observation values
        state['open'] = current_data['open']
        state['previous_action'] = action
        state['is_bought'] = self.is_bought
        
        return state
        
    def reset(self, seed=None, options=None):
        """
        Reset the environment to the initial state. This also loads a new episode.
        params:
        seed: int - The random seed to use
        options: dict - Additional options to pass to the reset function
        returns:
        state: dict - The observation for the initial state
        """
        super().reset(seed=seed) 

        #reset positions
        self.close_all_positions()
        self.is_bought = 0
        
        #set internal time
        self.time = self.window_length + 1
        
        #load episode
        self.get_complex_sin_wave_episode(self.episode_length, noise = 0,bin_size = 10,return_data = False)
        
        #generate initial observation
        self.state = self.generate_observation(0)
        
        #reset money values
        self.money = 20
        self.previous_value = 20

        return self.state, {}

    def step(self, action):
        """
        Step the environment with a given action.
        params:
        action: int - The action to take
        returns:
        state: dict - The observation for the current step
        reward: float - The reward for the current step
        done: bool - True if the episode is complete
        truncated: bool - True if the episode was truncated
        info: dict - Additional information
        """
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
    
class RlTradingEnvBTC(BinanceTradingEnv,gymnasium.Env):
    """ 
    A trading enviroment tha inherits from BinanceTradingEnv 
    This enviroment is setup to use historical BTC price data.
    This enviroment follows the gymnasium interface.
    """
    def __init__(self, window_length = 10, episode_length = 1000):
        super().__init__()
        """
        Initialize the enviroment with the following parameters:
        window_length: int - The number of previous data points to include in the observation space
        episode_length: int - The number of data points in the episode
        
        """

        # Load dataset form inherited trading env
        self.episode_length = episode_length
        
        #load the BTC price dataset
        self.load_token_dataset('dataset_100000_1m.h5', directory = '../data/')
        
        # Define action and observation space
        self.action_space = spaces.Discrete(2)  # 0 = Hold, 1 = Buy/Sell
        n = window_length
        
        #Define observation space
        self.observation_space = spaces.Dict({
                'open_returns': spaces.Box(low=-np.inf, high=np.inf, shape=(n,)),
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
        """
        Calculate the reward for the current step
        params:
        current_value: float - The current portfolio value
        previous_value: float - The previous portfolio value
        returns:
        reward: float - The reward for the current step
        """
        #calculate reward
        reward = np.log(current_value / previous_value)
        
        #ensure reward cannot be too large
        if reward > 100:
            reward = 100
        elif reward < -100:
            reward = 100
        
        return reward

    def generate_observation(self,action):
        """
        Generate the observation for the current step.
        The observation is a dictionary containing the following
        keys:
        open_returns: np.array - The previous open log returns for the last window_length steps
        previous_action: int - The action taken in the previous step
        is_bought: int - 1 if the token is held, 0 if not
        params:
        action: int - The action taken in the previous step
        returns:
        state: dict - The observation for the current step
        """
        state = {}

        #Fetch the historical prices
        current_data = self.get_historical_prices(self.token, self.window_length)[self.token]
        
        #load the log returns into the ebvironment state
        state['open_returns'] = current_data['log_return_open']
        
        #set the previous action and the bought indicator
        state['previous_action'] = action
        state['is_bought'] = self.is_bought
        
        return state
        
    def reset(self, seed=None, options=None):
        """
        Reset the environment to the initial state. This also loads a new episode.
        params:
        seed: int - The random seed to use
        options: dict - Additional options to pass to the reset function
        returns:
        state: dict - The observation for the initial state
        """
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
        """
        Step the environment with a given action.
        params:
        action: int - The action to take
        returns:
        state: dict - The observation for the current step
        reward: float - The reward for the current step
        done: bool - True if the episode is complete
        truncated: bool - True if the episode was truncated
        info: dict - Additional information
        """

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
    """
    A simple test environment with a continuous observation space and a discrete action space.
    The observation space is a 3-dimensional continuous space and the action space is a discrete space with 2 actions.
    """
    
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

        
  

