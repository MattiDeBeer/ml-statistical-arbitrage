import sys
import os
import gymnasium
from gymnasium import spaces
import numpy as np
from envs.binance_trading_enviroment import BinanceTradingEnv

class MeanRevertingProcess:
    def __init__(self, mu=0, theta=0.1, sigma=0.1, x0=None, dt=1.0):
        """
        Initialize the mean-reverting Ornstein-Uhlenbeck process.
        
        :param mu: Long-term mean
        :param theta: Mean reversion speed
        :param sigma: Volatility
        :param x0: Initial value (if None, set to mu)
        :param dt: Time step size
        """
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.dt = dt
        self.x = x0 if x0 is not None else mu

    def step(self):
        """Simulate one step of the process and return the new value."""
        dW = np.random.normal(0, np.sqrt(self.dt))  # Brownian increment
        self.x += self.theta * (self.mu - self.x) * self.dt + self.sigma * dW
        return self.x
    


class RlTradingEnvSin(BinanceTradingEnv,gymnasium.Env):

    """ 
    A trading enviroment tha inherits from BinanceTradingEnv 
    This enviroment is setup to use algromithmically generates sin wave price data.
    This enviroment follows the gymnasium interface.
    """

    def __init__(self, **kwargs):
        super().__init__()
        """
        Initialize the enviroment with the following parameters:
        window_length: int - The number of previous data points to include in the observation space
        episode_length: int - The number of data points in the episode
        """

        #Fetch arguments
        self.episode_length = kwargs.get('episode_length', 1000)
        self.window_length = kwargs.get('continious_dim', 10)
        self.token = 'SIN'
        self.transaction_percentage = kwargs.get('transaction_percentage', 0.01)
        self.timeseries_obs = kwargs.get('timeseries_obs', {})
        self.discrerete_obs = kwargs.get('discrete_obs', {})
        self.indicator_keys = kwargs.get('indicator_obs', {}).keys()

        assert len(self.indicator_keys) == 0, "The indicator observation space is not allowed in this enviroment, so you must remove it from your config"

        assert len(self.timeseries_obs.keys()) != 0 or len(self.discrerete_obs.keys()) != 0, "You musy provide at least one valid key"

        #check if continious keys are allowed
        allowed_timeseries_keys = ['open','high','low','close']
        for key in self.timeseries_obs.keys():
            if key not in allowed_timeseries_keys:
                raise ValueError(f"The timeseries key '{key}' not allowed in this enviroment. Please use one of the following keys: {allowed_timeseries_keys}")
            
        #check if discrete keys are allowed
        allowed_disc_keys = ['is_bought','previous_action']
        for key in self.discrerete_obs.keys():
            if key not in allowed_disc_keys:
                raise ValueError(f"Key discrete key '{key}' not allowed in this enviroment. Please use one of the following keys: {allowed_disc_keys}")
            
        if kwargs.get('verbose', False):
            #print environment parameters if verbose is set to True
            print("\nEnvironment parameters:")
            print(f"Episode length: {self.episode_length}")
            print(f"Token: {self.token}")
            print(f"Transaction percentage: {self.transaction_percentage}")
            print(f"Timeseries keys: {self.timeseries_obs.keys()}")
            print(f"Discrete keys: {self.discrerete_obs.keys()}")

        #set the window length to the maximin timeseries observation to allow for sufficient data availability
        self.window_length = 1
        for value in self.timeseries_obs.values():
            if value[0] > self.window_length:
                self.window_length = value[0]

        # Define action and observation space
        self.action_space = spaces.Discrete(2)  # 0 = Hold, 1 = Buy/Sell
        
        #Define observation space
        self.observation_space = spaces.Dict({})
        
        #populate the observation space dictionary with continious keys
        for key in self.timeseries_obs.keys():
            self.observation_space[key] = spaces.Box(low=self.timeseries_obs[key][1], high=self.timeseries_obs[key][2], shape=(self.timeseries_obs[key][0],))

        #populate the discrete keys
        for key in self.discrerete_obs.keys():
            self.observation_space[key] = spaces.Discrete(self.discrerete_obs[key])

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
        if current_value == 0:
            current_value = 1e-6
            
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
        
        #get current prices
        current_data = self.get_historical_prices(self.token, self.window_length)[self.token]
        
        #load the log returns into the environment state
        for key in self.timeseries_obs.keys():
            state[key] = current_data[key][-self.timeseries_obs[key][0]:]
        
        # Populate discrete observarions
        if 'previous_action' in self.discrerete_obs.keys():
            state['previous_action'] = action
        if 'is_bought' in self.discrerete_obs.keys():
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
        
        #load episode
        self.get_complex_sin_wave_episode(self.episode_length, noise = 0,bin_size = 10,return_data = False)

        #set internal time
        self.time = self.window_length + 1

        #reset money values
        self.money = 20
        self.previous_value = 20
        
        #generate initial observation
        self.state = self.generate_observation(0)
        
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
        done = self.step_time(1)  
        
        #calculate reward
        self.current_value = self.get_current_portfolio_value()
        reward = self.reward_function(self.current_value,self.previous_value)
        
        #generate next observation
        self.state = self.generate_observation(action)
        
        # Return uselesss variables
        truncated = False  
        info = {}
        
        return self.state, reward, done, truncated, info
    
class RlTradingEnvToken(BinanceTradingEnv,gymnasium.Env):
    """ 
    A trading enviroment tha inherits from BinanceTradingEnv 
    This enviroment is setup to use historical BTC price data.
    This enviroment follows the gymnasium interface.
    """
    def __init__(self, **kwargs):
        super().__init__()
        """
        Initialize the enviroment with the following parameters:
        window_length: int - The number of previous data points to include in the observation space
        episode_length: int - The number of data points in the episode
        
        """

        #Fetch arguments
        self.episode_length = kwargs.get('episode_length', 1000)
        self.token = kwargs.get('token', 'BTCUSDT')
        self.transaction_percentage = kwargs.get('transaction_percentage', 0.001)
        self.dataset_filename = kwargs.get('dataset_file', 'data/dataset_100000_1m.h5')
        self.timeseries_obs = kwargs.get('timeseries_obs', {'open' : (10, -np.inf, np.inf)})
        self.discrerete_obs = kwargs.get('discrete_obs', {'is_bought' : 2, 'previous_action' : 2})
        self.indicator_keys = kwargs.get('indicator_obs', {}).keys()

        #Assert there are no indicator observations
        assert len(self.indicator_keys) == 0, "The indicator observation space is not allowed in this enviroment, so you must remove it from your config"

        #Make sure theres something in the observation space
        assert len(self.timeseries_obs.keys()) != 0 or len(self.discrerete_obs.keys()) != 0, "You musy provide at least one valid key"
        
        #Check to see if continious keys are allowed
        allowed_timeseries_keys = ['open','high','low','close','volume','log_return_open','log_return_high','log_return_low','log_return_close']
        for key in self.timeseries_obs.keys():
            if key not in allowed_timeseries_keys:
                raise ValueError(f"The timeseries key '{key}' is not allowed in this enviroment. Please use one of the following keys: {allowed_timeseries_keys}")
            
        #check to see if discrete keys are allowed
        allowed_disc_keys = ['is_bought','previous_action']
        for key in self.discrerete_obs.keys():
            if key not in allowed_disc_keys:
                raise ValueError(f"Key discrete key '{key}' not allowed in this enviroment. Please use one of the following keys: {allowed_disc_keys}")
        
        if kwargs.get('verbose', False):
            #print environment parameters if verbose is set to True
            print("\nEnvironment parameters:")
            print(f"Episode length: {self.episode_length}")
            print(f"Token: {self.token}")
            print(f"Transaction percentage: {self.transaction_percentage}")
            print(f"Dataset: {self.dataset_filename}")
            cont_params = [{key : length} for key, length in self.timeseries_obs.items()]
            print(f"Continious keys: {cont_params}")
            print(f"Discrete keys: {self.discrerete_obs.keys()}")

        #set the window length to the maximin timeseries observation to allow for sufficient data availability
        self.window_length = 1
        for value in self.timeseries_obs.values():
            if value[0] > self.window_length:
                self.window_length = value[0]

        #load the price dataset
        self.load_token_dataset(self.dataset_filename)
        
        # Define action and observation space
        self.action_space = spaces.Discrete(2)  # 0 = Hold, 1 = Buy/Sell
    
        #Define observation space
        self.observation_space = spaces.Dict({})
        
        #populate the observation space dictionary with continious keys
        for key in self.timeseries_obs.keys():
            self.observation_space[key] = spaces.Box(low=self.timeseries_obs[key][1], high=self.timeseries_obs[key][2], shape=(self.timeseries_obs[key][0],))

        #populate the discrete keys
        for key in self.discrerete_obs.keys():
            self.observation_space[key] = spaces.Discrete(self.discrerete_obs[key])

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
        
        #load the timeseries into the environment state
        for key in self.timeseries_obs.keys():
            state[key] = current_data[key][-(self.timeseries_obs[key][0]):]
        
        # Populate the state with the specified discrete observations
        if 'previous_action' in self.discrerete_obs.keys():
            state['previous_action'] = action
        if 'is_bought' in self.discrerete_obs.keys():
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
        done = self.step_time(1)  
        
        #calculate reward
        self.current_value = self.get_current_portfolio_value()
        reward = self.reward_function(self.current_value,self.previous_value)
        
        #generate next observation
        self.state = self.generate_observation(action)
        
        # Return uselesss variables
        truncated = False  
        info = {}
        
        return self.state, reward, done, truncated, info
    

class RlTradingEnvPairs(BinanceTradingEnv,gymnasium.Env):
    """ 
    A trading enviroment tha inherits from BinanceTradingEnv 
    This enviroment is setup to use historical BTC price data.
    This enviroment follows the gymnasium interface.
    """
    def __init__(self, **kwargs):
        super().__init__()
        """
        Initialize the enviroment with the following parameters:
        window_length: int - The number of previous data points to include in the observation space
        episode_length: int - The number of data points in the episode
        
        """

        #Fetch arguments
        self.episode_length = kwargs.get('episode_length', 1000)
        self.token_pair = kwargs.get('token_pair', None)
        self.transaction_percentage = kwargs.get('transaction_percentage', 0.001)
        self.dataset_filename = kwargs.get('dataset_file', 'data/dataset_100000_1m.h5')
        self.timeseries_obs = kwargs.get('timeseries_obs', {})
        self.discrerete_obs = kwargs.get('discrete_obs', {})
        self.indicator_obs = kwargs.get('indicator_obs', {})
        self.GPU_AVAILABLE = kwargs.get('GPU_available', False)
        
        if self.token_pair is None:
            print("You have selected the pairs trading enviroment, but have not provided a token pair.")
            raise ValueError("You must provide a token pair in the form of a tuple e.g token_pair = (token1, token2)")
        
        assert len(self.timeseries_obs.keys()) != 0 or len(self.discrerete_obs.keys()) != 0 or len(self.indicator_obs.keys()), "You musy provide at least one valid key"
        
        #Check to see if continious keys are allowed
        allowed_cont_keys = ['open','high','low','close','volume','log_return_open','log_return_high','log_return_low','log_return_close','z_score']
        for key in self.timeseries_obs.keys():
            if not key in allowed_cont_keys:
                raise ValueError(f"The timeseries key '{key}' is not allowed in this enviroment. Please use one of the following timeseries keys: {allowed_cont_keys}")
            
        #Check to see if discrete keys are allowed
        allowed_disc_keys = ['is_bought','previous_action']
        for key in self.discrerete_obs.keys():
            if not key in allowed_disc_keys:
                raise ValueError(f"The discrete key '{key}' is not allowed in this environment. Please use one of the following discrete keys: {allowed_disc_keys}")
            
        #Check to see if indicator keys are allowed
        allowed_indicator_keys = ['adfuller','coint_p_value', 'z_score']
        for key in self.indicator_obs.keys():
            if not key in allowed_indicator_keys:
                raise ValueError(f"The indicator key '{key}' is not allowed in this environment. Please use one of the following indicator keys: {allowed_indicator_keys}")
            
        #set window length to the largest timeseries observations
        self.window_length = 0
        for value in self.timeseries_obs.values():
            if value[0] > self.window_length:
                self.window_length = value[0]

        #set teh z score and cointegration context lengths
        self.z_score_context_length = kwargs.get('z_score_context_length', self.window_length)
        self.coint_context_length = kwargs.get('coint_context_length', self.window_length)
        
        #set the window length to the maximum of these
        self.window_length = max([self.z_score_context_length,self.window_length,self.coint_context_length])

        if kwargs.get('verbose', False):
            #print environment parameters if verbose is set to True
            print("\nEnvironment parameters:")
            print(f"Episode length: {self.episode_length}")
            print(f"Continious dim: {self.window_length}")
            print(f"Pair: {self.token_pair}")
            print(f"Dataset: {self.dataset_filename}")
            print(f"Transaction percentage: {self.transaction_percentage}")
            print(f"Continious keys: {self.timeseries_obs.keys()}")
            print(f"Discrete keys: {self.discrerete_obs.keys()}")
            print(f"Cointegration context length {self.coint_context_length}")
            print(f"z_score_context_length: {self.z_score_context_length}")
            print(f"GPU available {self.GPU_AVAILABLE}")

        #load the price dataset
        self.load_token_dataset(self.dataset_filename)
        
        # Define action and observation space
        self.action_space = spaces.Discrete(2)  # 0 = Hold, 1 = Long on arb / exit arb position
    
        #Define observation space
        self.observation_space = spaces.Dict({})
        
        #for each token, create timeseries observations
        for token in self.token_pair:
            #populate the observation space dictionary with continious keys
            for key in self.timeseries_obs.keys():
                if key != 'z_score':
                    self.observation_space[token+'_'+key] = spaces.Box(low=self.timeseries_obs[key][1], high=self.timeseries_obs[key][2], shape=(self.timeseries_obs[key][0],))

            #populate the indicator observations
            for key in self.indicator_obs.keys():
                if key != 'coint_p_value' and key!='z_score':
                    self.observation_space[token+'_'+key] = spaces.Box(self.indicator_obs[key][0],self.indicator_obs[key][1],shape=(1,))

        #if the cointegration p value is selected, add it to the observation space
        if 'coint_p_value' in self.indicator_obs.keys():
            self.observation_space['coint_p_value'] = spaces.Box(low = self.indicator_obs['coint_p_value'][0], high= self.indicator_obs['coint_p_value'][1], shape = (1,))


        if 'z_score' in self.indicator_obs.keys():
            self.observation_space['z_score'] = spaces.Box(low=self.indicator_obs['z_score'][0], high=self.indicator_obs['z_score'][1], shape=(1,))

        #if the z_score it present, add it to the observation space
        if 'z_score' in self.timeseries_obs.keys():
            self.observation_space['z_score'] = spaces.Box(low = self.timeseries_obs['z_score'][1], high= self.timeseries_obs['z_score'][2], shape= (self.timeseries_obs['z_score'][0],))

        #populate the discrete keys
        for key in self.discrerete_obs.keys():
            self.observation_space[key] = spaces.Discrete(self.discrerete_obs[key])

        assert not ('z_score' in self.indicator_obs.keys() and 'z_score' in self.timeseries_obs.keys()), "z_score cannot be both a timeseries observation and an indicator observation"

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

        #Fetch the data
        current_data = self.get_historical_prices(self.token_pair, self.window_length)

        if ('coint_p_value' in self.indicator_obs.keys() or 'adfuller' in self.indicator_obs.keys()) and self.GPU_AVAILABLE:
            #Use GPU if available
            coint_results = self.calc_coint_values_GPU(self.token_pair[0],self.token_pair[1],self.coint_context_length,key='open')
        elif ('coint_p_value' in self.indicator_obs.keys() or 'adfuller' in self.indicator_obs.keys()) and not self.GPU_AVAILABLE:
            #Use CPU
            coint_results = self.calc_coint_values(self.token_pair[0],self.token_pair[1],self.coint_context_length,key='open')            

        #populate the timeseries information for all tokens
        for token in self.token_pair:
            for key in self.timeseries_obs.keys():
                if key != 'z_score':
                    state[token+'_'+key] = current_data[token][key][-self.timeseries_obs[key][0]:]

        #populate the cointegreation p calue once
        if 'coint_p_value' in self.indicator_obs.keys():
            state['coint_p_value'] = np.array([coint_results[0]])

        #populate the historical z scores once
        if 'z_score' in self.timeseries_obs.keys():
            state['z_score'] = self.get_z_scores(self.token_pair[0],self.token_pair[1],self.z_score_context_length)['open'][-self.timeseries_obs['z_score'][0]:]

        #populate the z score
        if 'z_score' in self.indicator_obs.keys():
            state['z_score'] = self.get_z_scores(self.token_pair[0],self.token_pair[1],self.z_score_context_length)['open'][-1:]

        #populate the adfuller metrics once
        if 'adfuller' in self.indicator_obs.keys():
            state[self.token_pair[0]+"_adfuller"] = np.array([coint_results[1]])
            state[self.token_pair[1]+"_adfuller"] = np.array([coint_results[2]])

        #populate the discrete keys
        if 'previous_action' in list(self.observation_space.keys()):
            state['previous_action'] = action
        if 'is_bought' in list(self.observation_space.keys()):
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
        self.get_token_episode(self.token_pair,self.episode_length)
        
        #generate initial observation
        self.state = self.generate_observation(0)
        
        #reset money values
        self.money = 100
        self.previous_value = 100

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
            # Exit arb position of you are currently in one
            if self.is_bought == 1:
                self.is_bought = 0
                self.close_all_positions()
                
            else:
                # Enter arb positon if you are not in one
                self.is_bought = 1
                self.buy_token(self.token_pair[0], 1)
                self.short_token(self.token_pair[1],1)
                
        #step internal time
        done = self.step_time(1)  
        
        #calculate reward
        self.current_value = self.get_current_portfolio_value()
        reward = self.reward_function(self.current_value,self.previous_value)
        
        #generate next observation
        self.state = self.generate_observation(action)
        
        # Return uselesss variables
        truncated = False  
        info = {}
        
        return self.state, reward, done, truncated, info
    

class RlTradingEnvPairsExtendedActions(BinanceTradingEnv,gymnasium.Env):
    """ 
    A trading enviroment tha inherits from BinanceTradingEnv 
    This enviroment is setup to use historical BTC price data.
    This enviroment follows the gymnasium interface.
    """
    def __init__(self, **kwargs):
        super().__init__()
        """
        Initialize the enviroment with the following parameters:
        window_length: int - The number of previous data points to include in the observation space
        episode_length: int - The number of data points in the episode
        
        """

        #Fetch arguments
        self.episode_length = kwargs.get('episode_length', 1000)
        self.token_pair = kwargs.get('token_pair', None)
        self.transaction_percentage = kwargs.get('transaction_percentage', 0.001)
        self.dataset_filename = kwargs.get('dataset_file', 'data/dataset_100000_1m.h5')
        self.timeseries_obs = kwargs.get('timeseries_obs', {})
        self.discrerete_obs = kwargs.get('discrete_obs', {})
        self.indicator_obs = kwargs.get('indicator_obs', {})
        self.GPU_AVAILABLE = kwargs.get('GPU_available', False)
        self.use_algo = kwargs.get('use_algo',False)
        
        if self.token_pair is None:
            print("You have selected the pairs trading enviroment, but have not provided a token pair.")
            raise ValueError("You must provide a token pair in the form of a tuple e.g token_pair = (token1, token2)")
        
        assert len(self.timeseries_obs.keys()) != 0 or len(self.discrerete_obs.keys()) != 0 or len(self.indicator_obs.keys()), "You musy provide at least one valid key"
        
        #Check to see if continious keys are allowed
        allowed_cont_keys = ['open','high','low','close','volume','log_return_open','log_return_high','log_return_low','log_return_close','z_score']
        for key in self.timeseries_obs.keys():
            if not key in allowed_cont_keys:
                raise ValueError(f"The timeseries key '{key}' is not allowed in this enviroment. Please use one of the following timeseries keys: {allowed_cont_keys}")
            
        #Check to see if discrete keys are allowed
        allowed_disc_keys = ['is_bought','previous_action']
        for key in self.discrerete_obs.keys():
            if not key in allowed_disc_keys:
                raise ValueError(f"The discrete key '{key}' is not allowed in this environment. Please use one of the following discrete keys: {allowed_disc_keys}")
            
        #Check to see if indicator keys are allowed
        allowed_indicator_keys = ['adfuller','coint_p_value','amount_bought','z_score']
        for key in self.indicator_obs.keys():
            if not key in allowed_indicator_keys:
                raise ValueError(f"The indicator key '{key}' is not allowed in this environment. Please use one of the following indicator keys: {allowed_indicator_keys}")
            
        #set window length to the largest timeseries observations
        self.window_length = 0
        for value in self.timeseries_obs.values():
            if value[0] > self.window_length:
                self.window_length = value[0]

        #set teh z score and cointegration context lengths
        self.z_score_context_length = kwargs.get('z_score_context_length', self.window_length)
        self.coint_context_length = kwargs.get('coint_context_length', self.window_length)
        
        #set the window length to the maximum of these
        self.window_length = max([self.z_score_context_length,self.window_length,self.coint_context_length])

        if kwargs.get('verbose', False):
            #print environment parameters if verbose is set to True
            print("\nEnvironment parameters:")
            print(f"Episode length: {self.episode_length}")
            print(f"Continious dim: {self.window_length}")
            print(f"Pair: {self.token_pair}")
            print(f"Dataset: {self.dataset_filename}")
            print(f"Transaction percentage: {self.transaction_percentage}")
            print(f"Continious keys: {self.timeseries_obs.keys()}")
            print(f"Discrete keys: {self.discrerete_obs.keys()}")
            print(f"Cointegration context length {self.coint_context_length}")
            print(f"z_score_context_length: {self.z_score_context_length}")
            print(f"GPU available {self.GPU_AVAILABLE}")
            print(f"using_algo: {self.use_algo}")

        if not self.use_algo:
            #load the price dataset
            self.load_token_dataset(self.dataset_filename)
        
        # Define action and observation space
        self.action_space = spaces.Discrete(3)  # 0 = Hold, 1 = Long on arb, 2 = sell
    
        #Define observation space
        self.observation_space = spaces.Dict({})
        
        #for each token, create timeseries observations
        for token in self.token_pair:
            #populate the observation space dictionary with continious keys
            for key in self.timeseries_obs.keys():
                if key != 'z_score':
                    self.observation_space[token+'_'+key] = spaces.Box(low=self.timeseries_obs[key][1], high=self.timeseries_obs[key][2], shape=(self.timeseries_obs[key][0],))

            #populate the indicator observations
            for key in self.indicator_obs.keys():
                if key != 'coint_p_value' and key != 'amount_bought' and key!='z_score':
                    self.observation_space[token+'_'+key] = spaces.Box(self.indicator_obs[key][0],self.indicator_obs[key][1],shape=(1,))

        #if the cointegration p value is selected, add it to the observation space
        if 'coint_p_value' in self.indicator_obs.keys():
            self.observation_space['coint_p_value'] = spaces.Box(low = self.indicator_obs['coint_p_value'][0], high= self.indicator_obs['coint_p_value'][1], shape = (1,))

        if 'amount_bought' in self.indicator_obs.keys():
            self.observation_space['amount_bought'] = spaces.Box(low=self.indicator_obs['amount_bought'][0], high=self.indicator_obs['amount_bought'][1], shape=(1,))

        if 'z_score' in self.indicator_obs.keys():
            self.observation_space['z_score'] = spaces.Box(low=self.indicator_obs['z_score'][0], high=self.indicator_obs['z_score'][1], shape=(1,))

        #if the z_score it present, add it to the observation space
        if 'z_score' in self.timeseries_obs.keys():
            self.observation_space['z_score'] = spaces.Box(low = self.timeseries_obs['z_score'][1], high= self.timeseries_obs['z_score'][2], shape= (self.timeseries_obs['z_score'][0],))

        #populate the discrete keys
        for key in self.discrerete_obs.keys():
            self.observation_space[key] = spaces.Discrete(self.discrerete_obs[key])

        assert not ('z_score' in self.indicator_obs.keys() and 'z_score' in self.timeseries_obs.keys()), "z_score cannot be both a timeseries observation and an indicator observation"

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
        current_value = max(current_value,1e-7)
        previous_value = max(previous_value,1e-6)

        #calculate reward
        reward = np.log(current_value / previous_value)
        
        #ensure reward cannot be too large
        if reward > 100:
            reward = 100
        elif reward < -100:
            reward = 100
        
        if self.money <= 0:
            reward -= 10
        
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

        #Fetch the data
        current_data = self.get_historical_prices(self.token_pair, self.window_length)

        if ('coint_p_value' in self.indicator_obs.keys() or 'adfuller' in self.indicator_obs.keys()) and self.GPU_AVAILABLE:
            #Use GPU if available
            coint_results = self.calc_coint_values_GPU(self.token_pair[0],self.token_pair[1],self.coint_context_length,key='open')
        elif ('coint_p_value' in self.indicator_obs.keys() or 'adfuller' in self.indicator_obs.keys()) and not self.GPU_AVAILABLE:
            #Use CPU
            coint_results = self.calc_coint_values(self.token_pair[0],self.token_pair[1],self.coint_context_length,key='open')            

        #populate the timeseries information for all tokens
        for token in self.token_pair:
            for key in self.timeseries_obs.keys():
                if key != 'z_score':
                    state[token+'_'+key] = current_data[token][key][-self.timeseries_obs[key][0]:]

        #populate the cointegreation p calue once
        if 'coint_p_value' in self.indicator_obs.keys():
            state['coint_p_value'] = np.array([coint_results[0]])

        #populate the historical z scores once
        if 'z_score' in self.timeseries_obs.keys():
            state['z_score'] = self.get_z_scores(self.token_pair[0],self.token_pair[1],self.z_score_context_length)['open'][-self.timeseries_obs['z_score'][0]:]

        #populate the z score
        if 'z_score' in self.indicator_obs.keys():
            state['z_score'] = self.get_z_scores(self.token_pair[0],self.token_pair[1],self.z_score_context_length)['open'][-1:]

        #populate the adfuller metrics once
        if 'adfuller' in self.indicator_obs.keys():
            state[self.token_pair[0]+"_adfuller"] = np.array([coint_results[1]])
            state[self.token_pair[1]+"_adfuller"] = np.array([coint_results[2]])

        #populate the discrete keys
        if 'previous_action' in list(self.observation_space.keys()):
            state['previous_action'] = action
        if 'is_bought' in list(self.observation_space.keys()):
            state['is_bought'] = self.is_bought

        #populate amount_bought indicator
        if 'amount_bought' in list(self.observation_space.keys()):
            state['amount_bought'] = np.array([self.amount_bought])
        
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
        self.amount_bought = 0
        
        #set internal time
        self.time = self.window_length + 1
        
        #load episode
        if self.use_algo:
            self.get_algo_token_episode(self.token_pair,self.episode_length)
        else:
            self.get_token_episode(self.token_pair,self.episode_length)
        
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

        elif action == 1:
            self.amount_bought += 1
            self.is_bought = 1
            self.buy_token(self.token_pair[0], 1)
            self.short_token(self.token_pair[1],1)
            
        elif action == 2:
            self.amount_bought = 0
            self.is_bought = 0
            self.close_all_positions()
                
        #step internal time
        done = self.step_time(1)  
        
        #calculate reward
        self.current_value = self.get_current_portfolio_value()
        reward = self.reward_function(self.current_value,self.previous_value)
        
        #generate next observation
        self.state = self.generate_observation(action)
        
        # Return uselesss variables
        truncated = False  
        info = {}
        
        return self.state, reward, done, truncated, info
    
class RlPretrainEnvExtendedActions(gymnasium.Env):
    """ 
    A trading enviroment tha inherits from BinanceTradingEnv 
    This enviroment is setup to use historical BTC price data.
    This enviroment follows the gymnasium interface.
    """
    def __init__(self, **kwargs):
        super().__init__()
        """
        Initialize the enviroment with the following parameters:
        window_length: int - The number of previous data points to include in the observation space
        episode_length: int - The number of data points in the episode
        
        """

        #Fetch arguments
        self.episode_length = kwargs.get('episode_length', 1000)
        self.timeseries_obs = kwargs.get('timeseries_obs', {})
        self.discrerete_obs = kwargs.get('discrete_obs', {})
        self.indicator_obs = kwargs.get('indicator_obs', {})
        self.token_pair = kwargs.get('token_pair', None)

        if self.token_pair is None:
            print("You have selected the pretrain enviroment, but have not provided a token pair.")
            raise ValueError("You must provide a token pair in the form of a tuple e.g token_pair = (token1, token2)")
        
        assert len(self.timeseries_obs.keys()) != 0 or len(self.discrerete_obs.keys()) != 0 or len(self.indicator_obs.keys()), "You musy provide at least one valid key"
        
        #Check to see if continious keys are allowed
        allowed_cont_keys = ['open','high','low','close','volume','log_return_open','log_return_high','log_return_low','log_return_close','z_score']
        for key in self.timeseries_obs.keys():
            if not key in allowed_cont_keys:
                raise ValueError(f"The timeseries key '{key}' is not allowed in this enviroment. Please use one of the following timeseries keys: {allowed_cont_keys}")
            
        #Check to see if discrete keys are allowed
        allowed_disc_keys = ['is_bought','previous_action']
        for key in self.discrerete_obs.keys():
            if not key in allowed_disc_keys:
                raise ValueError(f"The discrete key '{key}' is not allowed in this environment. Please use one of the following discrete keys: {allowed_disc_keys}")
            
        #Check to see if indicator keys are allowed
        allowed_indicator_keys = ['adfuller','coint_p_value','amount_bought','z_score']
        for key in self.indicator_obs.keys():
            if not key in allowed_indicator_keys:
                raise ValueError(f"The indicator key '{key}' is not allowed in this environment. Please use one of the following indicator keys: {allowed_indicator_keys}")
            

        # Define action and observation space
        self.action_space = spaces.Discrete(3)  # 0 = Hold, 1 = Long on arb, 2 = sell
    
        #Define observation space
        self.observation_space = spaces.Dict({})
        
        #for each token, create timeseries observations
        for token in self.token_pair:
            #populate the observation space dictionary with continious keys
            for key in self.timeseries_obs.keys():
                if key != 'z_score':
                    self.observation_space[token+'_'+key] = spaces.Box(low=self.timeseries_obs[key][1], high=self.timeseries_obs[key][2], shape=(self.timeseries_obs[key][0],))

            #populate the indicator observations
            for key in self.indicator_obs.keys():
                if key != 'coint_p_value' and key != 'amount_bought' and key!='z_score':
                    self.observation_space[token+'_'+key] = spaces.Box(self.indicator_obs[key][0],self.indicator_obs[key][1],shape=(1,))

        #if the cointegration p value is selected, add it to the observation space
        if 'coint_p_value' in self.indicator_obs.keys():
            self.observation_space['coint_p_value'] = spaces.Box(low = self.indicator_obs['coint_p_value'][0], high= self.indicator_obs['coint_p_value'][1], shape = (1,))

        if 'amount_bought' in self.indicator_obs.keys():
            self.observation_space['amount_bought'] = spaces.Box(low=self.indicator_obs['amount_bought'][0], high=self.indicator_obs['amount_bought'][1], shape=(1,))

        if 'z_score' in self.indicator_obs.keys():
            self.observation_space['z_score'] = spaces.Box(low=self.indicator_obs['z_score'][0], high=self.indicator_obs['z_score'][1], shape=(1,))

        #if the z_score it present, add it to the observation space
        if 'z_score' in self.timeseries_obs.keys():
            self.observation_space['z_score'] = spaces.Box(low = self.timeseries_obs['z_score'][1], high= self.timeseries_obs['z_score'][2], shape= (self.timeseries_obs['z_score'][0],))

        #populate the discrete keys
        for key in self.discrerete_obs.keys():
            self.observation_space[key] = spaces.Discrete(self.discrerete_obs[key])

        assert not ('z_score' in self.indicator_obs.keys() and 'z_score' in self.timeseries_obs.keys()), "z_score cannot be both a timeseries observation and an indicator observation"

        #call reset function
        self.state, _ = self.reset()
    
    def reward_function(self,action):
        """
        Calculate the reward for the current step
        params:
        current_value: float - The current portfolio value
        previous_value: float - The previous portfolio value
        returns:
        reward: float - The reward for the current step
        """
        # Create dictionary for state

        z_score = self.state.get('z_score',None)
        adfuller1 = self.state.get(self.token_pair[0]+"_adfuller",None)
        adfuller2 = self.state.get(self.token_pair[1]+"_adfuller",None)
        coint_p_value = self.state.get('coint_p_value',None)

        if z_score[-1] <=-1.5 and self.amount_bought <= 2 and self.previous_action != 1 and action == 1 and adfuller1 == 0 and adfuller2 == 0 and coint_p_value <= 0.1:
            reward = 1e-2
        elif z_score[-1] >= 0 and self.amount_bought > 0 and self.previous_action != 2 and action == 2:
            reward = 1e-2
        elif self.amount_bought > 2 and action == 1:
            reward = -1e-2
        elif self.amount_bought == 0 and action == 2:
            reward = -1e-2
        else:
            reward = 0

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
        
        #populate the timeseries information for all tokens
        for token in self.token_pair:
            for key in self.timeseries_obs.keys():
                if key != 'z_score':
                    state[token+'_'+key] = np.random.randn(self.timeseries_obs[key])

        #populate the cointegreation p calue once
        if 'coint_p_value' in self.indicator_obs.keys():
            state['coint_p_value'] = np.array([np.random.randint(2)])

        #populate the historical z scores once
        if 'z_score' in self.timeseries_obs.keys():
            state['z_score'] = np.random.randn(self.timeseries_obs['z_score'][0])

        #populate the z score
        if 'z_score' in self.indicator_obs.keys():
            state['z_score'] = np.array([np.sin(self.step_num/100)])

        #populate the adfuller metrics once
        if 'adfuller' in self.indicator_obs.keys():
            state[self.token_pair[0]+"_adfuller"] = np.array([np.random.randint(2)])
            state[self.token_pair[1]+"_adfuller"] = np.array([np.random.randint(2)])

        #populate the discrete keys
        if 'previous_action' in list(self.observation_space.keys()):
            state['previous_action'] = action
        if 'is_bought' in list(self.observation_space.keys()):
            state['is_bought'] = self.is_bought

        #populate amount_bought indicator
        if 'amount_bought' in list(self.observation_space.keys()):
            state['amount_bought'] = np.array([self.amount_bought])

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

        self.is_bought = 0
        self.amount_bought = 0
        self.previous_action = 0
        
        self.step_num = 0
        
        #generate initial observation
        self.state = self.generate_observation(0)

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

        reward = self.reward_function(action)
        
        if action == 1:
            self.amount_bought += 1
            self.is_bought = 1
        elif action == 2:
            self.amount_bought = 0
            self.is_bought = 0

        self.step_num+=1
    
        #step internal time
        done = self.step_num >= self.episode_length
        
        
        #generate next observation
        self.state = self.generate_observation(action)
        # Return uselesss variables
        truncated = False  
        info = {}

        self.previous_action = action

        return self.state, reward, done, truncated, info
    

class RlPretrainEnvSingleAction(gymnasium.Env):
    """ 
    A trading enviroment tha inherits from BinanceTradingEnv 
    This enviroment is setup to use historical BTC price data.
    This enviroment follows the gymnasium interface.
    """
    def __init__(self, **kwargs):
        super().__init__()
        """
        Initialize the enviroment with the following parameters:
        window_length: int - The number of previous data points to include in the observation space
        episode_length: int - The number of data points in the episode
        
        """

        #Fetch arguments
        self.episode_length = kwargs.get('episode_length', 1000)
        self.timeseries_obs = kwargs.get('timeseries_obs', {})
        self.discrerete_obs = kwargs.get('discrete_obs', {})
        self.indicator_obs = kwargs.get('indicator_obs', {})
        self.token_pair = kwargs.get('token_pair', None)

        if self.token_pair is None:
            print("You have selected the pretrain enviroment, but have not provided a token pair.")
            raise ValueError("You must provide a token pair in the form of a tuple e.g token_pair = (token1, token2)")
        
        assert len(self.timeseries_obs.keys()) != 0 or len(self.discrerete_obs.keys()) != 0 or len(self.indicator_obs.keys()), "You musy provide at least one valid key"
        
        #Check to see if continious keys are allowed
        allowed_cont_keys = ['open','high','low','close','volume','log_return_open','log_return_high','log_return_low','log_return_close','z_score']
        for key in self.timeseries_obs.keys():
            if not key in allowed_cont_keys:
                raise ValueError(f"The timeseries key '{key}' is not allowed in this enviroment. Please use one of the following timeseries keys: {allowed_cont_keys}")
            
        #Check to see if discrete keys are allowed
        allowed_disc_keys = ['is_bought','previous_action']
        for key in self.discrerete_obs.keys():
            if not key in allowed_disc_keys:
                raise ValueError(f"The discrete key '{key}' is not allowed in this environment. Please use one of the following discrete keys: {allowed_disc_keys}")
            
        #Check to see if indicator keys are allowed
        allowed_indicator_keys = ['adfuller','coint_p_value','z_score']
        for key in self.indicator_obs.keys():
            if not key in allowed_indicator_keys:
                raise ValueError(f"The indicator key '{key}' is not allowed in this environment. Please use one of the following indicator keys: {allowed_indicator_keys}")
            

        # Define action and observation space
        self.action_space = spaces.Discrete(2)  # 0 = Hold, 1 = Long on arb / sell
    
        #Define observation space
        self.observation_space = spaces.Dict({})
        
        #for each token, create timeseries observations
        for token in self.token_pair:
            #populate the observation space dictionary with continious keys
            for key in self.timeseries_obs.keys():
                if key != 'z_score':
                    self.observation_space[token+'_'+key] = spaces.Box(low=self.timeseries_obs[key][1], high=self.timeseries_obs[key][2], shape=(self.timeseries_obs[key][0],))

            #populate the indicator observations
            for key in self.indicator_obs.keys():
                if key != 'coint_p_value' and key!='z_score':
                    self.observation_space[token+'_'+key] = spaces.Box(self.indicator_obs[key][0],self.indicator_obs[key][1],shape=(1,))

        #if the cointegration p value is selected, add it to the observation space
        if 'coint_p_value' in self.indicator_obs.keys():
            self.observation_space['coint_p_value'] = spaces.Box(low = self.indicator_obs['coint_p_value'][0], high= self.indicator_obs['coint_p_value'][1], shape = (1,))

        if 'z_score' in self.indicator_obs.keys():
            self.observation_space['z_score'] = spaces.Box(low=self.indicator_obs['z_score'][0], high=self.indicator_obs['z_score'][1], shape=(1,))

        #if the z_score it present, add it to the observation space
        if 'z_score' in self.timeseries_obs.keys():
            self.observation_space['z_score'] = spaces.Box(low = self.timeseries_obs['z_score'][1], high= self.timeseries_obs['z_score'][2], shape= (self.timeseries_obs['z_score'][0],))

        #populate the discrete keys
        for key in self.discrerete_obs.keys():
            self.observation_space[key] = spaces.Discrete(self.discrerete_obs[key])

        self.process = MeanRevertingProcess(sigma=0.5)

        assert not ('z_score' in self.indicator_obs.keys() and 'z_score' in self.timeseries_obs.keys()), "z_score cannot be both a timeseries observation and an indicator observation"

        #call reset function
        self.state, _ = self.reset()
    
    def reward_function(self,action):
        """
        Calculate the reward for the current step
        params:
        current_value: float - The current portfolio value
        previous_value: float - The previous portfolio value
        returns:
        reward: float - The reward for the current step
        """
        # Create dictionary for state

        z_score = self.state.get('z_score',None)
        adfuller1 = self.state.get(self.token_pair[0]+"_adfuller",None)
        adfuller2 = self.state.get(self.token_pair[1]+"_adfuller",None)
        coint_p_value = self.state.get('coint_p_value',None)

        low_bound = -2
        high_bound = 0
        is_bought = self.is_bought
        reward = 0

        if action == 1:  # Buy or Sell decision
            if is_bought == 0:  # Buying scenario
                if z_score < low_bound:  
                    reward += 1  # Correct buy timing
                else:
                    reward -= 1  # Incorrect buy timing (bought too early)
            
            elif is_bought == 1:  # Selling scenario
                if z_score > high_bound:  
                    reward += 1  # Correct sell timing
                else:
                    reward -= 1  # Incorrect sell timing (sold too early)

        else:  # Action = 0 (Hold)
            if is_bought == 1 and z_score > high_bound:
                reward += 0.5  # Holding when waiting to sell is good
            elif is_bought == 0 and z_score < low_bound:
                reward += 0.5  # Holding when waiting to buy is good
            else:
                reward -= 0.2  # Slight penalty for indecision

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
        
        #populate the timeseries information for all tokens
        for token in self.token_pair:
            for key in self.timeseries_obs.keys():
                if key != 'z_score':
                    state[token+'_'+key] = np.random.randn(self.timeseries_obs[key][0])

        #populate the cointegreation p calue once
        if 'coint_p_value' in self.indicator_obs.keys():
            state['coint_p_value'] = np.array([np.random.randint(2)])

        #populate the historical z scores once
        if 'z_score' in self.timeseries_obs.keys():
            state['z_score'] = np.random.randn(self.timeseries_obs['z_score'][0])

        #populate the z score
        if 'z_score' in self.indicator_obs.keys():
            state['z_score'] = np.array([self.process.step()])

        #populate the adfuller metrics once
        if 'adfuller' in self.indicator_obs.keys():
            state[self.token_pair[0]+"_adfuller"] = np.array([np.random.randint(2)])
            state[self.token_pair[1]+"_adfuller"] = np.array([np.random.randint(2)])

        #populate the discrete keys
        if 'previous_action' in list(self.observation_space.keys()):
            state['previous_action'] = action
        if 'is_bought' in list(self.observation_space.keys()):
            state['is_bought'] = self.is_bought

        #populate amount_bought indicator
        if 'amount_bought' in list(self.observation_space.keys()):
            state['amount_bought'] = np.array([self.amount_bought])

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

        self.is_bought = 0
        self.amount_bought = 0
        self.previous_action = 0
        
        self.step_num = 0
        
        #generate initial observation
        self.state = self.generate_observation(0)

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
        reward = self.reward_function(action)

        if action == 1:
            if self.is_bought == 1:
                self.is_bought = 0
            elif self.is_bought == 0:
                self.is_bought == 1

        self.step_num+=1
    
        #step internal time
        done = self.step_num >= self.episode_length
    
        
        #generate next observation
        self.state = self.generate_observation(action)
        # Return uselesss variables
        truncated = False  
        info = {}

        self.previous_action = action

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

        
  

