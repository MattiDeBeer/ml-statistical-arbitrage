#from binance import Client, ThreadedWebsocketManager, ThreadedDepthCacheManager
import numpy as np
from statsmodels.tsa.stattools import adfuller, coint
import h5py

"""
Desctiprion
-----------
This is an object that simulates a trading enviroment.
It uses either generated data or historical price data.


EXAMPLE USAGE
.get_sin_wave_dataset(window_length) or .get_complex_sin_wave_dataset()
Initialises the model with generated price data for a token 'SIN'.
Hyperparameters such as dataset length and bin size can be specified as arguments.

self.time = 0
starts the models internal time at the start of the dataset


.get_historical_prices(amount)
Will return the specified amount of historical datapoints, with the 
final index corresponding to the price at the current model timestep.

.step_time(amount)
Increments the enviroments internal time.
This simulates real world time passing

.buy_token(token) .short_token(token) .close_all_positions()
These are used to buy, sell and close the current positions at the models
current price (specified by the dataset and it's internal time')
"""

class BinanceTradingEnv: 
   
    def __init__(self):
        
        #The instantiation of the client is not serialisable. so this has been commented out
        #funcitonality for realtime binance data will be added later
        """
        Initilises a wrapper with the binance api client using the keys

        Returns
        -------
        None
        
        """

        #spcifies the model starting time
        self.time = 0
        #specifiec the default model funds (USD)
        self.money = 1
        #specified default model transaction fee (0.1%)
        self.transaction_percentage = 0.001
        #creates an empty dictionary for model positions
        self.positions = {}
        #print("Binance Enviroment initialized")
        
        
    def __repr__(self):
        """

        Returns
        -------
        string
            A representation of the current state of the object.

        """
        representation = {}
        if not hasattr(self,'dataset_filename'):
            representation['Dataset file'] = None
        else:
            representation['Dataset file'] = self.dataset_filename
        
        if not hasattr(self,'dataset_klines'):
            representation['Loaded tokens'] = None
        else:
            representation['Loaded Tokens'] =list(self.dataset_klines.keys())
            representation['Available datasets'] = [list(self.dataset_klines[key].keys()) for key in list(self.klines.keys())]
            
        if hasattr(self, 'klines'): 
            representation['Tokens loaded in episode'] = list(self.klines.keys())
            representation['Current (open) prices'] = [{symbol: self.get_current_price(symbol)} for symbol in list(self.klines.keys())]
            try:
                representation['Previous (open) prices'] = [{symbol: self.get_historical_prices(symbol,5,return_data=True)[symbol]['open']} for symbol in list(self.klines.keys())]
            except:
                representation['Previous (open) prices'] = [{symbol: 'Insufficent time for 5 historical pices'} for symbol in list(self.klines.keys())]
                
        else:
            representation['Tokens loaded in episode'] = None
            representation['Current (open) prices'] = None
            
        representation['Time'] = self.time
        representation['Transaction Prececntage'] = self.transaction_percentage
        representation['Money'] = self.money
        representation['Open positions'] = self.positions
        
        return str(representation)
        
        
    def get_sin_wave_dataset(self, num_data_points, period = 0.01, noise = 0,bin_size = 10,return_data = False):
        x = np.linspace(0,1,num_data_points*bin_size)
        y = 0.5*np.sin(2 * np.pi * x / period) + np.random.uniform(0,noise,num_data_points*bin_size) + (1+noise)*np.ones(num_data_points*bin_size)
        
        
        klines = {}
        
        klines['open'] = y[0::bin_size]
        klines['close'] = y[bin_size - 1 :: bin_size]
        binned_data = y.reshape(-1,bin_size)
        klines['high'] = np.max(binned_data, axis=1)
        klines['low'] = np.min(binned_data, axis=1)
        
        if not hasattr(self, 'klines'):
            self.klines = {}
            
        self.klines['SIN'] = klines
        self.max_time = num_data_points
        
        if return_data:
            return self.klines
        
        
    def get_complex_sin_wave_dataset(self, num_data_points, noise = 0,bin_size = 10,return_data = False):
        """

        Parameters
        ----------
        num_data_points : int
            The number of historical datapoints to be used
        noise : float, optional
            The madnitude of noise to be used by the model. The default is 0.
        bin_size : int, optional
            The amount of points samples over so sample the high, low, open and close from.
            The default is 10.
        return_data : bool, optional
            Specifies if the generated dataset should be returned. The default is False.

        Returns
        -------
        dict
            The generated dataset.

        """

        number_of_waves = np.random.randint(low=2, high = 5)
        y = np.zeros(num_data_points*bin_size)
        x = np.linspace(0,1,num_data_points*bin_size)
        
        for i in range(0,number_of_waves):
            amplitude = np.random.uniform(0,1)
            period = np.random.uniform(0.1,0.3)
            y += amplitude * np.sin(2 * np.pi * x / period)
            
        y = 1 + y/(number_of_waves*2)
            
        y += noise*np.random.uniform(0,1,num_data_points*bin_size)
        
        klines = {}
        
        klines['open'] = y[0::bin_size]
        klines['close'] = y[bin_size - 1 :: bin_size]
        binned_data = y.reshape(-1,bin_size)
        klines['high'] = np.max(binned_data, axis=1)
        klines['low'] = np.min(binned_data, axis=1)
        
        if not hasattr(self, 'klines'):
            self.klines = {}
            
        self.klines['SIN'] = klines
        self.max_time = num_data_points
        
        if return_data:
            return self.klines
        
    def load_token_dataset(self, filename,directory = 'data/'):
        self.dataset_filename = directory+filename
        if not hasattr(self,'dataset_klines'):
            self.dataset_klines = {}
        with h5py.File(directory+filename, 'r') as f:
            for token in list(f.keys()):
                token_klines = {}
                ### ADD CHECK TO ENSURE ALL TIMESERIES HAVE THE SAME LENGTH ###
                for timeseries_key in list(f[token].keys()):
                    token_klines[timeseries_key] = f[token][timeseries_key][:]
                    self.dataset_length = token_klines[timeseries_key].shape[0]
                self.dataset_klines[token] = token_klines
                
    ### Need to change klines to episode_klines ###
    def get_token_episode(self,tokens,length,return_data = False):
        if not hasattr(self,'episode_klines'):
            self.klines = {}
            
        if isinstance(tokens,str):
            token = tokens
            assert token in self.dataset_klines.keys(), "Token {token} does not exist in the dataset"
            klines = {}
            start = np.random.randint(0,self.dataset_length - length - 1)
            for key in self.dataset_klines[token]:
                klines[key] = self.dataset_klines[token][key][start:start+length]
                self.max_time = klines[key].shape[0]
            
            self.klines[token] = klines
            if return_data:
                return self.klines
        
        elif isinstance(tokens,(list,tuple)):
            start = np.random.randint(0,self.dataset_length - length - 1)
            for token in tokens:
                assert token in self.dataset_klines.keys(), "Token {token} does not exist in the dataset"
                klines = {}
                for key in self.dataset_klines[token]:
                    klines[key] = self.dataset_klines[token][key][start:start+length]
                    self.max_time = klines[key].shape[0]
                
                self.klines[token] = klines
            if return_data:
                return self.klines
        else:
            raise ValueError("input {tokens} is neither a string, tuple or array")
            

    def get_historical_prices(self,symbols,amount):
        """
        

        Parameters
        ----------
        symbols : TYPE
            DESCRIPTION.
        amount : TYPE
            DESCRIPTION.

        Raises
        ------
        ValueError
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        assert(amount < self.max_time), "Can't chosse a window that is larger than the dataset"
        
        if not hasattr(self,'historical_klines'):
            self.historical_klines = {}
            
        assert amount < self.time, f"You requested {amount} datapoints, at time {self.time}. Ensure your historical request is smaller than the current time."
            
            
        if isinstance(symbols,str):
            assert symbols in self.klines.keys(), f"No data with key {symbols} has been found"
            tmp_dict = {}
            for key, values in self.klines[symbols].items():
                tmp_dict[key] = values[1+self.time-amount:self.time+1]
                
            self.historical_klines[symbols] = tmp_dict
            
            return self.historical_klines
            
        elif isinstance(symbols,(list,tuple)):
        
            for symbol in symbols:
                assert symbol in self.klines.keys(), f"No data with key {symbol} has been found"
                tmp_dict = {}
                for key, values in self.klines[symbol].items():
                    tmp_dict[key] = values[self.time-amount:self.time]
                    
                self.historical_klines[symbol] = tmp_dict
            
            return self.historical_klines
        
        else:
            raise ValueError("The passed input {symbols} is neither a string, tuple or list")
        
    def calc_hedge_ratio(self,timeseries1,timeseries2):
        ### This can be changed to another method ###
        return np.mean(timeseries1) / np.mean(timeseries2)
    
    def get_z_scores(self,token1,token2,length,return_data=False,excluded_keys = ['log_return_high','log_return_low','log_return_open','log_return_close','volume','time']):
        assert isinstance(token1,str) and isinstance(token2,str), "Tokens must be strings"
        assert token1 in self.klines and token2 in self.klines, "The token was not found in the loaded episode. Ensure an episode with the required keys are loaded"
        
        klines = self.get_historical_prices([token1,token2], length)
        
        z_scores_dict = {}
        for key in klines[token1].keys():
            if key not in excluded_keys:
                timeseries1 = klines[token1][key]
                try:
                    timeseries2 = klines[token2][key]
        
                    hedge_ratio = self.calc_hedge_ratio(timeseries1,timeseries2)
                    
                    spread = timeseries1 - hedge_ratio * timeseries2
                    z_scores = (spread - np.mean(spread)) / np.std(spread)
                    z_scores_dict[key] = z_scores
                except KeyError as e:
                    print(f"Timeseries {key}, was found in the {token1} dataset but not the {token2}")
                    z_scores_dict[key] = None
        
        return z_scores_dict

                
    def calc_coint_values(self,token1,token2,length,key='open'):
        assert isinstance(token1,str) and isinstance(token2,str), "Tokens must be specified as string"
        assert token1 in self.dataset_klines and token1 in self.dataset_klines, "Specified tokens were not fount in the loaded dataset"
        
        klines = self.get_historical_prices([token1,token2],length)
        asset1 = klines[token1][key]
        asset2 = klines[token2][key]
        
        #preform adf on asset prices
        adf_result_asset1 = adfuller(asset1)
        adf_result_asset2 = adfuller(asset2)

        # If p-value of these is greater than 0.05, the series is non-stationary
        #It may be a good idea to also include these in the state space

        # Step 3: Engle-Granger Cointegration Test (Two-Step Approach)
        score, p_value, _ = coint(asset1, asset2)
        
        return p_value, adf_result_asset1, adf_result_asset2
        
    
    def get_current_portfolio_value(self):
        
        asset_values = 0
    
        for key in self.positions.keys():
            token_amount = self.positions[key]
            token_price = self.get_current_price(key)
            if token_amount > 0:
                gain = token_amount * token_price
                discounted_gain = gain - gain*self.transaction_percentage
                asset_values += discounted_gain
            else:
                token_amount = -1*token_amount
                loss = token_amount * token_price
                additional_loss = loss + loss*self.transaction_percentage
                asset_values -= additional_loss
               
        return self.money + asset_values
        
    
    def get_current_price(self,symbol,key = 'open'):
        """

        Parameters
        ----------
        symbol : srting
            The token you want the price fot
        key : str, optional
            The price type you want e.g high, low, close. The default is 'open'.

        Returns
        -------
        TYPE: float
            The current token price as determined by the internal time

        """
        if not symbol in self.klines.keys():
            print("No data exists for this token")
            return None
        else:
            return self.klines[symbol][key][self.time]
        
        
    def buy_token(self,token,amount,return_data=False,verbose=False):
        """
        Parameters
        ----------
        token : string
            The token you which to buy
        amount : float
            The USD value you whish to spend buying the token
        return_data : float, optional
            True if you whish to return the amount of token baught. The default is False.

        Returns
        -------
        token_amount : float
            The quantity of the token you bought, accouting for transaction costs

        """
        
        if not token in self.klines.keys():
            print('No token exists with this name')
        else:
            
            discounted_amount = amount*(1-self.transaction_percentage)
            price = self.get_current_price(token)
            token_amount =  discounted_amount /price
            self.money -= amount
            
            if token in self.positions.keys():
                self.positions[token] += token_amount
            else:
                self.positions[token] = token_amount
             
            if verbose:
                print(f"Bought {token_amount} of {token} at ${price} (cost ${amount})")
            
            if return_data:
                return token_amount
                
    def short_token(self,token,amount,return_data = False,verbose=False):
        """
        Parameters
        ----------
        token : string
            The token you which to short
        amount : float
            The USD value you whish to short
        return_data : float, optional
            True if you whish to return the amount of token baught. The default is False.

        Returns
        -------
        token_amount : float
            The quantity of the token you shorted, accounting for transaciton costs

        """
        if not token in self.klines.keys():
            print('No token exists with this name')
        else:
            
            discounted_amount = amount*(1-self.transaction_percentage)
            price = self.get_current_price(token)
            token_amount = discounted_amount /price
            self.money += discounted_amount
            
            if verbose:
                print(f"Shorted {token_amount} of {token} at ${price} (position value ${discounted_amount})")
            
            if token in self.positions.keys():
                self.positions[token] -= token_amount
            else:
                self.positions[token] = -token_amount
            
            if return_data:
                return token_amount
                
    def step_time(self,time_step):
        """

        Parameters
        ----------
        time_step : int
            The amount of timesteps you which to increment the models steps by

        Returns
        -------
        bool
            True if the model time has not exceeded the dataset size, false of it has

        """
        if self.time + time_step >= self.max_time:
            #print(f"Max time exceeded, setting time to final dataset time ({self.max_time})")
            self.time = self.max_time-1
            return False
        else:
            self.time += time_step
            return True
        
    
    def close_all_positions(self,verbose = False):
        """
        Returns
        -------
        None.
        
        Desctiption
        -----------
        Closes all posiitons held by the model at the current prices.
        This will sell any held tokens (accounting for transaction fees)
        It will cover any short positions (accounting for transaction fees)

        """
        for key in self.positions.keys():
            token_amount = self.positions[key]
            token_price = self.get_current_price(key)
            if token_amount > 0:
                gain = token_amount * token_price
                discounted_gain = gain - gain*self.transaction_percentage
                self.money += discounted_gain
                if verbose:
                    print(f"Sold {token_amount} {key} for ${discounted_gain}")
            else:
                token_amount = -1*token_amount
                loss = token_amount * token_price
                additional_loss = loss + loss*self.transaction_percentage
                self.money -= additional_loss
                if verbose:
                    print(f"Bought {token_amount} {key} for ${additional_loss}")

        self.positions = {}
        
    
    
            
        
        
