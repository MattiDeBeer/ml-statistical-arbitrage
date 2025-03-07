#from binance import Client, ThreadedWebsocketManager, ThreadedDepthCacheManager
import numpy as np
import matplotlib.pyplot as plt 
import copy
from nolds import hurst_rs

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

.align_timeseries(series1,series2)
Aligns the timeseries data such that each index corresponds to the same time.



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
        
        
    def get_sin_wave_dataset(self, num_data_points, period = 0.01, noise = 0,bin_size = 10,return_data = False):
        
        x = np.linspace(0,1,num_data_points*bin_size)
        y = 0.5*np.sin(2 * np.pi * x / period) + np.random.uniform(0,noise,num_data_points*bin_size) + (1+noise)*np.ones(num_data_points*bin_size)
        
        klines = {}
        
        klines['open'] = y[0::bin_size]
        klines['close'] = y[bin_size - 1 :: bin_size]
        binned_data = y.reshape(-1,bin_size)
        klines['high'] = np.max(binned_data, axis=1)
        klines['low'] = np.min(binned_data, axis=1)
        
        self.dataset_klines = {}
        self.dataset_klines['SIN'] = klines
        self.max_time = num_data_points
        
        if return_data:
            return self.dataset_klines
        
        
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
        
        self.dataset_klines = {}
        self.dataset_klines['SIN'] = klines
        self.max_time = num_data_points
        
        if return_data:
            return self.dataset_klines
        
 
    def align_time_series(self,pair,return_data=False):
        """
        Parameters
        ----------
        pair : string
            The token pair you which to align, formatted at token1-token2
            passing pair = 'all' will align all token datasets to a common time
        return_data : book, optional
            Specifies if you which to return the aligned data The default is False.

        Returns
        -------
        TYPE: dict
            A dict of all the price data, now aligned in time.
            
        Description
        -----------
        This function aligns the price data in time, so each index of each token corresponds
        to the same time.
        This overwrites the data in self.dataset_klines

        """
        def filter_dict(data_dict, common_times):
            time_array = np.array(data_dict['time'])
            indices = np.isin(time_array, common_times)
            filtered_dict = {'time': time_array[indices]}
            for key, values in data_dict.items():
                if key != 'time':
                    filtered_dict[key] = np.array(np.array(values)[indices])
                    
            return filtered_dict
        
        if pair != 'all':
            ticker1, ticker2 = pair.split("-")
            dict1 = self.dataset_klines[ticker1]
            dict2 = self.dataset_klines[ticker2]
        
            # Extract the time entries
            time1 = np.array(dict1['time'])
            time2 = np.array(dict2['time'])
        
            # Find the common time values
            common_times = np.intersect1d(time1, time2)
        
            # Filter both dictionaries
            aligned_dict1 = filter_dict(dict1, common_times)
            aligned_dict2 = filter_dict(dict2, common_times)
            
            self.dataset_klines[ticker1] = aligned_dict1
            self.dataset_klines[ticker2] = aligned_dict2
            self.max_time = len(common_times)
        
            if return_data:
                return aligned_dict1, aligned_dict2
        
        else:
            # Extract the 'time' arrays from all dictionaries
            all_times = [set(np.array(data['time'])) for data in self.dataset_klines.values()]
        
            # Find the common time values across all dictionaries
            common_times = sorted(set.intersection(*all_times))
        
            # Helper function to filter a single dictionary by the common times
            def filter_dict(data_dict, common_times):
                time_array = np.array(data_dict['time'])
                indices = np.isin(time_array, common_times)
                filtered_dict = {'time': time_array[indices]}
                for key, values in data_dict.items():
                    if key != 'time':
                        filtered_dict[key] = np.array(values)[indices]
                return filtered_dict
        
            # Align all sub-dictionaries to the common time indices
            aligned_klines = {
                ticker: filter_dict(data, common_times) for ticker, data in self.dataset_klines.items()
            }
            self.max_time = len(common_times)
            self.dataset_klines = aligned_klines
        
            if return_data:
                return aligned_klines
            
            
    def get_historical_prices(self,symbols,amount,return_data=False):
        """
        Parameters
        ----------
        symbols : tuple
            A tuple of strings for the tokens you whish to get price data for
        amount : int
            The n previous datapoint you whish to get the prices for
        return_data : bool, options
             If true, the price data will be returned by the funciton. The default is False.

        Returns
        -------
        TYPE : Dictionary
            All gathered price data for the current time
            
        Desctiption
        -----------
        This funciton gathers the historical data from the time specified by the model.
        It will set time to amount if time is 0
        It stores gathered prices internally in self.klines

        """
        assert(amount < self.max_time), "Can't chosse a window that is larger than the dataset"
        if self.time > self.max_time:
            return None
        
        if not hasattr(self,'klines'):
            self.klines = {}
        
        if self.time == 0:
            self.time = amount
            
            
        if isinstance(symbols,str):
            assert symbols in self.dataset_klines.keys(), f"No data with key {symbols} has been found"
            tmp_dict = {}
            for key, values in self.dataset_klines[symbols].items():
                tmp_dict[key] = values[self.time-amount:self.time]
                
            self.klines[symbols] = tmp_dict
            
        else:
        
            for symbol in symbols:
                assert symbol in self.dataset_klines.keys(), f"No data with key {symbol} has been found"
                tmp_dict = {}
                for key, values in self.dataset_klines[symbol].items():
                    tmp_dict[key] = values[self.time-amount:self.time]
                    
                self.klines[symbol] = tmp_dict
            
        if return_data:
            return self.klines
        
            
    def generate_arbritrage_pair(self,pair,alpha=None,lag = 0,return_data=False):
        """

        Parameters
        ----------
        pair : str
            The pair you whish to arbritrage. Formatted as 'token1-token2'
        alpha : float
            The hedge ratio, optional
        lag : int, optional
            The amout you whish the first timeseries to lag the second. The default is 0.
        return_data : bool, optional
            Specified if you whish to return the data. The default is False.

        Returns
        -------
        tmp_dict : dict
            The generatied arbritrage data
            
        Description
        -----------
        This funciton generates an arbritrage timeseries for a given metric.
        It uses the data gathered by the self.get_prices() function,
        so this function must be run first.

        """

        assert lag >= 0
        ticker1, ticker2 = pair.split("-")
        
        if not hasattr(self, 'arbritrage_pairs'):
            self.arbritrage_pairs = {}
            
        if alpha is None:
            alpha = np.mean(np.array(self.klines[ticker1]['open'])) / np.mean(np.array(self.klines[ticker2]['open']))
        
        tmp_dict = {}
        for key in self.klines[ticker1]:
                
            series1 = np.array(self.klines[ticker1][key])
            series2 = np.array(self.klines[ticker2][key])
            
            
            if lag != 0:
                series1 = series1[lag:]
                series2 = series2[:-lag]
                
            if key != 'volume' and key != 'time':
                tmp_dict[key] = series1 - alpha*series2
            elif key =='time':
                tmp_dict[key] = series1
            
        self.arbritrage_pairs[pair] = tmp_dict
        return tmp_dict
    
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
        if not symbol in self.dataset_klines.keys():
            print("No data exists for this token")
        else:
            return self.dataset_klines[symbol][key][self.time-1]
        
        
    def buy_token(self,token,amount,return_data=False):
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
        
        if not token in self.dataset_klines.keys():
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
                
            #print(f"Bought {token_amount} of {token} at ${price} (cost ${amount})")
            
            if return_data:
                return token_amount
                
    def short_token(self,token,amount,return_data = False):
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
        if not token in self.dataset_klines.keys():
            print('No token exists with this name')
        else:
            
            discounted_amount = amount*(1-self.transaction_percentage)
            price = self.get_current_price(token)
            token_amount = discounted_amount /price
            self.money += discounted_amount
            
            #print(f"Shorted {token_amount} of {token} at ${price} (position value ${discounted_amount})")
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
            self.time = self.max_time
            return False
        else:
            self.time += time_step
            return True
        
    
    def close_all_positons(self):
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
                #print(f"Sold {token_amount} {key} for ${discounted_gain}")
            else:
                token_amount = -1*token_amount
                loss = token_amount * token_price
                additional_loss = loss + loss*self.transaction_percentage
                self.money -= additional_loss
                #print(f"Bought {token_amount} {key} for ${additional_loss}")
            
            
        self.positions = {}
        
    
    
            
        
        
