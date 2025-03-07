from binance import Client, ThreadedWebsocketManager, ThreadedDepthCacheManager
import numpy as np
import matplotlib.pyplot as plt 
import copy
from nolds import hurst_rs


"""
EXAMPLE USAGE

#ensure a file called keys.txt is in the local directory and contains binance keys
#creates a wrapper object
wrapper = api_wrapper() 


#gets and stores specified price dataset in wrapper.dataset_klines from binance
wrapper.get_second_prices_dataset((token1, token2, . . . ), window_length)


#retrieves data from wrapper object
price_data = wrapper.get_prices(('BTCUSDT','SOLUSDT'), 100,return_data=True)

This returns the price data for a certian winow for a specific time.
Running this will return the first 100 prices in the dataset
If time is 0, it will set time to 100 and return the 100 last prices
This function follows the internal wrapper time, so if time is set to 1000,
it will return the prices from indec 900 - 1000

#generates an arbritrage pair for a specified combination and specified hedge_ratio
wrapper.generate_arbritrage_pair(token1+'-'+token2, hedge_ratio,return_data=True)

This will return the arbitrage of the recent prices as specified in the code above

#Gets current token price in terms of the internal time
price = wrapper.get_current_price('BTCUSDT') 

#steps model time forward by 10 datapoints
wrapper.step_time(10)

#Buys amount of token at current price
wrapper.buy_token(token,amount)

amount is specified in USD, and transaction fees are accounted for

#Shorts specified USD value of token, including transcation fees
wrapper.short_token(token,amount)

#closes all currently open positions
wrapper.close_all_positions

for example, it will cover any shorts by buying the token, and selling any held tokens
"""


class binance_api: 
    '''
    Defines a wrapper object for the binance api
    '''
    
    def __init__(self,key_filename = 'keys.txt'):
        '''
        Initilises a wrapper with the binance api client using the keys

        Returns
        -------
        None.

        '''
        #reads keys from file
        with open(key_filename) as key_file:
            key = key_file.readline().strip('\n')
            secret = key_file.readline().strip('\n')
        
        self.__api_key = key
        self.__api_secret = secret
        
        #creates a binance client object for data collection
        self.client = Client(self.__api_key, self.__api_secret)
        
        #spcifies the model starting time
        self.time = 0
        #specifiec the default model funds (USD)
        self.money = 1
        #specified default model transaction fee (0.1%)
        self.transaction_percentage = 0.001
        #creates an empty dictionary for model positions
        self.positions = {}
        print("Binance API initialized")
        
        
    def get_minute_prices_dataset(self,symbols,amount,return_data = False):
        
        '''
        
        Parameters
        ----------
        symbols : Array of strincg
            An array of the required token prices e.g BTCUSDT.
        amount : Integer
            The amount of historical datapoints you require.

        Returns
        -------
        klines : Dictonary
            A Dictonary of the historical price data.
            This is stored as a 2d dictionary of values over the given interval.
            e.g klines['BTCUSDT']['open'] is a 1d numpy array of open prices over the interval

        '''
        assert isinstance(symbols, tuple) and all(isinstance(item, str) for item in symbols), "symbols must be a tuple containing only strings"
        assert isinstance(amount, int), "amount must be an integer"
        
        klines = {}
    
        for symbol in symbols:
            print('Fetching {0} minute prices for the last {1} minutes'.format(symbol,amount))
            labeled_price_data = {}
            price_data = np.array(self.client.get_historical_klines(symbol, Client.KLINE_INTERVAL_1MINUTE, "{0} minutes ago UTC".format(amount)))
            price_data = price_data.T
            price_data = price_data.astype(float)
            
            labeled_price_data['time'] = np.array(price_data[0])
            labeled_price_data['open'] = np.array(price_data[1])
            labeled_price_data['high'] = np.array(price_data[2])
            labeled_price_data['low'] = np.array(price_data[3])
            labeled_price_data['close'] = np.array(price_data[4])
            labeled_price_data['volume'] = np.array(price_data[5])
            
            klines[symbol] = labeled_price_data  
            
        self.dataset_klines = klines
        if return_data:
            return self.dataset_klines
    
    
    def get_second_prices_dataset(self,symbols,amount,return_data = False):
        
        '''
        
        Parameters
        ----------
        symbols : Array of strincg
            An array of the required token prices e.g BTCUSDT.
        amount : Integer
            The amount of historical datapoints you require.

        Returns
        -------
        klines : Dictonary
            A Dictonary of the historical price data.
            This is stored as a 2d dictionary of values over the given interval.
            e.g klines['BTCUSDT']['open'] is a 1d numpy array of open prices over the interval

        '''
        assert isinstance(symbols, tuple) and all(isinstance(item, str) for item in symbols), "symbols must be a tuple containing only strings"
        assert isinstance(amount, int), "amount must be an integer"
        
        klines = {}
    
        for symbol in symbols:
            print('Fetching {0} second prices for the last {1} seconds'.format(symbol,amount))
            labeled_price_data = {}
            price_data = np.array(self.client.get_historical_klines(symbol, Client.KLINE_INTERVAL_1SECOND, "{0} seconds ago UTC".format(amount)))
            price_data = price_data.T
            price_data = price_data.astype(float)
            
            labeled_price_data['time'] = np.array(price_data[0])
            labeled_price_data['open'] = np.array(price_data[1])
            labeled_price_data['high'] = np.array(price_data[2])
            labeled_price_data['low'] = np.array(price_data[3])
            labeled_price_data['close'] = np.array(price_data[4])
            labeled_price_data['volume'] = np.array(price_data[5])
            
            klines[symbol] = labeled_price_data  
            
        self.dataset_klines = klines
        if return_data:
            return self.dataset_klines
        
        
    def get_5_minute_prices_dataset(self, symbols, amount, return_data=False):
        """
        Fetches historical 5-minute price data for given symbols.
    
        Parameters
        ----------
        symbols : tuple of str
            A tuple of required token prices (e.g., ('BTCUSDT', 'ETHUSDT')).
        amount : int
            The number of historical 5-minute intervals required.
        return_data : bool, optional
            If True, returns the dataset.
    
        Returns
        -------
        klines : dict
            A dictionary containing historical price data.
            Structured as a nested dictionary:
            e.g., klines['BTCUSDT']['open'] is a 1D NumPy array of open prices.
        """
        assert isinstance(symbols, tuple) and all(isinstance(item, str) for item in symbols), "symbols must be a tuple containing only strings"
        assert isinstance(amount, int), "amount must be an integer"
    
        klines = {}
    
        for symbol in symbols:
            print(f'Fetching {amount} 5-minute prices for {symbol}')
            labeled_price_data = {}
    
            # Fetch historical 5-minute Kline (candlestick) data
            price_data = np.array(
                self.client.get_historical_klines(symbol, Client.KLINE_INTERVAL_5MINUTE, f"{amount * 5} minutes ago UTC")
            )
    
            # Transpose and convert to float
            price_data = price_data.T.astype(float)
    
            # Store labeled data
            labeled_price_data['time'] = np.array(price_data[0])      # Timestamp
            labeled_price_data['open'] = np.array(price_data[1])      # Open price
            labeled_price_data['high'] = np.array(price_data[2])      # High price
            labeled_price_data['low'] = np.array(price_data[3])       # Low price
            labeled_price_data['close'] = np.array(price_data[4])     # Close price
            labeled_price_data['volume'] = np.array(price_data[5])    # Volume
    
            klines[symbol] = labeled_price_data  
    
        self.dataset_klines = klines
        if return_data:
            return self.dataset_klines


    def get_15_minute_prices_dataset(self, symbols, amount, return_data=False):
        """
        Fetches historical 15-minute price data for given symbols.
    
        Parameters
        ----------
        symbols : tuple of str
            A tuple of required token prices (e.g., ('BTCUSDT', 'ETHUSDT')).
        amount : int
            The number of historical 15-minute intervals required.
        return_data : bool, optional
            If True, returns the dataset.
    
        Returns
        -------
        klines : dict
            A dictionary containing historical price data.
            Structured as a nested dictionary:
            e.g., klines['BTCUSDT']['open'] is a 1D NumPy array of open prices.
        """
        assert isinstance(symbols, tuple) and all(isinstance(item, str) for item in symbols), "symbols must be a tuple containing only strings"
        assert isinstance(amount, int), "amount must be an integer"
    
        klines = {}
    
        for symbol in symbols:
            print(f'Fetching {amount} 15-minute prices for {symbol}')
            labeled_price_data = {}
    
            # Fetch historical 15-minute Kline (candlestick) data
            price_data = np.array(
                self.client.get_historical_klines(symbol, Client.KLINE_INTERVAL_15MINUTE, f"{amount * 15} minutes ago UTC")
            )
    
            # Transpose and convert to float
            price_data = price_data.T.astype(float)
    
            # Store labeled data
            labeled_price_data['time'] = np.array(price_data[0])      # Timestamp
            labeled_price_data['open'] = np.array(price_data[1])      # Open price
            labeled_price_data['high'] = np.array(price_data[2])      # High price
            labeled_price_data['low'] = np.array(price_data[3])       # Low price
            labeled_price_data['close'] = np.array(price_data[4])     # Close price
            labeled_price_data['volume'] = np.array(price_data[5])    # Volume
    
            klines[symbol] = labeled_price_data  
    
        self.dataset_klines = klines
        if return_data:
            return self.dataset_klines

    def get_hour_prices_dataset(self, symbols, amount, return_data=False):
        """
        Fetches historical 1-hour price data for given symbols.
    
        Parameters
        ----------
        symbols : tuple of str
            A tuple of required token prices (e.g., ('BTCUSDT', 'ETHUSDT')).
        amount : int
            The number of historical 1-hour intervals required.
        return_data : bool, optional
            If True, returns the dataset.
    
        Returns
        -------
        klines : dict
            A dictionary containing historical price data.
        """
        assert isinstance(symbols, tuple) and all(isinstance(item, str) for item in symbols), "symbols must be a tuple containing only strings"
        assert isinstance(amount, int), "amount must be an integer"
    
        klines = {}
    
        for symbol in symbols:
            print(f'Fetching {amount} 1-hour prices for {symbol}')
            labeled_price_data = {}
    
            # Fetch historical 1-hour Kline (candlestick) data
            price_data = np.array(
                self.client.get_historical_klines(symbol, Client.KLINE_INTERVAL_1HOUR, f"{amount} hours ago UTC")
            )
    
            # Transpose and convert to float
            price_data = price_data.T.astype(float)
    
            # Store labeled data
            labeled_price_data['time'] = np.array(price_data[0])      # Timestamp
            labeled_price_data['open'] = np.array(price_data[1])      # Open price
            labeled_price_data['high'] = np.array(price_data[2])      # High price
            labeled_price_data['low'] = np.array(price_data[3])       # Low price
            labeled_price_data['close'] = np.array(price_data[4])     # Close price
            labeled_price_data['volume'] = np.array(price_data[5])    # Volume
    
            klines[symbol] = labeled_price_data  
    
        self.dataset_klines = klines
        if return_data:
            return self.dataset_klines


    def get_4_hour_prices_dataset(self, symbols, amount, return_data=False):
        """
        Fetches historical 4-hour price data for given symbols.
    
        Parameters
        ----------
        symbols : tuple of str
            A tuple of required token prices (e.g., ('BTCUSDT', 'ETHUSDT')).
        amount : int
            The number of historical 4-hour intervals required.
        return_data : bool, optional
            If True, returns the dataset.
    
        Returns
        -------
        klines : dict
            A dictionary containing historical price data.
        """
        assert isinstance(symbols, tuple) and all(isinstance(item, str) for item in symbols), "symbols must be a tuple containing only strings"
        assert isinstance(amount, int), "amount must be an integer"
    
        klines = {}
    
        for symbol in symbols:
            print(f'Fetching {amount} 4-hour prices for {symbol}')
            labeled_price_data = {}
    
            # Fetch historical 4-hour Kline (candlestick) data
            price_data = np.array(
                self.client.get_historical_klines(symbol, Client.KLINE_INTERVAL_4HOUR, f"{amount * 4} hours ago UTC")
            )
    
            # Transpose and convert to float
            price_data = price_data.T.astype(float)
    
            # Store labeled data
            labeled_price_data['time'] = np.array(price_data[0])      # Timestamp
            labeled_price_data['open'] = np.array(price_data[1])      # Open price
            labeled_price_data['high'] = np.array(price_data[2])      # High price
            labeled_price_data['low'] = np.array(price_data[3])       # Low price
            labeled_price_data['close'] = np.array(price_data[4])     # Close price
            labeled_price_data['volume'] = np.array(price_data[5])    # Volume
    
            klines[symbol] = labeled_price_data  
    
        self.dataset_klines = klines
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
        
            # Helper function to filter a dictionary by the common time values
            
        
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
            
    def generate_log_returns(self,symbols, return_data=False):
        self.log_returns_dataset = {}

        for symbol in symbols:
            if symbol not in self.dataset_klines:
                print(f"Skipping {symbol}, not in dataset.")
                # Skip if the symbol is not in the dataset
            
            self.log_returns_dataset[symbol] = {}  # Initialize nested dictionary
    
            for key, series in self.dataset_klines[symbol].items():
                if key in ["time", "volume"]:
                    self.log_returns_dataset[symbol][key] = series  # Keep time and volume unchanged
                else:
                    self.log_returns_dataset[symbol][key] = np.log(series[1:] / series[:-1]) # Compute log returns
        
        if return_data:
            return self.log_returns_dataset
        
            
    def get_prices(self,symbols,amount,return_data=False):
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
        
        for symbol in symbols:
            assert symbol in self.dataset_klines.keys()
            tmp_dict = {}
            for key, values in self.dataset_klines[symbol].items():
                tmp_dict[key] = values[self.time-amount:self.time]
                
            self.klines[symbol] = tmp_dict
            
        if return_data:
            return self.klines
        
            
    def generate_arbritrage_pair(self,pair,alpha,lag = 0,return_data=False):
        """

        Parameters
        ----------
        pair : str
            The pair you whish to arbritrage. Formatted as token-token2
        alpha : float
            The hedge ratio
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
        This funciton generates an arbrotrage timeseries for a given metric.
        It uses the data gathered by the self.get_prices() function,
        so this function must be run first.

        """

        assert lag >= 0
        ticker1, ticker2 = pair.split("-")
        
        if not hasattr(self, 'arbritrage_pairs'):
            self.arbritrage_pairs = {}
        
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
                
            print(f"Bought {token_amount} of {token} at ${price} (cost ${amount})")
            
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
            print(f"Max time exceeded, setting time to final dataset time ({self.max_time})")
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
                print(f"Sold {token_amount} {key} for ${discounted_gain}")
            else:
                token_amount = -1*token_amount
                loss = token_amount * token_price
                additional_loss = loss + loss*self.transaction_percentage
                self.money -= additional_loss
                print(f"Bought {token_amount} {key} for ${additional_loss}")
            
            
        self.positions = {}
        
    
    
            
        
        
