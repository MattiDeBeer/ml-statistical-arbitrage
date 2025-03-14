import numpy as np
from statsmodels.tsa.stattools import adfuller, coint
import h5py

class BinanceTradingEnv: 
   
    def __init__(self):
        """
        Initilises a wrapper with the binance api client using the keys
        Returns:
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
        
        
    def __repr__(self):
        """
        Returns a string representation of the current state of the object.
        returns:
        str: A string representation of the current state of the object.
        """
        # Initialize an empty dictionary to store the representation
        representation = {}
        
        # Check if the dataset filename attribute exists
        if not hasattr(self,'dataset_filename'):
            # If not, set the 'Dataset file' key to None
            representation['Dataset file'] = None
        else:
            # Otherwise, set it to the dataset filename
            representation['Dataset file'] = self.dataset_filename
        
        # Check if the dataset klines attribute exists
        if not hasattr(self,'dataset_klines'):
            # If not, set the 'Loaded tokens' key to None
            representation['Loaded tokens'] = None
        else:
            # Otherwise, set it to the list of loaded tokens
            representation['Loaded Tokens'] = list(self.dataset_klines.keys())
            # Also, set the 'Available datasets' key to the list of available datasets for each token
            representation['Available datasets'] = [list(self.dataset_klines[key].keys()) for key in list(self.klines.keys())]
        
        # Check if the klines attribute exists
        if hasattr(self, 'klines'): 
            # If it exists, set the 'Tokens loaded in episode' key to the list of tokens loaded in the episode
            representation['Tokens loaded in episode'] = list(self.klines.keys())
            # Set the 'Current (open) prices' key to the current open prices for each token
            representation['Current (open) prices'] = [{symbol: self.get_current_price(symbol)} for symbol in list(self.klines.keys())]
            try:
                # Try to set the 'Previous (open) prices' key to the previous open prices for each token
                representation['Previous (open) prices'] = [{symbol: self.get_historical_prices(symbol,5,return_data=True)[symbol]['open']} for symbol in list(self.klines.keys())]
            except:
                # If there is an exception, set the 'Previous (open) prices' key to a message indicating insufficient time for 5 historical prices
                representation['Previous (open) prices'] = [{symbol: 'Insufficent time for 5 historical pices'} for symbol in list(self.klines.keys())]
        else:
            # If the klines attribute does not exist, set the relevant keys to None
            representation['Tokens loaded in episode'] = None
            representation['Current (open) prices'] = None
        
        # Set the 'Time' key to the current time
        representation['Time'] = self.time
        # Set the 'Transaction Percentage' key to the transaction percentage
        representation['Transaction Prececntage'] = self.transaction_percentage
        # Set the 'Money' key to the current amount of money
        representation['Money'] = self.money
        # Set the 'Open positions' key to the current open positions
        representation['Open positions'] = self.positions
        
        # Return the string representation of the dictionary
        return str(representation)
        
        
    def get_sin_wave_episode(self, num_data_points, period = 0.01, noise = 0,bin_size = 10,return_data = False):
        """
        Generates a synthetic sine wave episode with optional noise and bins the data.
        Parameters:
        num_data_points (int): Number of data points to generate.
        period (float, optional): Period of the sine wave. Default is 0.01.
        noise (float, optional): Amplitude of the uniform noise to add. Default is 0.
        bin_size (int, optional): Size of the bins to aggregate data. Default is 10.
        return_data (bool, optional): If True, returns the generated klines data. Default is False.
        Returns:
        dict: If return_data is True, returns a dictionary containing the generated klines data with keys 'open', 'close', 'high', and 'low'.
        """

        # Make the x values for the sine wave
        x = np.linspace(0,1,num_data_points*bin_size)

        # Generate the sine wave with noise
        y = 0.5*np.sin(2 * np.pi * x / period) + np.random.uniform(0,noise,num_data_points*bin_size) + (1+noise)*np.ones(num_data_points*bin_size)
        
        # Create a dictionary to store the klines data
        klines = {}

        # Calculate the open prices
        klines['open'] = y[0::bin_size]

        # Calculate the close prices
        klines['close'] = y[bin_size - 1 :: bin_size]

        # Bin the data to calculate the high and low prices
        binned_data = y.reshape(-1,bin_size)
        klines['high'] = np.max(binned_data, axis=1)
        klines['low'] = np.min(binned_data, axis=1)
        
        # If the klines attribute does not exist, create it
        if not hasattr(self, 'klines'):
            self.klines = {}
        
        # Add the generated klines data to the klines attribute
        self.klines['SIN'] = klines
        self.max_time = num_data_points
        
        # If return_data is True, return the generated klines data
        if return_data:
            return self.klines
        
        
    def get_complex_sin_wave_episode(self, num_data_points, noise = 0,bin_size = 10,return_data = False):
        """
        Generates a synthetic sine wave episode consisting of multiple sin waves, with optional noise, and bins the data.
        Parameters:
        num_data_points (int): Number of data points to generate.   
        noise (float, optional): Amplitude of the uniform noise to add. Default is 0.
        bin_size (int, optional): Size of the bins to aggregate data. Default is 10.
        return_data (bool, optional): If True, returns the generated klines data. Default is False.
        Returns:
        dict: If return_data is True, returns a dictionary containing the generated klines data with keys 'open', 'close', 'high', and 'low'.
        """

        # Generate a random number of sine waves
        number_of_waves = np.random.randint(low=2, high = 5)

        # Initialize the y values
        y = np.zeros(num_data_points*bin_size)

        # Make the x values for the sine wave
        x = np.linspace(0,1,num_data_points*bin_size)
        
        # Itterates theough the number of waves and adds them to the y values
        for i in range(0,number_of_waves):
            amplitude = np.random.uniform(0,1)
            period = np.random.uniform(0.1,0.3)
            y += amplitude * np.sin(2 * np.pi * x / period)
            
        # Normalises the y values
        y = 1 + y/(number_of_waves*2)
            
        # Adds noise to the y values
        y += noise*np.random.uniform(0,1,num_data_points*bin_size)
        
        klines = {}
        
        #Creates the open, close, high and low values for the klines
        klines['open'] = y[0::bin_size]
        klines['close'] = y[bin_size - 1 :: bin_size]
        binned_data = y.reshape(-1,bin_size)
        klines['high'] = np.max(binned_data, axis=1)
        klines['low'] = np.min(binned_data, axis=1)
        
        # If the klines attribute does not exist, create it
        if not hasattr(self, 'klines'):
            self.klines = {}
            
        # Add the generated klines data to the klines attribute
        self.klines['SIN'] = klines

        # Set the maximum time to the number of data points
        self.max_time = num_data_points
        
        if return_data:
            return self.klines
        
    def load_token_dataset(self, filename='dataset_100000_1m.h5',directory = 'data/'):
        """
        Loads a dataset from a hdf5 file and stores it in the dataset_klines attribute.
        Parameters:
        filename (str): The name of the file to load.
        directory (str, optional): The directory to load the file from. Default is 'data/'.
        Returns:
        None
        """
        # Set the dataset filename attribute
        self.dataset_filename = directory+filename

        # If the dataset klines attribute does not exist, create it
        if not hasattr(self,'dataset_klines'):
            self.dataset_klines = {}

        # Open the hdf5 file
        with h5py.File(directory+filename, 'r') as f:

            # Iterate through the tokens in the file
            for token in list(f.keys()):
                token_klines = {}
                ### ADD CHECK TO ENSURE ALL TIMESERIES HAVE THE SAME LENGTH ###
                for timeseries_key in list(f[token].keys()):
                    token_klines[timeseries_key] = f[token][timeseries_key][:]
                    if 'dataset_length' not in self.__dict__:
                        self.dataset_length = token_klines[timeseries_key].shape[0]
                    else:
                        assert self.dataset_length == token_klines[timeseries_key].shape[0], "The program detected that some of the timeseries in the dataset have different lengths. Ensure all timeseries have the same length"

                # Add the token klines to the dataset klines attribute
                self.dataset_klines[token] = token_klines
                
    def get_token_episode(self,tokens,length,return_data = False):
        """
        Generates an episode of token data for the specified tokens and length.
        Parameters:
        tokens (str, list, tuple): The tokens to generate the episode for.
        length (int): The length of the episode.
        return_data (bool, optional): If True, returns the generated episode data. Default is False.
        Returns:
        dict: If return_data is True, returns a dictionary containing the generated episode data.
        """

        # If the episode klines attribute does not exist, create it
        if not hasattr(self,'klines'):
            self.klines = {}
            
        # If the tokens input is a string
        if isinstance(tokens,str):
            token = tokens

            # Check if the token exists in the dataset
            assert token in self.dataset_klines.keys(), "Token {token} does not exist in the dataset"
            klines = {}

            # Set a random start point in the dataset
            start = np.random.randint(0,self.dataset_length - length - 1)

            # Generate an eposide for all the timeseries in the token
            for key in self.dataset_klines[token]:
                klines[key] = self.dataset_klines[token][key][start:start+length]
                self.max_time = klines[key].shape[0]
            
            # Add the generated episode to the episode klines attribute
            self.klines[token] = klines

            # If return_data is True, return the generated episode data
            if return_data:
                return self.klines
        
        # If the tokens input is a list or tuple, iterate through the tokens
        elif isinstance(tokens,(list,tuple)):

            # Set a random start point in the dataset
            start = np.random.randint(0,self.dataset_length - length - 1)

            # Iterate through the tokens
            for token in tokens:

                # Check if the token exists in the dataset
                assert token in self.dataset_klines.keys(), "Token {token} does not exist in the dataset"
                klines = {}

                # Generate an eposide for all the timeseries in the token
                for key in self.dataset_klines[token]:

                    # Add the generated episode to the episode klines attribute
                    klines[key] = self.dataset_klines[token][key][start:start+length]

                    # Set the maximum time to the length of the episode
                    self.max_time = klines[key].shape[0]
                # Add the generated episode to the episode klines attribute
                self.klines[token] = klines

            # If return_data is True, return the generated episode data
            if return_data:
                return self.klines
            
        # If the tokens input is not a string, list, or tuple, raise an error
        else:
            raise ValueError("input {tokens} is neither a string, tuple or array")
            

    def get_historical_prices(self,symbols,amount):
        """
        Returns the historical prices for the specified tokens and amount of data points.
        Parameters:
        symbols (str, list, tuple): The tokens to get the historical prices for.
        amount (int): The number of data points to get.
        Returns:
        dict: A dictionary containing the historical prices for the specified tokens.
        """

        # Ensure that the requested window is not larger than the dataset
        assert(amount < self.max_time), "Can't chosse a window that is larger than the dataset"
        
        # If the historical klines attribute does not exist, create it
        if not hasattr(self,'historical_klines'):
            self.historical_klines = {}
            
        # Assert that the amount of data points requested is less than the current time
        assert amount < self.time, f"You requested {amount} datapoints, at time {self.time}. Ensure your historical request is smaller than the current time."
            
        # If the symbols input is a string, i.e. a single token
        if isinstance(symbols,str):
            # Check if the token exists in the episode
            assert symbols in self.klines.keys(), f"No data with key {symbols} has been found"
            tmp_dict = {}

            # Iterate through the timeseries in the token
            for key, values in self.klines[symbols].items():

                # Add the historical prices to the temporary dictionary
                tmp_dict[key] = values[1+self.time-amount:self.time+1]
                
            # Add the historical prices to the historical klines attribute
            self.historical_klines[symbols] = tmp_dict
            
            # Return the historical klines
            return self.historical_klines
            
        # If the symbols input is a list or tuple, i.e. multiple tokens, iterate through the tokens
        elif isinstance(symbols,(list,tuple)):
        
            # Iterate through the tokens
            for symbol in symbols:
                # Check if the token exists in the episode
                assert symbol in self.klines.keys(), f"No data with key {symbol} has been found"
                tmp_dict = {}

                # Iterate through the timeseries in the token
                for key, values in self.klines[symbol].items():

                    # Add the historical prices to the temporary dictionary
                    tmp_dict[key] = values[self.time-amount:self.time]
                    
                # Add the historical prices of the specific token to the historical klines attribute
                self.historical_klines[symbol] = tmp_dict
            
            return self.historical_klines
        
        else:
            # If the symbols input is not a string, list, or tuple, raise an error
            raise ValueError("The passed input {symbols} is neither a string, tuple or list")
        
    def calc_hedge_ratio(self,timeseries1,timeseries2):
        """
        Calculates the hedge ratio between two timeseries.
        Parameters:
        timeseries1 (array): The first timeseries.
        timeseries2 (array): The second timeseries.
        Returns:
        float: The hedge ratio between the two timeseries.
        """
        return np.mean(timeseries1) / np.mean(timeseries2)
    
    def get_z_scores(self,token1,token2,length):
        """
        Calculates the z-scores for the spread between two tokens.
        Parameters:
        token1 (str): The first token.
        token2 (str): The second token.
        length (int): The length of the historical data to use.
        return_data (bool, optional): If True, returns the z-scores data. Default is False.
        excluded_keys (list, optional): The keys to exclude from the calculation. Default is ['log_return_high','log_return_low','log_return_open','log_return_close','volume','time'].
        Returns:
        dict: If return_data is True, returns a dictionary containing the z-scores data.
        """

        excluded_keys = ['volume','time']

        # Ensure that the tokens are strings
        assert isinstance(token1,str) and isinstance(token2,str), "Tokens must be strings"

        # Ensure that the tokens are in the loaded episode
        assert token1 in self.klines and token2 in self.klines, "The token was not found in the loaded episode. Ensure an episode with the required keys are loaded"
        
        # Get the historical prices for the two tokens
        klines = self.get_historical_prices([token1,token2], length)
        
        z_scores_dict = {}

        # Iterate through the timeseries in the first token
        for key in klines[token1].keys():

            # If the key is not in the excluded keys
            if key not in excluded_keys:

                # Get the timeseries for the two tokens
                timeseries1 = klines[token1][key]
                try:
                    # If the key is in the second token, calculate the z-scores
                    timeseries2 = klines[token2][key]
        
                    # Calculate the hedge ratio
                    hedge_ratio = self.calc_hedge_ratio(timeseries1,timeseries2)
                    
                    # Calculate the spread and z-scores
                    spread = timeseries1 - hedge_ratio * timeseries2
                    z_scores = (spread - np.mean(spread)) / np.std(spread)
                    z_scores_dict[key] = z_scores
                
                # If the key is not in the second token, raise a KeyError
                except KeyError as e:
                    print(f"Timeseries {key}, was found in the {token1} dataset but not the {token2}")
                    z_scores_dict[key] = None

        return z_scores_dict

                
    def calc_coint_values(self,token1,token2,length,key='open'):
        """
        Calculates the p-value for the cointegration test between two tokens for a specific timeseries key.
        Parameters:
        token1 (str): The first token.
        token2 (str): The second token.
        length (int): The length of the historical data to use.
        key (str, optional): The key to use for the calculation. Default is 'open'.
        Returns:
        float: The p-value for the cointegration test between the two tokens.
        """
        # Ensure that the tokens are strings
        assert isinstance(token1,str) and isinstance(token2,str), "Tokens must be specified as string"

        # Ensure that the tokens are in the loaded dataset
        assert token1 in self.dataset_klines and token1 in self.dataset_klines, "Specified tokens were not fount in the loaded dataset"
        
        # Get the historical prices for the two tokens
        klines = self.get_historical_prices([token1,token2],length)

        # Get the timeseries for the two tokens
        asset1 = klines[token1][key]
        asset2 = klines[token2][key]
        
        #preform adf on asset prices, this checks to see if the time series is stationary
        adf_result_asset1 = adfuller(asset1)
        adf_result_asset2 = adfuller(asset2)

        # Calculate the cointegration test p-value and score
        score, p_value, _ = coint(asset1, asset2)
        

        return p_value, adf_result_asset1[1], adf_result_asset2[1]
        
    
    def get_current_portfolio_value(self):
        """
        Returns the current value of the portfolio.
        Returns:
        float: The current value of the portfolio.
        """
        # Initialize the asset values
        asset_values = 0
    
        # Iterate through the positions
        for key in self.positions.keys():

            # Get the token amount and price
            token_amount = self.positions[key]
            token_price = self.get_current_price(key)

            # Check to see if token is held or shorted
            if token_amount > 0:

                # Calculate the gain when selling the token at the current price
                gain = token_amount * token_price

                # Calculate the discounted gain (this includes the transaction percentage fee that will be applied when selling)
                discounted_gain = gain - gain*self.transaction_percentage
                asset_values += discounted_gain
            else:

                # Calculate the loss when buying the covered token at the current price
                token_amount = -1*token_amount
                loss = token_amount * token_price

                # Calculate the additional loss (this includes the transaction percentage fee that will be applied when buying)
                additional_loss = loss + loss*self.transaction_percentage

                # Subtract the additional loss from the asset values
                asset_values -= additional_loss
               
        # Return the total value of the portfolio
        return self.money + asset_values
        
    
    def get_current_price(self,symbol,key = 'open'):
        """
        Returns the current price of the specified token.
        Parameters:
        symbol (str): The token to get the current price for.
        key (str, optional): The key to use for the calculation. Default is 'open'.
        Returns:
        float: The current price of the token.
        """

        # Ensure that the token is a string
        assert isinstance(symbol,str), "Token must be a string"

        # Ensure that the token is in the loaded episode
        if not symbol in self.klines.keys():
            # If the token is not in the loaded episode, print a message and return None
            print("No data exists for this token")
            return None
        else:
            # Return the current price of the token
            return self.klines[symbol][key][self.time]
        
        
    def buy_token(self,token,amount,return_data=False,verbose=False):
        """
        Buys a specified amount of a token at the current price.
        Parameters:
        token (str): The token to buy.
        amount (float): The amount of the token to buy specified in the same unit as self.money (e.g USD).
        return_data (bool, optional): If True, returns the amount of token bought. Default is False.
        """

        # Ensure that the token is a string
        assert isinstance(token,str), "Token must be a string"

        # Ensure that the token is in the loaded episode
        if not token in self.klines.keys():
            # If the token is not in the loaded episode, print a message and return None
            print('No token exists with this name')
        else:
            
            # Calculate the discounted amount of the token to buy (this includes the transaction fee)
            discounted_amount = amount*(1-self.transaction_percentage)

            # Get the current price of the token
            price = self.get_current_price(token)

            # Calculate the amount of token to buy
            token_amount =  discounted_amount /price

            # Subtract the discounted amount from the money
            self.money -= amount
            
            # Add the token amount to the positions dictionary
            if token in self.positions.keys():
                self.positions[token] += token_amount
            else:
                self.positions[token] = token_amount
             
            # Print a message if verbose is True
            if verbose:
                print(f"Bought {token_amount} of {token} at ${price} (cost ${amount})")
            
            # If return_data is True, return the amount of token bought
            if return_data:
                return token_amount
                
    def short_token(self,token,amount,return_data = False,verbose=False):
        """
        Shorts a specified amount of a token at the current price.
        Parameters:
        token (str): The token to short.
        amount (float): The amount of the token to short specified in the same unit as self.money (e.g USD).
        return_data (bool, optional): If True, returns the amount of token shorted. Default is False.
        verbose (bool, optional): If True, prints a message when shorting the token. Default is False.
        returns:
        float: The amount of token shorted.
        """
        # Ensure that the token is a string
        assert isinstance(token,str), "Token must be a string"

        # Ensure that the token is in the loaded episode
        if not token in self.klines.keys():
            print('No token exists with this name')

        else:
            
            # Calculate the discounted amount of the token to short (this includes the transaction fee)
            discounted_amount = amount*(1-self.transaction_percentage)

            # Get the current price of the token
            price = self.get_current_price(token)

            # Calculate the amount of token to short
            token_amount = discounted_amount /price

            # Add the discounted amount to the money
            self.money += discounted_amount
            
            # Verbose the shorting of the token if verbose is True
            if verbose:
                print(f"Shorted {token_amount} of {token} at ${price} (position value ${discounted_amount})")
            
            # Add the token amount to the positions dictionary
            if token in self.positions.keys():
                self.positions[token] -= token_amount
            else:
                self.positions[token] = -token_amount
            
            # If return_data is True, return the amount of token shorted
            if return_data:
                return token_amount
                
    def step_time(self,time_step):
        """
        Steps the time forward by the specified amount.
        Parameters:
        time_step (int): The amount of time to step forward.
        Returns:
        bool: False if the time was stepped forward (i.e the episode is not done), True if the maximum time was exceeded.
        """

        # Check if the time plus the time step is greater than the maximum time
        if self.time + time_step >= self.max_time:
            # If the time plus the time step is greater than the maximum time, set the time to the maximum time minus 1 and return True
            self.time = self.max_time-1
            return True
        else:
            # If the time plus the time step is less than the maximum time, step the time forward by the time step and return False
            self.time += time_step
            return False
        
    
    def close_all_positions(self,verbose = False):
        """
        Closes all open positions and returns the money to the portfolio.
        Parameters:
        verbose (bool, optional): If True, prints a message when closing the positions. Default is False.
        returns:
        None
        """
        # Iterate through the positions
        for key in self.positions.keys():

            # Get the token amount and price
            token_amount = self.positions[key]
            token_price = self.get_current_price(key)

            # Check to see if the token is held or shorted
            if token_amount > 0:

                # Calculate the gain when selling the token at the current price
                gain = token_amount * token_price

                # Calculate the discounted gain (this includes the transaction fee)
                discounted_gain = gain - gain*self.transaction_percentage

                # Add the discounted gain to the money
                self.money += discounted_gain

                # Print a message if verbose is True
                if verbose:
                    print(f"Sold {token_amount} {key} for ${discounted_gain}")
            else:

                # Set the token amount to a positive value
                token_amount = -1*token_amount

                # Calculate the loss when buying the covered token at the current price
                loss = token_amount * token_price

                # Calculate the additional loss (this includes the transaction fee)
                additional_loss = loss + loss*self.transaction_percentage

                # Subtract the additional loss from the money
                self.money -= additional_loss

                # Print a message if verbose is True
                if verbose:
                    print(f"Bought {token_amount} {key} for ${additional_loss}")

        # Set the positions dictionary to an empty dictionary
        self.positions = {}
        
    
    
            
        
        
