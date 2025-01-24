from binance import Client, ThreadedWebsocketManager, ThreadedDepthCacheManager
import numpy as np

"""
EXAMPLE USAGE

#ensure a file called keys.txt is in the local directory and contains binance keys
#creates a wrapper object
wrapper = api_wrapper() 


#retrieves data from wrapper object, get the most recent realtieme data
minute_price_data = wrapper.get_minute_prices(('BTCUSDT','SOLUSDT'), 100,return_data=True)
This will save the most recent 100 second and minute prices to the model attribute wrapper.klines
and return them

second_price_data = wrapper.get_second_prices(('BTCUSDT','SOLUSDT'), 100,return_data=True)
Does the same for the second data

This returns the price data for a certian winow for a specific time.
Running this will return the first 100 prices in the dataset
If time is 0, it will set time to 100 and return the 100 last prices
This function follows the internal wrapper time, so if time is set to 1000,
it will return the prices from indec 900 - 1000

#generates an arbritrage pair for a specified combination and specified hedge_ratio
wrapper.generate_arbritrage_pair(token1+'-'+token2, hedge_ratio,return_data=True)

This will return the arbitrage of the recent prices as specified iby get_x_prices()
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
        with open(key_filename) as key_file:
            key = key_file.readline().strip('\n')
            secret = key_file.readline().strip('\n')
        
        self.__api_key = key
        self.__api_secret = secret
        self.client = Client(self.__api_key, self.__api_secret)
        self.klines = None
        print("Binance API initialized")
        
    def get_minute_prices(self,symbols,amount,return_data = False):
        
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
            
            
        self.klines = klines
        if return_data:
            return self.klines
    
    
    def get_second_prices(self,symbols,amount,return_data = False):
        
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
            
            
        self.klines = klines
        if return_data:
            return self.klines
        
        
        
    def align_time_series(self,pair,return_data=False):
        """
        Aligns two dictionaries containing time series data by their common time entries.
    
        Parameters:
            pair: string
            formetted as TICKER1-TICKER2.e.g BTCUSDT-XRPUSDT
    
        Returns:
            dict, dict: Two dictionaries aligned to the same time indices.
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
            dict1 = self.klines[ticker1]
            dict2 = self.klines[ticker2]
        
            # Extract the time entries
            time1 = np.array(dict1['time'])
            time2 = np.array(dict2['time'])
        
            # Find the common time values
            common_times = np.intersect1d(time1, time2)
        
            # Helper function to filter a dictionary by the common time values
            
        
            # Filter both dictionaries
            aligned_dict1 = filter_dict(dict1, common_times)
            aligned_dict2 = filter_dict(dict2, common_times)
            
            self.klines[ticker1] = aligned_dict1
            self.klines[ticker2] = aligned_dict2
        
            if return_data:
                return aligned_dict1, aligned_dict2
        
        else:
            # Extract the 'time' arrays from all dictionaries
            all_times = [set(np.array(data['time'])) for data in self.klines.values()]
        
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
                ticker: filter_dict(data, common_times) for ticker, data in self.klines.items()
            }
        
            self.klines = aligned_klines
        
            if return_data:
                return aligned_klines
            
            
            
    def generate_arbritrage_pair(self,pair,alpha,lag = 0,return_data=False):
        """
        Parameters
        ----------
        pair : str
            pair you whish to arbritrage
        alpha : float
            The hedge ratio you whish to use to geenrate an arbritrage pair
        lag : int, optional
            The amount of timesteps you whish token 2 to lag token 1. The default is 0.
        return_data : bool, optional
            A flag that returns the arbritrage timeseries. The default is False.

        Returns
        -------
        tmp_dict : dict
            The arbritrage timeseries dictionary

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
            
        
        return tmp_dict
        
        
        
            
        
        
