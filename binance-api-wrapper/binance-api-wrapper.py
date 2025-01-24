from binance import Client, ThreadedWebsocketManager, ThreadedDepthCacheManager
import numpy as np
import matplotlib.pyplot as plt 
import copy
from nolds import hurst_rs


class api_wrapper(): 
    '''
    Defines a wrapper object for the binance api
    '''
    
    def __init__(self):
        '''
        Initilises a wrapper with the binance api client using the keys

        Returns
        -------
        None.

        '''
        self.__api_key = 'bE9ZMb1UxoUJjjpehciUNQNwB41e34oF7MJkqvlqO3zxo7a8YsAUS3pgDDahRg3N'
        self.__api_secret = 'DfDTkUz03XmYGAIYRwE3ymb24KEIyjAXv6kUct5AK5tNqQsSPsk3ANgqmKGveqtG'
        self.client = Client(self.__api_key, self.__api_secret)
        self.klines = None
        
    def get_minute_prices(self,symbols,amount):
        
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
            
            labeled_price_data['open'] = price_data[1]
            labeled_price_data['high'] = price_data[2]
            labeled_price_data['low'] = price_data[3]
            labeled_price_data['close'] = price_data[4]
            labeled_price_data['volume'] = price_data[5]
            
            klines[symbol] = labeled_price_data  
            
            
        self.klines = klines
        return self.klines
    
    
    def get_second_prices(self,symbols,amount):
        
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
            print('Fetching {0} minute prices for the last {1} seconds'.format(symbol,amount))
            labeled_price_data = {}
            price_data = np.array(self.client.get_historical_klines(symbol, Client.KLINE_INTERVAL_1SECOND, "{0} seconds ago UTC".format(amount)))
            price_data = price_data.T
            price_data = price_data.astype(float)
            
            labeled_price_data['time'] = price_data[0]
            labeled_price_data['open'] = price_data[1]
            labeled_price_data['high'] = price_data[2]
            labeled_price_data['low'] = price_data[3]
            labeled_price_data['close'] = price_data[4]
            labeled_price_data['volume'] = price_data[5]
            
            klines[symbol] = labeled_price_data  
            
            
        self.klines = klines
        return self.klines
    
    def normalise_prices(self):
        print('Generating Normalised Price Data')
        assert self.klines is not None, "Data must be fetched first, try running the .get_minute_prices() method"
        normalised_klines = copy.deepcopy(self.klines)
        for token in self.klines:
            for metric in self.klines[token]:
                if metric != 'volume' and metric != 'time':
                    mean = np.mean(self.klines[token][metric])
                    std = np.std(self.klines[token][metric])
                    normalised_klines[token][metric] = (normalised_klines[token][metric] - mean)/std
               
        self.normalised_klines = normalised_klines            
        return self.normalised_klines
    
    def generate_arbritrage_pairs(self,custom_pair = None, lag = 0):
        assert self.normalised_klines is not None, "There is no normalised price data available, try running the .normalise_prices() method first"
        assert type(lag) is int, "Lag variable must be an integer"
        self.arbritrage_pairs = {}
        print('Generating Abrritrage Pairs')
        if custom_pair is None:
            for i, token1 in enumerate(self.normalised_klines):
                for j, token2 in enumerate(self.normalised_klines):
                    if i < j:
                        try:
                            tempdict = {}
                            n1 = max(np.min(self.normalised_klines[token1]['time']),np.min(self.normalised_klines[token2]['time']))
                            n2 = min(np.max(self.normalised_klines[token1]['time']),np.max(self.normalised_klines[token2]['time']))
                            
                            token1start = np.where(self.normalised_klines[token1]['time'] == n1)[0][0]
                            token2start = np.where(self.normalised_klines[token2]['time'] == n1)[0][0]
                            token1stop = np.where(self.normalised_klines[token1]['time'] == n2)[0][0]
                            token2stop = np.where(self.normalised_klines[token2]['time'] == n2)[0][0]
                        
                            for key in ['open','high','low','close']:
                                tempdict[key] = self.normalised_klines[token1][key][token1start:int(token1stop-lag)] - self.normalised_klines[token2][key][int(token2start+lag):token2stop]
                            self.arbritrage_pairs[token1 + '-' + token2] = tempdict
                        except:
                            print("An error occured when calculating the arbritrage spread for {0} against {1}".format(token1,token2))
                            print("This error likley occured due to there not being a temporal overlap between the 2 tokens timeseries data")
            
            return self.arbritrage_pairs
        
        else:
            assert isinstance(custom_pair, tuple) and all(isinstance(item, str) for item in custom_pair) and len(custom_pair) == 2, "custom pair must be a tuple containing only 2 strings"
            token1 = custom_pair[0]
            token2 = custom_pair[1]
            
            try:
                tempdict = {}
                n1 = max(np.min(self.normalised_klines[token1]['time']),np.min(self.normalised_klines[token2]['time']))
                n2 = min(np.max(self.normalised_klines[token1]['time']),np.max(self.normalised_klines[token2]['time']))
                
                token1start = np.where(self.normalised_klines[token1]['time'] == n1)[0][0]
                token2start = np.where(self.normalised_klines[token2]['time'] == n1)[0][0]
                token1stop = np.where(self.normalised_klines[token1]['time'] == n2)[0][0]
                token2stop = np.where(self.normalised_klines[token2]['time'] == n2)[0][0]
                
             
                for key in ['open','high','low','close']:
                    tempdict[key] = self.normalised_klines[token1][key][token1start:int(token1stop-lag)] - self.normalised_klines[token2][key][int(token2start+lag):token2stop]
               
                self.arbritrage_pairs[token1 + '-' + token2] = tempdict
                return self.arbritrage_pairs
            
            except:
                print("An error occured when calculating the arbritrage spread for {0} against {1}".format(token1,token2))
                print("This error likley occured due to there not being a temporal overlap between the 2 tokens timeseries data")
                print("It may be the case that this is caused by the lag variable is too large, try using a smaller lag and inspecting the data time values to check for an overlap")
                print("This error was raised when attempting to calculate for a custom pair, so ensure the argument is a valid pair spelt correctly")
            
            
    def calulate_hurst_exp(self, custom_pair=None):
        """
        Parameters
        ----------
        custom_pair : string, optional
            DESCRIPTION. The default is None. returns hurts exponential of specified arbritrage pair.

        Returns
        -------
        Dictionary
            DESCRIPTION. Returns dictionary of pairs, each containing a dictionary of their metrics. The values of these are the hurst exponential for the given timeseries
            e.g .calculate_hurst_exp_window_average()['BTCUSDT-XRPUSDT']['open'] will be the Hurst exponential of 'BTCUSDT-XRPUSDT'.

        """
        assert self.arbritrage_pairs is not None, "Please generate arbritrage pairs first"
        
        if custom_pair is None:
            print('Generating Hurst exponential for all arbritrage pairs')
            self.hurst_exp = {}
            for pair in self.arbritrage_pairs:
                tmpdict = {}
                for key in self.arbritrage_pairs[pair]:
                    tmpdict[key] = hurst_rs(self.arbritrage_pairs[pair][key])
                self.hurst_exp[pair] = tmpdict
            return self.hurst_exp
        else:
            assert type(custom_pair) is str, "custom pair must be a tuple containing only 2 strings"
            print('Generating Hurst exponential for {0}'.format(custom_pair))
            self.hurst_exp = {}
            try:
                tmpdict = {}
                for key in self.arbritrage_pairs[custom_pair]:
                    tmpdict[key] = hurst_rs(self.arbritrage_pairs[custom_pair][key])
                self.hurst_exp[custom_pair] = tmpdict
                
                return self.hurst_exp
            except:
                print('Failed to generate custom hurst exopnential pair, please ensure custom pair exists in arbritrage data')
            
    def calculate_hurst_exp_window_average(self, custom_pair=None,window_size=100):
        """
        Parameters
        ----------
        custom_pair : string, optional
            DESCRIPTION. The default is None. returns hurts exponential of specified arbritrage pair.
        window_size : integer, optional
            DESCRIPTION. The default is 10. This specifies the size of the window to be avereged over. Must be less than the timeseries length

        Returns
        -------
        Dictionary
            DESCRIPTION. Returns dictionary of pairs, each containing a dictionary of their matrics. These contain the mean, standard deviation and standard error of the hurst exponential
            e.g .calculate_hurst_exp_window_average()['BTCUSDT-XRPUSDT']['open'] will be a numpy array of [mean, standard deviaton, standard error] of the 'BTCUSDT-XRPUSDT' open price husrt exponential over all windows

        """
        assert self.arbritrage_pairs is not None, "Please generate arbritrage pairs first"
        
        if custom_pair is None:
            print('Generating window averaged Hurst exponential for all arbritrage pairs')
            self.hurst_exp = {}
            for pair in self.arbritrage_pairs:
                tmpdict = {}
                for key in self.arbritrage_pairs[pair]:
                    assert window_size < len(self.arbritrage_pairs[pair][key]), "Window size must be smaller than timeseries length"
                    hurst_array = []
                    for i in range(0,len(self.arbritrage_pairs[pair][key]) - window_size):
                        hurst_array.append(hurst_rs(self.arbritrage_pairs[pair][key][i:i+window_size]))
                    
                    tmpdict[key] = [np.mean(hurst_array),np.std(hurst_array),np.std(hurst_array)/np.sqrt(i)]
                
                
                self.hurst_exp[pair] = tmpdict
                
                
            return self.hurst_exp
        else:
            assert type(custom_pair) is str, "custom pair must be a valid string"
            self.hurst_exp = {}
            print('Generating window averaged Hurst exponential for {0}'.format(custom_pair))
            try:
                tmpdict = {}
                for key in self.arbritrage_pairs[custom_pair]:
                    hurst_array = []
                    for i in range(0,len(self.arbritrage_pairs[custom_pair][key]) - window_size - 1):
                        hurst_array.append(hurst_rs(self.arbritrage_pairs[custom_pair][key][i:i+window_size]))
                    
                    tmpdict[key] = [np.mean(hurst_array),np.std(hurst_array),np.std(hurst_array)/np.sqrt(i)]
                self.hurst_exp[custom_pair] = tmpdict
                
                return self.hurst_exp
            except:
                print('Failed to generate custom hurst exopnential pair, please ensure custom pair exists in arbritrage data')
            
        
            
"""
EXAMPLE USAGE

#creates a wrapper object
wrapper = api_wrapper() 

#gets and stores specified price data in wrapper.klines
wrapper.get_second_prices(('BTCUSDT','XRPUSDT','BNBUSDT'),1000)

#normalises price data to have a mean of 0 and std of 1. returns and also stores in wrapper.normalised_klines
wrapper.normalise_prices()

#retrieves data from wrapper object
price = wrapper.klines
normalised_prices = wrapper.normalised_klines

#generates an arbritrage pair for all token combination
abpairs = wrapper.generate_arbritrage_pairs()

#calculates the hurst exponential for all token arbritrage pairs
hexp = wrapper.calulate_hurst_exp()

#calculates a window averaged hurst exponential for all token arbritrage pairs, returns the mean, std an standard error
hexp_window = wrapper.calculate_hurst_exp_window_average(window_size = 100)

"""