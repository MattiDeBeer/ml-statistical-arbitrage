# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 18:49:04 2024

@author: matti
"""
import matplotlib.pyplot as plt
import numpy as np
from api_wrappers.binance_api_simulation_wrapper import binance_api
from trading_models.simple_stat_arb import simple_stat_arb_trader
from statsmodels.tsa.stattools import adfuller

#%%
#defines the model tokens
token1 = 'BTCUSDT'
token2 = 'SOLUSDT'

#creatates binance api and populates with training data
wrapper = binance_api() 
wrapper.get_minute_prices_dataset((token1,token2),10000,return_data=True)
wrapper.align_time_series('all')

#%%

def cointegration_test(price1, price2, significance_level=0.05, det_order=0, k_ar_diff=1):
    """
    Tests whether two assets are cointegrated using the Johansen Test and checks stationarity of the spread.

    Parameters:
    - asset1, asset2: numpy arrays of price series for the two assets
    - significance_level: float, critical value threshold for cointegration (default: 0.05)
    - det_order: int, deterministic trend order (default: 0; no trend in data)
    - k_ar_diff: int, lag order for Johansen test (default: 1)
    """
    
    hedge_ratio = np.mean(price1)/np.mean(price2)
    spread = price1 - hedge_ratio * price2

    # Test the spread for stationarity using the Augmented Dickey-Fuller (ADF) test
    adf_stat, adf_p_value, *_ = adfuller(spread)

    # Check if the spread is stationary
    is_cointegrated = adf_p_value < significance_level

    return is_cointegrated, hedge_ratio, adf_p_value
    
def calc_current_ab(wrapper,token1,token2,window_length):
    """
    Parameters
    ----------
    wrapper : api wrapper object
        The wrapper model of the curent simulation
    token1 : str
        A token symbol
    token2 :str
        A token symbol
    window_length : int
        The historical contxt length, used to calculate the hedge ratio

    Returns
    -------
    float
        The current normalised price arbritrage value

    """
    #define a normalise function
    normalise = lambda x: (x-np.mean(x))/np.std(x)
    
    #get the previouse 'window_length' prices
    price_data = wrapper.get_prices((token1,token2), window_length,return_data=True)
    
    #extract open price data
    btc = price_data[token1]['open']
    xrp = price_data[token1]['open']
    
    #calculate the hedge ration
    hedge_ratio = np.mean(btc)/np.mean(xrp)
    
    #generate arbritrage spread
    ab_data = wrapper.generate_arbritrage_pair(token1+'-'+token2, hedge_ratio,return_data=True)['open']
    
    #normalise the spread
    ab_data = normalise(ab_data)
    
    #return the current value
    return ab_data[-1]

#%%
#define the context window length, this is the historical window over which the hegde ratio is calculated
window_length = 1000

#set the starting funds
starting_funds = 100

#set the risk (proportion of total funds committed by trade)
risk = 0.2

#set transaction fees
wrapper.transaction_percentage = 0.0001


#define the position entry and exit boudns or the arb value (these are symmetric)
enter_bound = 2.6
exit_bound = 0

#set the models initial conditions
wrapper.money=starting_funds
wrapper.time=0

trader = simple_stat_arb_trader(wrapper, [enter_bound,exit_bound,risk,token1, token2, window_length])
#set flag that allows trading
print('trading')
trader.start_trading(2500)
trader.start_trading(2500)
trader.start_trading(2500)
trader.start_trading(2500)
    
#close all positions if simulation has exited
wrapper.close_all_positons()

#calculate percentage gain
percentage_gain =round( 100* (wrapper.money - starting_funds)/starting_funds, 4)

#print the gain made
print(f"{trader.trades_number} trades performed, made a {percentage_gain}% gain")

trade_logger = trader.trades_logger


    
    