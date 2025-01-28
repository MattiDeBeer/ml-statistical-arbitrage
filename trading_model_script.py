# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 18:49:04 2024

@author: matti
"""
import matplotlib.pyplot as plt
from api_wrappers.binance_api_simulation_wrapper import binance_api
from trading_models.simple_stat_arb import simple_stat_arb_trader

#creates binance api and populates with training data
wrapper = binance_api() 

#defines the model tokens
token1 = 'BTCUSDT'
token2 = 'ETHUSDT'

wrapper.get_minute_prices_dataset((token1,token2),10000,return_data=True)
   
wrapper.align_time_series('all')


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

#trades for 50000 minutes
trader.start_trading(10000)
        

#close all positions if simulation has exited
wrapper.close_all_positons()

#calculate percentage gain
percentage_gain =round( 100* (wrapper.money - starting_funds)/starting_funds, 4)

#print the gain made
print(f"{trader.trades_number} trades performed, made a {percentage_gain}% gain")

trade_logger = trader.trades_logger

    