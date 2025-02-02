#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 15:35:00 2025

@author: matti
"""

import matplotlib.pyplot as plt
import numpy as np
from api_wrappers.binance_api_simulation_wrapper import binance_api
from statsmodels.tsa.stattools import adfuller

#defines the model tokens
token1 = 'BTCUSDT'
token2 = 'SOLUSDT'

### creatates binance api and populates with training data
#This block does not have to be rerun to resimuate a strategy
#This section only loads the dataset
wrapper = binance_api() 
wrapper.get_minute_prices_dataset((token1,token2),10000,return_data=True)

#time aligns dataset timeseries
wrapper.align_time_series('all')
###

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


#define the context window length, this is the historical window over which the hegde ratio is calculated
window_length = 1000

#set the starting funds
starting_funds = 100

#set the risk (proportion of total funds committed by trade)
risk = 0.2

#set the value of each individual trade
trade_value = starting_funds * risk/2

#set transaction fees
wrapper.transaction_percentage = 0.0001

#create varibles that log important statistics
arbs = []
trades = 0
trades_logger = []

#define the position entry and exit boudns or the arb value (these are symmetric)
enter_bound = 2.6
exit_bound_lower = 0

#set the models initial conditions
wrapper.money=starting_funds
wrapper.time=0

#set flag that allows trading
trading_flag = True 
while trading_flag:
    
    #calculate the arb at the current time
    current_ab = calc_current_ab(wrapper,token1,token2,window_length)
    arbs.append(current_ab)
    
    #checks to see if arb is above bound
    if current_ab > enter_bound:
        
        #enter long position on arb
        current_money = wrapper.money
        arb_tracker = []
        print(f"entered short at {wrapper.time} with arb of {current_ab}")
        trades += 1
        
        #initiate trades
        wrapper.short_token(token1, trade_value)
        wrapper.buy_token(token2, trade_value)
        
        #wait until arb has decreased to exit_bound level or the simulation time has expired
        #This steps the model by 1 datapoint forward in time each time it checks
        while current_ab  > exit_bound_lower and wrapper.step_time(1):
            current_ab = calc_current_ab(wrapper,token1,token2,window_length)
            arbs.append(current_ab)
            arb_tracker.append(current_ab)
            
        #close positions when exit condition is met
        wrapper.close_all_positons()
        print(f"Exited long at {wrapper.time} with arb of {current_ab}")
        print(f"Made: ${wrapper.money - current_money}")
        print('-'* 20,end = '\n')
        trades_logger.append({'type' : 'long', 'gain' : wrapper.money - current_money, 'arb_data' : arb_tracker})
        
    #checks to see if arb is below bound
    if current_ab < -1*enter_bound:
        
        #enters long arb position
        current_money = wrapper.money
        arb_tracker = []
        print(f"entered long at {wrapper.time} with arb of {current_ab}")
        trades += 1
        wrapper.buy_token(token1, trade_value)
        wrapper.short_token(token2, trade_value)
        
        #wait until arb has increased to exit_bound level or the simulation time has expired
        #This steps the model by 1 datapoint forward in time each time it checks
        while current_ab < -1* exit_bound_lower  and wrapper.step_time(1):
            current_ab  = calc_current_ab(wrapper,token1,token2,window_length)
            arbs.append(current_ab)
            arb_tracker.append(current_ab)
            
        #close positions when exit condition is met 
        wrapper.close_all_positons()
        print(f"exited short at {wrapper.time} with arb of {current_ab}")
        print(f"Made: ${wrapper.money - current_money}")
        print('-'* 20,end = '\n')
        trades_logger.append({'type' : 'short', 'gain' : wrapper.money - current_money, 'arb_data' : arb_tracker})
        
    #If no contition is met, increment model time
    #This will set teh flag to false if the simulation time has expired
    trading_flag = wrapper.step_time(1) 
        

#close all positions if simulation has exited
wrapper.close_all_positons()

#calculate percentage gain
percentage_gain =round( 100* (wrapper.money - starting_funds)/starting_funds, 4)

#print the gain made
print(f"{trades} trades performed, made a {percentage_gain}% gain")


    
    