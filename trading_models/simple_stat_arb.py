#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 17:22:08 2025

@author: matti
"""
import numpy as np

class simple_stat_arb_trader:
    
    def __init__(self, wrapper, parameters):
        """
        Parameters
        ----------
        wrapper : api wrapper object
            The wrapper being used to provide data and initialise trades
        parameters : array
            [enter_bound, exit_bound, risk, token1, token2, window_length]
            
            enter_bound
                The upper and lower arbritrage bound that will initiate a trade
            exit_bound
                The lower bound that will exit a position
            risk
                The proportion of capital committed to each position
            token1
                The symbol of the first token to trade
            token2
                The symbol of the second token to trade
            window_length
                The number of historical datapoints used to calculate the hedge ratio

        Returns
        -------
        None.

        """
        
        self.enter_bound = parameters[0]
        self.exit_bound = parameters[1]
        self.risk = parameters[2]
        self.token1 = parameters[3]
        self.token2 = parameters[4]
        self.window_length = parameters[5]
        self.trades_logger = []
        self.wrapper = wrapper
        self.trades_number = 0
        self.funds_tracker= [[],[]]
        
        
    def calculate_hedge_ratio(self,price1, price2):
        
        hedge_ratio = np.mean(price1)/np.mean(price2)
        
        return hedge_ratio
    
    def calc_current_ab(self,wrapper,token1,token2,window_length):
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
        asset1 = price_data[token1]['open']
        asset2 = price_data[token1]['open']
        
        #calculate the hedge ration
        hedge_ratio = self.calculate_hedge_ratio(asset1,asset2)
        
        #generate arbritrage spread
        ab_data = wrapper.generate_arbritrage_pair(token1+'-'+token2, hedge_ratio,return_data=True)['open']
        
        #normalise the spread
        ab_data = normalise(ab_data)
        
        #return the current value
        return ab_data[-1]
    
    def execute_short(self, current_ab):
        
        #enter long position on arb
        current_money = self.wrapper.money
        print(f"{self.wrapper.time}: Entered short at {self.wrapper.time} with arb of {current_ab}")
        self.trades_number += 1
        arb_tracker = []
        
        #initiate trades
        trade_value = self.risk * self.wrapper.money / 2
        amount_short = self.wrapper.short_token(self.token1, trade_value, return_data=True)
        amount_buy = self.wrapper.buy_token(self.token2, trade_value, return_data = True)
        print('\n')
        
        token_1_value = [amount_short * self.wrapper.get_current_price(self.token1)]
        token_2_value = [amount_buy * self.wrapper.get_current_price(self.token2)]
        discounted_position_value = [token_2_value[0] - token_1_value[0]]
        
        self.trades_logger.append({'type' : 'enter short', 'time' : self.wrapper.time , 'cost' : trade_value*2, 'current arb' : current_ab, \
                                   'token 1 price' : self.wrapper.get_current_price(self.token1), 'amount shotred': amount_short, \
                                    'token 2 price' : self.wrapper.get_current_price(self.token2), 'amount bought': amount_buy })
        
        #wait until arb has decreased to exit_bound level or the simulation time has expired
        #This steps the model by 1 datapoint forward in time each time it checks
        while current_ab  > self.exit_bound and self.wrapper.step_time(1):
            current_ab = self.calc_current_ab(self.wrapper,self.token1,self.token2,self.window_length)
            arb_tracker.append(current_ab)
            token_1_value.append(amount_short * self.wrapper.get_current_price(self.token1))
            token_2_value.append(amount_buy * self.wrapper.get_current_price(self.token2))
            discounted_position_value.append(token_2_value[-1] - token_1_value[-1])
            
            
        #close positions when exit condition is met ot dataset has expired
        print(f"{self.wrapper.time}: Exited long at {self.wrapper.time} with arb of {current_ab}")
        self.wrapper.close_all_positons()
        print('\n')
        print(f"Made: ${self.wrapper.money - current_money}")
        print('-'* 50,end = '\n')
        self.trades_logger.append({'type' : 'exit short', 'time' : self.wrapper.time, 'gain' : float(self.wrapper.money - current_money), 'arb_data' : arb_tracker, \
                                   'token 1 value' : token_1_value, 'token 2 value' : token_2_value, \
                                    'discounted_position_value' : discounted_position_value})
        self.trades_number += 1
        self.funds_tracker[0].append(self.wrapper.time)
        self.funds_tracker[1].append(self.wrapper.money)
            
    def execute_long(self,current_ab):
        """
        Parameters
        ----------
        current_ab : float
            The current arbritrage value

        Returns
        -------
        None.

        Description
        -----------
        Executes a long position and runs the simulation until it closes
        or until the dataset expires.
        This will hold a trade longer than the allowed timesteps if it is still active
        It records the trade statistics in self.trade and verboses trade ststistics
        """
        
        #enters long arb position
        current_money = self.wrapper.money
        arb_tracker = []
        print(f"{self.wrapper.time}: Entered long at {self.wrapper.time} with arb of {current_ab}\n")
        self.trades_number += 1
        
        trade_value = self.risk * self.wrapper.money / 2
        amount_buy = self.wrapper.buy_token(self.token1, trade_value, return_data=True)
        amount_short = self.wrapper.short_token(self.token2, trade_value, return_data = True)
        
        print('\n')
        
        token_1_value = [amount_buy * self.wrapper.get_current_price(self.token1)]
        token_2_value = [amount_short * self.wrapper.get_current_price(self.token2)]
        discounted_position_value = [token_1_value[0] - token_2_value[0]]
        
        self.trades_logger.append({'type' : 'enter long', 'time' : self.wrapper.time , 'cost' : trade_value*2, 'current arb' : current_ab, \
                                   'token 1 price' : self.wrapper.get_current_price(self.token1), 'amount bought': amount_buy, \
                                    'token 2 price' : self.wrapper.get_current_price(self.token2), 'amount shorted': amount_short })
        
        #wait until arb has increased to exit_bound level or the simulation time has expired
        #This steps the model by 1 datapoint forward in time each time it checks
        while current_ab < -1* self.exit_bound  and self.wrapper.step_time(1):
            current_ab  = self.calc_current_ab(self.wrapper,self.token1,self.token2,self.window_length)
            arb_tracker.append(current_ab)
            token_1_value.append(amount_buy * self.wrapper.get_current_price(self.token1))
            token_2_value.append(amount_short * self.wrapper.get_current_price(self.token2))
            discounted_position_value.append(token_1_value[-1] - token_2_value[-1])
            
        #close positions when exit condition is met 
        print(f"{self.wrapper.time}: Exited short at {self.wrapper.time} with arb of {current_ab}")
        self.wrapper.close_all_positons()
        print('\n')
        print(f"Made: ${self.wrapper.money - current_money}")
        print('-'* 50,end = '\n')
        self.trades_logger.append({'type' : 'exit long', 'time' : self.wrapper.time, 'gain' : float(self.wrapper.money - current_money), 'arb_data' : arb_tracker, \
                                  'token 1 value' : token_1_value, 'token 2 value' : token_2_value, \
                                   'discounted_position_value' : discounted_position_value})
        self.trades_number += 1
        self.funds_tracker[0].append(self.wrapper.time)
        self.funds_tracker[1].append(self.wrapper.money)
        
    def start_trading(self,timesteps):
        """
        Parameters
        ----------
        timesteps : int
            The amount of timesteps you want to allow the algorithm to trade for

        Returns
        -------
        None.

        """
        
        
        start_time = self.wrapper.time
        trading_flag = True
        
        while self.wrapper.time - start_time < timesteps and trading_flag:
            
            #calculate the arb at the current time
            current_ab = self.calc_current_ab(self.wrapper,self.token1,self.token2,self.window_length)
            
            #checks to see if arb is above bound
            if current_ab > self.enter_bound:
                self.execute_short(current_ab)
            
            #checks to see if arb is below bound
            if current_ab < -1*self.enter_bound:
                self.execute_long(current_ab)
        
            trading_flag = self.wrapper.step_time(1)
        
        return trading_flag
