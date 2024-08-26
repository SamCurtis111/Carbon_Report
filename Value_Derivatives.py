# -*- coding: utf-8 -*-
"""
PORTFOLIO OPTIMISATION

Positions

This uses black-scholes. It is more correct to use black 76 
but the entered underlying price in black 76 is the futures price which requires more effort
and calcs. Start with black scholes and eventually migrate to 76


When assembling everything put all these functions in a class

"""
import math
from math import log, exp, sqrt
from scipy.stats import norm
import numpy as np
import pandas as pd
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# OPTION VALUATION
class Derivatives:
    def __init__(self, premiums):
        self.premiums = premiums.copy()        

    def black_scholes(self, option_type, S, K, r, T, sigma):
        """
        Calculates the value of a put or call option using the Black-Scholes formula.
        
        Parameters:
            - option_type: Either 'call' or 'put'.
            - S: Underlying asset price.
            - K: Strike price of the option.
            - r: Risk-free interest rate.
            - T: Time to expiration in years.
            - sigma: Annualized volatility of the underlying asset.
            
        Returns:
            The value of the option.
        """
        d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        #d1, d2 = self.calc_moments(S, K, r, T, sigma)
        
        if option_type == 'Call':
            value = S * self.norm_cdf(d1) - K * math.exp(-r * T) * self.norm_cdf(d2)
        elif option_type == 'Put':
            value = K * math.exp(-r * T) * self.norm_cdf(-d2) - S * self.norm_cdf(-d1)
        else:
            raise ValueError("Invalid option type. Choose 'call' or 'put'.")
        
        return value
    
    def black76(self, option_type, S, K, r, T, sigma, q=0):
        """
        Calculate the option price for commodity futures using the Black-Scholes-Merton model.

        Parameters:
        - option_type (str): 'call' for call option, 'put' for put option
        - S (float): Current commodity futures price
        - K (float): Option strike price
        - T (float): Time to expiration in years
        - r (float): Risk-free interest rate
        - sigma (float): Volatility of the commodity futures price
        - q (float): Continuous yield or cost of carry (default is 0)

        Returns:
        - float: Commodity option price
        """
        if option_type not in ['Call', 'Put']:
            raise ValueError("Invalid option type. Use 'Call' or 'Put'.")

        #d1 = (math.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
        d1 = (math.log(S / K) + 0.5 * sigma**2 * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)

        if option_type == 'Call':
            #option_price = math.exp(-r * T) * (S * self.norm_cdf(d1) - K * self.norm_cdf(d2))
            option_price = math.exp(-r * T) * (S * norm.cdf(d1) - K * norm.cdf(d2))
        else:
            #option_price = math.exp(-r * T) * (K * self.norm_cdf(-d2) - S * math.exp((r - q) * T) * self.norm_cdf(-d1))
            option_price = math.exp(-r * T) * (K * norm.cdf(-d2) - S * norm.cdf(-d1))

        return option_price
    
    def norm_cdf(self, x):
        """Calculates the cumulative distribution function of the standard normal distribution."""
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ## OPTION DELTA FUNCTION
    def option_delta(self, option_type, S, K, r, T, sigma):
        d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
        #d1, d2 = self.calc_moments(S,K,r,T,sigma)
        
        if option_type == 'Call':
            delta = norm.cdf(d1)
        elif option_type == 'Put':
            delta = norm.cdf(d1) - 1
        else:
            raise ValueError("Invalid option type. Please choose 'Call' or 'Put'.")
        
        return delta
    

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #### OPTION THETA
    def option_theta(self, S, K, r, sigma, T, option_type):
        # S: Underlying stock price
        # K: Strike price
        # r: Risk-free interest rate
        # sigma: Volatility
        # T: Time to expiration (in years)
        # option_type: "call" or "put"
    
        d1 = (math.log(S / K) + (r + (sigma ** 2) / 2) * T) / (sigma * sqrt(T))
        d2 = d1 - sigma * sqrt(T)
    
        if option_type == "Call":
            theta = (-S * norm.pdf(d1) * sigma) / (2 * sqrt(T)) - r * K * exp(-r * T) * norm.cdf(d2)
        elif option_type == "Put":
            theta = (-S * norm.pdf(d1) * sigma) / (2 * sqrt(T)) + r * K * exp(-r * T) * norm.cdf(-d2)
        else:
            raise ValueError("Invalid option type. Please enter either 'Call' or 'Put'.")
    
        return theta/365
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #### OPTION VEGA
    def option_vega(self, S, K, r, sigma, T):
        # S: Underlying stock price
        # K: Strike price
        # r: Risk-free interest rate
        # sigma: Volatility
        # T: Time to expiration (in years)
    
        d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        vega = S * norm.pdf(d1) * sqrt(T)
        
        return vega
            
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #### FWD VALUATION
    def fwd_pricer(self, spot, rate, time, subtype):
        #if np.isnan(subtype):
        #    prem=0
        try:
            prem = self.premiums[self.premiums.Spread==subtype].Price.values[0]
        except IndexError:
            prem=0
        
        fwd = (spot + prem) * (1+rate)**time   # to deal with method premiums
        return fwd
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #### STANDARD DEVIATION PRICE MOVE
    # Using empirical st. dev because returns are non-normally distributed (skew & kurtosis)
    def empirical(self, mkt, stds, freq):    
        sub = prices[mkt]#.pct_change()[-obs:]
        if freq == 'daily':
            sub = sub.pct_change()[-126:] # last 6 months
        elif freq=='weekly':
            sub = sub.resample('W').ffill().pct_change()[1:]
        
        last_price = prices[mkt][-1:]
        # Calculate positive and negative one standard deviation price moves
        if stds==1:
            positive_std_move = np.percentile(sub, 84.13)
            negative_std_move = np.percentile(sub, 15.87)
        elif stds==2:
            positive_std_move = np.percentile(sub, 97.72)
            negative_std_move = np.percentile(sub, 2.28)
            
        price_increase = (1+positive_std_move)*last_price
        price_decrease = (1+negative_std_move)*last_price
        return price_increase, price_decrease

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #### HISTORICAL VOLATILITY
    def calculate_rolling_volatility(self, mkt, prices, window_sizes=[10, 30, 50, 100]):
        """
        Calculate rolling historical volatility of daily financial data over multiple window sizes.
    
        Parameters:
            data (pd.Series): A pandas Series containing the daily financial data (e.g., stock prices, returns, etc.).
            window_sizes (list): A list of integers representing the window sizes for which to calculate volatility.
    
        Returns:
            pd.DataFrame: A DataFrame containing rolling historical volatility for each window size.
        """
        results = {}
        data = prices[mkt]
    
        for window_size in window_sizes:
            # Calculate the rolling standard deviation
            rolling_std = data.rolling(window=window_size).std()
    
            # Calculate the rolling volatility (annualized)
            volatility = rolling_std * np.sqrt(252)  # Assuming 252 trading days in a year
    
            # Store the results in the dictionary
            results[f'{window_size}-day Volatility'] = volatility
    
        # Combine the results into a DataFrame
        result_df = pd.DataFrame(results)
        result_df['Date'] = prices['Date']
        result_df.columns = ['10-day_Vol','30-day_Vol','50-day_Vol','100-day_Vol','Date']
        result_df = result_df[['Date','10-day_Vol','30-day_Vol','50-day_Vol','100-day_Vol']]
    
        return result_df
    



#derivative = Derivatives(premiums)
#v = calculate_rolling_volatility('ACCU')
#pos_reporting = Position_Reporting(positions, 'NZU', dates['OneYear'])
#pos_reporting = Position_Reporting(positions, 'NZU', dates['EoM'])
#spot = pos_reporting.spot
#spot_price = pos_reporting.spot.Price[0]
#current_op_rate = 0.065
#prices = pos_reporting.prices
#
#derivative = Derivatives(premiums)
#
#current_op = pos_reporting.ops.loc[0]    # the Dec23 55P
#price = 56
#op_price = derivative.black_scholes('Put', 40, 62,0.06, 0.0646, 0.34)
#
#current_op.Subtype
#current_op.Strike
#current_op.Time
#current_op.Vol
#
#for p in prices:
#    op_price = derivative.black_scholes(current_op.Subtype, p, current_op.Strike, current_op_rate, current_op.Time, current_op.Vol)
#    theta = derivative.option_theta(p, current_op.Strike, current_op_rate, current_op.Vol, current_op.Time, current_op.Subtype)
#    vega = derivative.option_vega(p, current_op.Strike, current_op_rate, current_op.Vol, current_op.Time)/100            
#    delta = derivative.option_delta(current_op.Subtype, p, current_op.Strike, current_op_rate, current_op.Time, current_op.Vol)  
#
#
#op_value = op_price * current_op.Qty
#op_pnl = (op_value - current_op.Value) * pos_reporting.fx
#op_delta = delta * current_op.Qty
#op_theta = theta * current_op.Qty
#op_vega = vega * current_op.Qty