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
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# OPTION VALUATION
class Derivatives:

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
        
        if option_type == 'Call':
            value = S * self.norm_cdf(d1) - K * math.exp(-r * T) * self.norm_cdf(d2)
        elif option_type == 'Put':
            value = K * math.exp(-r * T) * self.norm_cdf(-d2) - S * self.norm_cdf(-d1)
        else:
            raise ValueError("Invalid option type. Choose 'call' or 'put'.")
        
        return value
    
    def norm_cdf(self, x):
        """Calculates the cumulative distribution function of the standard normal distribution."""
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ## OPTION DELTA FUNCTION
    def option_delta(self, option_type, S, K, r, T, sigma):
        d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
        
        if option_type == 'Call':
            delta = norm.cdf(d1)
        elif option_type == 'Put':
            delta = norm.cdf(d1) - 1
        else:
            raise ValueError("Invalid option type. Please choose 'Call' or 'Put'.")
        
        return delta
            
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #### FWD VALUATION
    def fwd_pricer(spot, rate, time, subtype):
        prem = premiums[premiums.Spread==subtype].Price.values[0]
        
        fwd = (spot + prem) * (1+rate)**time   # to deal with method premiums
        return fwd

d = Derivatives()
