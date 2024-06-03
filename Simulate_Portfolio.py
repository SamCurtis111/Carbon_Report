# -*- coding: utf-8 -*-
"""
This code is designed to generate portfolios to be tested to create an optimal beta strategy.
It should generate portfolios that can then be used in conjunction with Test Portfolio Historically.py to assess returns and relativity to beta
"""
import pandas as pd
import os
os.chdir('C:\\GitHub\\Carbon_Report')


from Retrieve_Market_Info import Market_Info

mkts = ['ACCU','LGC','NZU','EUA', 'UKA', 'CCA']
market_info = Market_Info(mkts)
positions = market_info.positions

class Simulated_Portfolio:
    def __init__(self, timehorizon):
        self.horizon, self.changes, self.market_return, self.price_shift, self.market_move = market_info.price_moves(timehorizon)
        self.spot_prices = market_info.spot_price
        
    def calculate_sigma(self):  # what prices are an x sigma move over the designated time horizon
        products = []
        two_sigma = []
        onepointfive_sigma = []
        one_sigma = []
        
        for m in list(self.market_return):
            products.append(m)
            spot = self.spot_prices[m]
            two_sigma.append(round(spot * (1+ self.market_return[m].quantile(0.05, interpolation='nearest')), 2))
            onepointfive_sigma.append(round(spot * (1+ self.market_return[m].quantile(0.13, interpolation='nearest')), 2))
            one_sigma.append(round(spot * (1+ self.market_return[m].quantile(0.32, interpolation='nearest')), 2))
            
        sigma_prices = pd.DataFrame()
        sigma_prices['Mkt'] = products
        sigma_prices['TwoSigma'] = two_sigma
        sigma_prices['OnePointFive'] = onepointfive_sigma
        sigma_prices['OneSigma'] = one_sigma
        return sigma_prices
    
    
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Scratchpad
timehorizon = current_date + dt.timedelta(days=5)

sim = Simulated_Portfolio('5d')
sim.calculate_sigma()
sim.horizon

changes = sim.changes
mkt_return = sim.market_return