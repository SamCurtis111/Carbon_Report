# -*- coding: utf-8 -*-
"""
Class that gets information about market prices and fund positions
"""
import pandas as pd
import os
os.chdir('C:\\GitHub\\Carbon_Report')

import datetime as dt
current_date = dt.datetime.today().date()
weekly = current_date + dt.timedelta(days=5)
one_month = current_date + dt.timedelta(days=30)
three_months = current_date + dt.timedelta(days=3*30)



#mkts = ['ACCU','LGC','NZU','EUA', 'UKA', 'CCA','RGGI','VCM','OTHER']
mkts = ['ACCU','LGC','NZU','EUA', 'UKA', 'CCA','RGGI']



#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
## IMPORT POSITIONS, PRICES, OTHER REFERENCES (fx rates, ivols, spot prices)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Create a class that can be referenced to pull information about fund positions, market prices
# Should the function to generate mkt returns over different time horizons live here?
class Market_Info:
    def __init__(self, markets):
        self.mkts = markets.copy()
        
        self.prices = pd.read_excel('Positions.xlsx', sheet_name='Prices')
        self.prices = self.prices.drop(columns='Date_Reference')
        self.prices = self.prices.dropna()
        self.prices = self.prices.iloc[1:, :]   # this is so the price series starts on a Monday    
        
        self.ivols = pd.read_excel('Positions.xlsx', sheet_name='iVols')  # implied volatilities of UKA and EUA put options at different strikes (as per bloomberg last)
        self.references = pd.read_excel('Positions.xlsx', sheet_name='Index')  # fx rates, spot prices
        self.FUM = self.references[self.references.Spread=='FUM']['Price']
        self.FUM = self.FUM.reset_index(drop=True)[0] - 775000 #### REDEMPTION
        
        self.fx_rates = self.generate_fx_rates()
        self.positions = self.generate_positions()
        
        self.spot_price = self.generate_spot_prices()
        
    
    def generate_fx_rates(self):
        fx_dict = {}
        fx_dict['ACCU'] = 1
        fx_dict['LGC'] = 1
        fx_dict['NZU'] = self.references[self.references.Spread=='NZDAUD'].Price.reset_index(drop=True)[0]
        fx_dict['EUA'] = self.references[self.references.Spread=='EURAUD'].Price.reset_index(drop=True)[0]
        fx_dict['UKA'] = self.references[self.references.Spread=='GBPAUD'].Price.reset_index(drop=True)[0]
        fx_dict['CCA'] = self.references[self.references.Spread=='USDAUD'].Price.reset_index(drop=True)[0]
        fx_dict['VCM'] = self.references[self.references.Spread=='USDAUD'].Price.reset_index(drop=True)[0]
        fx_dict['RGGI'] = self.references[self.references.Spread=='USDAUD'].Price.reset_index(drop=True)[0]
        return fx_dict
    
    def generate_positions(self):
        positions = dict()
        for m in mkts:
            positions[m] = pd.read_excel("Positions.xlsx", sheet_name=m)
            positions[m]['Expiry'] = pd.to_datetime(positions[m].Expiry).dt.date
        return positions
    
    def generate_spot_prices(self):
        spot_price = dict()
        for m in self.mkts:
            sub = self.positions[m]
            spot_price[m] = sub[sub.Type=='Spot'].Price[0]
        return spot_price
    
    def time_horizon(self, horizon):  # Calculates market returns over different time horizons
        changes = self.prices.copy()
        changes = changes.set_index('Date')
        changes = changes[self.mkts]
        
        if horizon == 'daily':
            horizon = current_date
            changes = changes.pct_change()
            changes = changes.dropna()
        elif horizon == '5d':
            horizon = weekly
            cons = pd.DataFrame(index=pd.date_range(start=changes.index.min(), end=changes.index.max(), freq='D'))   # consecutive weekdays
            # Merge the original DataFrame with the DataFrame containing consecutive dates
            merged_changes = pd.merge(cons, changes, left_index=True, right_index=True, how='left')
            changes_5d = merged_changes.copy()
            for m in list(changes_5d):
                changes_5d[m] = changes_5d[m].rolling(window=5).apply(lambda x: (x[-1] - x[0]) / x[0], raw=True)
            changes = changes_5d.copy()
            changes = changes.dropna()
        elif horizon == '30d':
            horizon = one_month
            changes_1m = changes.copy()
            changes_1m = changes_1m.reset_index()
            changes_1m['year_month'] = changes_1m['Date'].dt.to_period('M')
            changes_1m = changes_1m.groupby(by='year_month').tail(1)
            changes_1m = changes_1m.iloc[:,:-1]
            changes_1m = changes_1m.set_index('Date')        
            changes_1m = changes_1m.pct_change()
            changes_1m = changes_1m.iloc[1:, :]
            changes = changes_1m.copy()
        elif horizon == '3m':
            horizon = three_months
            changes_3m = changes.copy().reset_index()
            changes_3m['quarter'] = changes_3m['Date'].dt.quarter
            changes_3m['year'] = changes_3m['Date'].dt.year
            changes_3m['y_q'] = [str(changes_3m['year'][i])+str(changes_3m['quarter'][i]) for i in range(0, len(changes_3m))]
            changes_3m = changes_3m.groupby(by='y_q').tail(1)
            changes_3m = changes_3m.drop(columns=['quarter','year','y_q'])
            changes_3m = changes_3m.set_index('Date')
            changes_3m = changes_3m.iloc[:-1, :]
            changes_3m = changes_3m.pct_change()
            changes_3m = changes_3m.iloc[1:, :]
            changes = changes_3m.copy()        
        else:
            print('use either daily or 5d or 30d as horizon inputs')
        return horizon, changes
    
    def price_moves(self, horizon):
        horizon, changes = self.time_horizon(horizon)
        market_return = changes.copy()  # this is the change expressed as a percentage
        
        price_shift = changes.copy()    # this is to see what the price would  have moved to (rebased from todays price)
        price_shift += 1
        for p in list(price_shift):
            price_shift[p] *= self.spot_price[p]
            
        for m in list(changes):   # today's spot price multiplied by he historical daily price change
            changes[m] += 1
            changes[m] *= self.spot_price[m]
            
        changes = changes.iloc[:-1, :]  # remove the current date
        
        market_move = changes.copy()
        for m in list(market_move):
            market_move[m] -= self.spot_price[m]
        
        return horizon, changes, market_return, price_shift, market_move

    def sigma_prices(self, horizon):    # What is a 1, 1.5, 2 sigma price shift over the input horizon
        horizon, changes, market_return, price_shift, market_move = self.price_moves(horizon)
        
        products = []
        two_sigma = []
        onepointfive_sigma = []
        one_sigma = []
        
        for m in list(market_return):
            products.append(m)
            spot = self.spot_price[m]
            two_sigma.append(round(spot * (1+ market_return[m].quantile(0.05, interpolation='nearest')), 2))
            onepointfive_sigma.append(round(spot * (1+ market_return[m].quantile(0.13, interpolation='nearest')), 2))
            one_sigma.append(round(spot * (1+ market_return[m].quantile(0.32, interpolation='nearest')), 2))
            
        sigma_prices = pd.DataFrame()
        sigma_prices['Mkt'] = products
        sigma_prices['TwoSigma'] = two_sigma
        sigma_prices['OnePointFive'] = onepointfive_sigma
        sigma_prices['OneSigma'] = one_sigma
        
        return sigma_prices


m = Market_Info(mkts)
#horizon, changes, market_return, price_shift, market_move = m.price_moves('daily')
p = m.positions
