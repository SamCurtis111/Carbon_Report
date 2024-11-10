# -*- coding: utf-8 -*-
"""
27/06/2023
functionality to value portfolio to visualise in ipynb

for calculations the options:
    # In this loop we loop on i which is each option in the dataframe
    # We then sub-loop on p which is each theoretical price in the provided list
    # For that theoretical price p we calculate what the value of the option would be
    # We then calc the option pnl by substracting that from the current option value
    # Option values are multiplied by qty so that we get the realised pnl
    # These values are then added into price frame which is a pnl matrix for each option
"""
import os
os.chdir('C:\\GitHub\\Carbon_Report')

import pandas as pd
import numpy as np
import math
from math import log, exp, sqrt
from scipy.stats import norm

from sqlalchemy import create_engine
engine = create_engine('postgresql://postgres:iforgot23@localhost/ACCU')

import datetime as dt
#from dt import date
#from datetime import date
import calendar

mkts = ['ACCU','LGC','NZU','EUA', 'UKA', 'CCA','RGGI','VCM','OTHER']
compliance_mkts = ['ACCU','NZU','EUA', 'UKA', 'CCA','RGGI']
positions = dict()
for m in mkts:
    positions[m] = pd.read_excel('Positions.xlsx', sheet_name=m)
    positions[m]['Expiry'] = pd.to_datetime(positions[m].Expiry).dt.date

premiums = pd.read_excel('Positions.xlsx', sheet_name='Index')

actual_prices = pd.read_excel('Positions.xlsx', sheet_name='Prices')
current_prices = premiums[premiums.Spread.isin(mkts)]

rates = ['USDAUD','GBPAUD','EURAUD','NZDAUD']
rates = premiums[premiums.Spread.isin(rates)]

from Value_Derivatives import Derivatives
derivative = Derivatives(premiums)

from Retrieve_Market_Info import Market_Info
market_info = Market_Info(compliance_mkts)
spot_prices = market_info.spot_price
fx_rates = market_info.fx_rates

#from Test_Portfolio_Historically import Portfolio_Performance_And_Risk
#portfolio = Portfolio_Performance_And_Risk('daily')

# Dates to use in calcs
# [today, last day of current month, 3 months, end of year]
# Get the last day of the current month
current_date = dt.datetime.today().date()

_, last_day = calendar.monthrange(current_date.year, current_date.month)
end_of_month = dt.date(current_date.year, current_date.month, last_day)

three_months = current_date + dt.timedelta(days=3*30) # CAN ENTER THIS MANUALLY
six_months = current_date + dt.timedelta(days=6*30) # CAN ENTER THIS MANUALLY   
#three_months = three_months.date()

end_of_year = dt.datetime.strptime('30-12-2024', '%d-%m-%Y').date()

one_year = current_date + dt.timedelta(days=365) # CAN ENTER THIS MANUALLY




class Position_Reporting:
    def __init__(self, position_frames, market, date):
        self.mkt = market # which market are we generating data for
        self.markets = ['ACCU','NZU','EUA','UKA','CCA','RGGI'] # The number of markets being tested across beta allocation
        self.positions = position_frames[self.mkt].copy() 
        self.fx = self.positions.FX[0]  # 
        self.fx_all = rates.copy()    # all FX rates
        self.date = date
        
        self.FUM = premiums[premiums.Spread=='FUM']
        self.FUM = self.FUM['Price'].reset_index(drop=True)[0]
        
        self.EUA_CARRY = 1.03148   #### THIS NEEDS MANUAL ADJUSTING
        
        self.prices = self.price_ranges()  # dont need a price range for VCM
        self.current_prices = current_prices.copy()
       
        self.spot = self.positions[self.positions.Type=='Spot']
        self.spot_qty = self.spot.Qty.sum()
        self.spot_price = self.spot.Price[0]
        
        self.fwds = self.positions[self.positions.Type=='Fwd'].reset_index(drop=True)
        self.fwds['Time'] = self.fwds.Expiry.astype('object') - self.date # Need to be able to alter this class to run it across multiple times
        self.fwds['Time'] = [i.days/365 for i in self.fwds.Time]
        self.fwds['Time'] = np.where(self.fwds.Time < 0, 0, self.fwds.Time)
        self.fwds['Value'] = self.fwds.Price * self.fwds.Qty
         
        
        self.ops = self.positions[self.positions.Type=='Option'].reset_index(drop=True)
        self.ops['Time'] = self.ops.Expiry.astype('object') - self.date
        self.ops['Time'] = [i.days/365 for i in self.ops.Time]
        self.ops['Time'] = np.where(self.ops.Time < 0, 0, self.ops.Time)
        self.ops['Value'] = self.ops.Price * self.ops.Qty
        
        
    def price_ranges(self):
        if self.mkt=='ACCU':
            price_range = list(range(20,46))
        elif self.mkt=='NZU':
            price_range = list(range(30,91))
        elif self.mkt=='LGC':
            price_range = list(range(25,51))            
        elif self.mkt=='EUA':
            price_range = list(range(30,101))
        elif self.mkt=='UKA':
            price_range = list(range(25,71))
        elif self.mkt=='CCA':
            price_range = list(range(20,56)) 
        elif self.mkt=='RGGI':
            price_range = list(range(10,41))             
        elif self.mkt=='VCM':
            price_range = list(range(0,101)) # random price range for VCM so the code runs
        elif self.mkt=='OTHER':
            price_range = list(range(30,101)) # random price range for OTHER... do the zooming etc yourself because it could be any market
        else:
            print('Mkt text entry error use either; ACCU, NZU, EUA, UKA, CCA,RGGI')
        return price_range    
    
    def forwards(self):
        pnl_fwd = pd.DataFrame()
        pnl_fwd['Price'] = self.prices
        
        delta_fwd = pd.DataFrame()
        delta_fwd['Price'] = self.prices
        
        value_fwd = pd.DataFrame()
        value_fwd['Price'] = self.prices        
        
        for i in range(0, len(self.fwds)):
            current_fwd = self.fwds.iloc[i]
            current_fwd_value = current_fwd.Value
            current_fwd_rate = current_fwd.Rate
            
            pnls_fwd = []
            prices_fwd = []
            deltas_fwd = []
            values_fwd = []
        
            for p in self.prices:
                fwd_price = derivative.fwd_pricer(p, current_fwd_rate, current_fwd.Time, current_fwd.Subtype)
                
                fwd_value = fwd_price * current_fwd.Qty
                pnl = (fwd_value - current_fwd_value) * self.fx
                #delta_fwd = current_fwd.Qty
                
                pnls_fwd.append(pnl)
                values_fwd.append(fwd_value * self.fx)
                
            pnl_fwd[current_fwd.Name] = pnls_fwd
            delta_fwd[current_fwd.Name] = current_fwd.Qty
            value_fwd[current_fwd.Name] = values_fwd
        
        pnl_fwd['Fwd_Pnl'] = pnl_fwd.iloc[:,1:].sum(axis=1)      
        delta_fwd['Fwd_Delta'] = delta_fwd.iloc[:,1:].sum(axis=1)
        value_fwd['Fwd_Value'] = value_fwd.iloc[:,1:].sum(axis=1)
        
        return pnl_fwd, delta_fwd, value_fwd
    
    def options(self):
        prices_opt = pd.DataFrame()
        prices_opt['Price'] = self.prices
        
        pnl_opt = pd.DataFrame()
        pnl_opt['Price'] = self.prices
        
        delta_opt = pd.DataFrame()
        delta_opt['Price'] = self.prices
        
        theta_opt = pd.DataFrame()
        theta_opt['Price'] = self.prices
        
        vega_opt = pd.DataFrame()
        vega_opt['Price'] = self.prices
        
        value_opt = pd.DataFrame()
        value_opt['Price'] = self.prices        

        for i in range(0, len(self.ops)):
            current_op = self.ops.loc[i]
            current_op_value = current_op.Price
            current_op_type = current_op.Subtype
            current_op_rate = current_op.Rate
            
            pnls_opt = [] # create a list for all the option values at each price
            prices_opt_list = []
            deltas_opt = []
            thetas_opt = []
            vegas_opt = []
            values_opt = []
            
            for p in self.prices:
                if self.mkt=='EUA' or self.mkt=='CCA' or self.mkt=='UKA' or self.mkt=='OTHER':  # for the EUA we need to run black-scholes against spot not futures price
                    spot = p/(1+current_op_rate)**current_op.Time # convert the futures price to spot
                    #op_price = derivative.black_scholes(current_op.Subtype, spot, current_op.Strike, current_op_rate, current_op.Time, current_op.Vol)
                    op_price = derivative.black76(current_op.Subtype, p, current_op.Strike, current_op_rate, current_op.Time, current_op.Vol)    ## USE BLACK76 (its what we use in excel)
                    theta = derivative.option_theta(spot, current_op.Strike, current_op_rate, current_op.Vol, current_op.Time, current_op_type)
                    vega = derivative.option_vega(spot, current_op.Strike, current_op_rate, current_op.Vol, current_op.Time)/100            
                    delta = derivative.option_delta(current_op.Subtype, spot, current_op.Strike, current_op_rate, current_op.Time, current_op.Vol)                    
                else:
                    op_price = derivative.black_scholes(current_op.Subtype, p, current_op.Strike, current_op_rate, current_op.Time, current_op.Vol)
                    theta = derivative.option_theta(p, current_op.Strike, current_op_rate, current_op.Vol, current_op.Time, current_op_type)
                    vega = derivative.option_vega(p, current_op.Strike, current_op_rate, current_op.Vol, current_op.Time)/100            
                    delta = derivative.option_delta(current_op.Subtype, p, current_op.Strike, current_op_rate, current_op.Time, current_op.Vol)                    
                
                if math.isnan(op_price):      # IF THE OPTION IS EXPIRED IT SHOULDNT NECESSARILY BE WORTH ZERO
                    op_price=0   
                
                op_value = op_price * current_op.Qty
                op_pnl = (op_price - current_op.Price) * current_op.Qty * self.fx
                #op_pnl = (op_value - current_op.Value) * self.fx
                op_delta = delta * current_op.Qty
                op_theta = theta * current_op.Qty
                op_vega = vega * current_op.Qty
                
                pnls_opt.append(op_pnl)
                prices_opt_list.append(op_price)
                deltas_opt.append(op_delta)
                thetas_opt.append(op_theta)
                vegas_opt.append(op_vega)
                values_opt.append(op_value * self.fx)
                
            pnl_opt[current_op.Name] = pnls_opt
            prices_opt[current_op.Name] = prices_opt_list
            delta_opt[current_op.Name] = deltas_opt
            theta_opt[current_op.Name] = thetas_opt
            vega_opt[current_op.Name] = vegas_opt
            value_opt[current_op.Name] = values_opt
            
            
        pnl_opt['Option_Pnl'] = pnl_opt.iloc[:,1:].sum(axis=1)
        delta_opt['Option_Delta'] = delta_opt.iloc[:,1:].sum(axis=1)
        theta_opt['Option_Theta'] = theta_opt.iloc[:,1:].sum(axis=1)
        vega_opt['Option_Vega'] = vega_opt.iloc[:,1:].sum(axis=1)
        value_opt['Option_Value'] = value_opt.iloc[:,1:].sum(axis=1)
        
        return pnl_opt, delta_opt, theta_opt, vega_opt, value_opt
    
    def combine_frame(self):
        pnl_fwds, delta_fwds, value_fwds = self.forwards()
        pnl_ops, delta_ops, theta_ops, vega_ops, value_ops = self.options()
        
        delta_frame = pd.DataFrame()
        delta_frame['Price'] = self.prices
        delta_frame['Spot'] = self.spot_qty
        delta_frame['Fwds'] = delta_fwds.Fwd_Delta
        delta_frame['Options'] = delta_ops.Option_Delta
        delta_frame['Total_Delta'] = delta_frame.iloc[:,1:].sum(axis=1)
        
        pnl_frame = pd.DataFrame()
        pnl_frame['Price'] = self.prices
        pnl_frame['Spot'] = ((pnl_frame.Price - self.spot.Price[0]) * self.spot_qty) * self.fx
        pnl_frame['Fwds'] = pnl_fwds.Fwd_Pnl
        pnl_frame['Options'] = pnl_ops.Option_Pnl
        pnl_frame['Total_Pnl'] = pnl_frame.iloc[:,1:].sum(axis=1)
        
        spot_value = pd.DataFrame()
        spot_value['Price'] = self.prices
        spot_value['Spot_Value'] = (spot_value['Price'] * self.spot_qty) * self.fx
        
        #value_frame = pd.DataFrame()  # This section is the sum of notional face value
        #value_frame['Price'] = self.prices
        #value_frame['Fwd_Value'] = value_fwds['Fwd_Value']
        #value_frame['Option_Value'] = value_ops['Option_Value']
        #value_frame['Spot_Value'] = spot_value['Spot_Value']
        #value_frame['Position_Value'] = value_frame.iloc[:,1:].sum(axis=1)

        value_frame = delta_frame.copy()  # This section is the sum of delta
        value_frame = value_frame.iloc[:,:-1]
        value_frame['Spot'] = (value_frame['Spot'] * value_frame.Price) * self.fx
        value_frame['Fwds'] = (value_frame.Fwds * value_frame.Price) * self.fx
        value_frame['Options'] = (value_frame.Options * value_frame.Price) * self.fx
        value_frame['Total_Allocation'] = value_frame.iloc[:,1:].sum(axis=1)
        
        return pnl_frame, delta_frame, theta_ops, vega_ops, value_frame
    
    def beta_units(self):   # How many units does a fully allocated beta portoflio hold
        market_allocation = self.FUM * 1/len(self.markets)
        
        aud_spot_prices = spot_prices.copy()
        for m in list(aud_spot_prices):
            aud_spot_prices[m] *= fx_rates[m]
            
        market_units = aud_spot_prices.copy()
        for m in market_units:
            market_units[m] = round(market_allocation/market_units[m])
        return market_units
            
    
    def beta_pnl(self): # TO COMPARE THE CURRENT PORTFOLIO AGAINST A BETA PORTFOLIO AND DERIVE ALPHA
        units = self.beta_units()
        units = units[self.mkt]
        pnl_frame = pd.DataFrame()
        pnl_frame['Price'] = self.prices
        pnl_frame['Spot'] = ((pnl_frame.Price - self.spot.Price[0]) * units) * self.fx
        return pnl_frame
        
        
    
    def current_values(self): # get the current AUD value of the positions in each mkt
        spot = self.spot.copy()
        spot['Value'] = spot.Qty * spot.Price * spot.FX
        spot_value = int(spot.Value.sum())
        
        fwd = self.fwds.copy()
        fwd['Value'] = fwd.Qty * fwd.Price * fwd.FX
        fwd_value = int(fwd.Value.sum())
        
        op = self.ops.copy()
        op['Value'] = op.Qty * op.Price * op.FX
        op_value = int(op.Value.sum())
        return [spot_value, fwd_value, op_value]

    
    def std_moves(self, stds, freq, calc_method=''):   # alt calc_method = [neg_std_move, pos_std_move] which will be manually entered values to see what pnl happens to those specific values.
        mkt_prices = actual_prices.copy()
        mkt_prices.set_index('Date', inplace=True)
        sub = mkt_prices[self.mkt]
        
        if freq == 'daily':
            sub = sub.pct_change()[-126:] # last 6 months of daily prices
        elif freq=='weekly':
            sub = sub.resample('W').ffill().pct_change()[1:] # convert prices to weekly
        elif freq=='monthly':
            sub = sub.resample('M').ffill().pct_change()[1:] # convert prices to weekly            
        
        last_price = self.spot_price
        # Calculate positive and negative one standard deviation price moves
        if calc_method == '':
            if stds==1:
                positive_std_move = np.percentile(sub, 84.13)
                negative_std_move = np.percentile(sub, 15.87)
            elif stds==1.5:
                positive_std_move = np.percentile(sub, 94.65)
                negative_std_move = np.percentile(sub, 5.65)            
            elif stds==2:
                positive_std_move = np.percentile(sub, 97.72)
                negative_std_move = np.percentile(sub, 2.28)
                
            price_increase = (1+positive_std_move)*last_price
            price_decrease = (1+negative_std_move)*last_price
            
        else:
            # this functionality allows you to manually enter values into the function rather than using sigma
            price_increase = calc_method[0]    
            price_decrease = calc_method[1]
            

        sigma_prices = [price_increase, price_decrease]
        
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Calc the PnLs
        ## SPOT PNL
        spot_profit = (price_increase - self.spot_price) * self.spot_qty * self.fx
        spot_loss = (price_decrease - self.spot_price) * self.spot_qty * self.fx
        spots = [spot_profit, spot_loss]
        
        # FWD 
        fwd_profits = []
        fwd_losses = []
        for i in range(0, len(self.fwds)):
            current_fwd = self.fwds.iloc[i]
            #current_fwd = fwds.iloc[i]
            current_fwd_value = current_fwd.Value
            current_fwd_rate = current_fwd.Rate            
            
            fwd_increase = derivative.fwd_pricer(price_increase, current_fwd_rate, current_fwd.Time, current_fwd.Subtype)
            value = fwd_increase * current_fwd.Qty
            fwd_profit = (value - current_fwd_value) * self.fx
            
            fwd_decrease = derivative.fwd_pricer(price_decrease, current_fwd_rate, current_fwd.Time, current_fwd.Subtype)
            value = fwd_decrease * current_fwd.Qty
            fwd_loss = (value - current_fwd_value) * self.fx
            
            fwd_profits.append(fwd_profit)
            fwd_losses.append(fwd_loss)
        fwd_results = [sum(fwd_profits), sum(fwd_losses)]
            
        # OPTION PNL
        op_profits = []
        op_losses = []
        for i in range(0, len(self.ops)):
            current_op = self.ops.loc[i]
            current_op_value = current_op.Price
            current_op_type = current_op.Subtype
            current_op_rate = current_op.Rate
            
            if self.mkt=='EUA' or self.mkt=='CCA' or self.mkt=='RGGI' or self.mkt=='OTHER':
                spot_increase = price_increase/(1+current_op_rate)**current_op.Time # convert the futures price to spot
                spot_decrease = price_decrease/(1+current_op_rate)**current_op.Time # convert the futures price to spot
                
                op_price_increase = derivative.black_scholes(current_op.Subtype, spot_increase, current_op.Strike, current_op_rate, current_op.Time, current_op.Vol)
                op_price_decrease = derivative.black_scholes(current_op.Subtype, spot_decrease, current_op.Strike, current_op_rate, current_op.Time, current_op.Vol)
            else:
                op_price_increase = derivative.black_scholes(current_op.Subtype, price_increase, current_op.Strike, current_op_rate, current_op.Time, current_op.Vol)  
                op_price_decrease = derivative.black_scholes(current_op.Subtype, price_decrease, current_op.Strike, current_op_rate, current_op.Time, current_op.Vol)  
                
            op_value_increase = op_price_increase * current_op.Qty
            op_value_decrease = op_price_decrease * current_op.Qty
            
            op_profit = (op_value_increase - current_op.Value) * self.fx            
            op_loss = (op_value_decrease - current_op.Value) * self.fx
            
            op_profits.append(op_profit)
            op_losses.append(op_loss)
        op_results = [sum(op_profits), sum(op_losses)]
        
        sigma_prices = [round(i,2) for i in sigma_prices]
        
        sigma_frame = pd.DataFrame()
        sigma_frame['Spot'] = spots
        sigma_frame['Forwards'] = fwd_results
        sigma_frame['Options'] = op_results
        sigma_frame['Total'] = sigma_frame.sum(axis=1)
        sigma_frame['Price'] = sigma_prices
        sigma_frame = sigma_frame[['Price','Spot','Forwards','Options','Total']]
            
        return sigma_frame
    
    
    def price_moves(self, PriceRange):    # to create a table and graph below the sigma moves graphs in the same style
        #mkt_prices = actual_prices.copy()
        #mkt_prices.set_index('Date', inplace=True)
        #sub = mkt_prices[self.mkt]
        
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Calc the PnLs
        ## SPOT PNL
        spots = [(i-self.spot_price)*self.spot_qty*self.fx for i in PriceRange]   # pnl on spot position
        
        # FWD 
        # Process:
            # we want the pnl of all the fwds at each entered price
            # at each price p, we take the pnl of each fwd i and append it to price_pnl
            # we then sum price pnl which is the pnl of all fwds at that price
            # we append that value to fwd pnls
        fwd_pnls = []
        for p in PriceRange:
            price_pnl = []
            for i in range(0, len(self.fwds)):
                current_fwd = self.fwds.iloc[i]
                current_fwd_value = current_fwd.Value
                current_fwd_rate = current_fwd.Rate            
                
                new_fwd_price = derivative.fwd_pricer(p, current_fwd_rate, current_fwd.Time, current_fwd.Subtype)
                value = new_fwd_price * current_fwd.Qty
                fwd_pnl = (value - current_fwd_value) * self.fx  
                
                price_pnl.append(fwd_pnl)
            fwd_pnls.append(sum(price_pnl))
            
        # OPTION PNL
        # Same as above, we loop through each of the prices and within that loop want to calculate pnl of every option at that price point then sum those values
        op_pnls = []
        for p in PriceRange:
            price_pnl = []
            for i in range(0, len(self.ops)):   # loop through each option
                current_op = self.ops.loc[i]
                current_op_value = current_op.Price
                current_op_type = current_op.Subtype
                current_op_rate = current_op.Rate
            
                if self.mkt=='EUA' or self.mkt=='CCA' or self.mkt=='UKA' or self.mkt=='OTHER':
                    new_op_price = derivative.black76(current_op.Subtype, p, current_op.Strike, current_op_rate, current_op.Time, current_op.Vol)
                    #if current_op.Time < 0.4:  # black76 seems to wrok better for shorter dated options... counterintuitive... 
                    #    new_op_price = derivative.black76(current_op.Subtype, p, current_op.Strike, current_op_rate, current_op.Time, current_op.Vol)
                    #else:
                    #    new_p = p/(1+current_op_rate)**current_op.Time   # convert the EUA futures price to spot
                    #    new_op_price = derivative.black_scholes(current_op.Subtype, new_p, current_op.Strike, current_op_rate, current_op.Time, current_op.Vol)                    
                else:
                    new_op_price = derivative.black_scholes(current_op.Subtype, p, current_op.Strike, current_op_rate, current_op.Time, current_op.Vol)
                  
                value = new_op_price * current_op.Qty
                op_pnl = (value - current_op.Value) * self.fx
                
                price_pnl.append(op_pnl)
            op_pnls.append(sum(price_pnl))

        # PUT IT ALL TOGETHER        
        price_frame = pd.DataFrame()
        price_frame['Spot'] = spots
        price_frame['Forwards'] = fwd_pnls
        price_frame['Options'] = op_pnls
        price_frame['Total'] = price_frame.sum(axis=1)
        price_frame['Price'] = PriceRange
        price_frame = price_frame[['Price','Spot','Forwards','Options','Total']]
            
        return price_frame    
    
    
    def rolling_vol(self):
        return derivative.calculate_rolling_volatility(self.mkt, actual_prices)






    class ACCU_ANALYSIS:    ## STILL WORKING THROUGH THIS ##
        def __init__(self):
            self.query = 'select * from \"ERF_Projects\"'
            
        def method_rename(self,sub):
             # Convert cell values containing substring "Landfill Gas" to "LFG"
            sub['Method'] = sub['Method'].apply(lambda x: 'Avoided Clearing' if 'Avoided Clearing' in x else x)
            sub['Method'] = sub['Method'].apply(lambda x: 'Avoided Deforestation' if 'Avoided Deforestation' in x else x)
            
            sub['Method'] = sub['Method'].apply(lambda x: 'Soil Carbon' if 'Soil Organic Carbon' in x else x)
            sub['Method'] = sub['Method'].apply(lambda x: 'Soil Carbon' if 'Soil Carbon' in x else x)
            sub['Method'] = sub['Method'].apply(lambda x: 'Soil Carbon' if 'Carbon in Soils' in x else x)
            
            sub['Method'] = sub['Method'].apply(lambda x: 'Afforestation' if 'Mallee Plantings' in x else x)
            sub['Method'] = sub['Method'].apply(lambda x: 'Afforestation' if 'Afforestation' in x else x)
            sub['Method'] = sub['Method'].apply(lambda x: 'NFMR' if 'Native Forest from Managed Regrowth' in x else x)
            sub['Method'] = sub['Method'].apply(lambda x: 'HIR' if 'Human-Induced' in x else x)
            sub['Method'] = sub['Method'].apply(lambda x: 'Environmental Plantings' if 'Environmental Plantings' in x else x)
            sub['Method'] = sub['Method'].apply(lambda x: 'Plantation Forestry' if 'Plantation Forestry' in x else x)
            sub['Method'] = sub['Method'].apply(lambda x: 'Savanna Burning' if 'Savanna' in x else x)
            
            sub['Method'] = sub['Method'].apply(lambda x: 'Organic Waste' if 'Organic Waste' in x else x)
            sub['Method'] = sub['Method'].apply(lambda x: 'Effluent Management' if 'Animal Effluent Management' in x else x)
            sub['Method'] = sub['Method'].apply(lambda x: 'Piggeries' if 'Piggeries' in x else x)
            sub['Method'] = sub['Method'].apply(lambda x: 'Herd Management' if 'Herd Management' in x else x)
            
            sub['Method'] = sub['Method'].apply(lambda x: 'Land and Sea Transport' if 'Land and Sea Transport' in x else x)
            sub['Method'] = sub['Method'].apply(lambda x: 'Commercial Buildings' if 'Commercial Buildings' in x else x)
            sub['Method'] = sub['Method'].apply(lambda x: 'Commercial and Public Lighting' if 'Commercial and Public Lighting' in x else x)
            sub['Method'] = sub['Method'].apply(lambda x: 'Commercial Appliances' if 'Commercial Appliances' in x else x)
            
            sub['Method'] = sub['Method'].apply(lambda x: 'LFG' if 'Landfill Gas' in x else x)
            sub['Method'] = sub['Method'].apply(lambda x: 'Alternative Waste' if 'Alternative Waste' in x else x)
            sub['Method'] = sub['Method'].apply(lambda x: 'Coal Mine Waste Gas' if 'Coal Mine Waste Gas' in x else x)
            sub['Method'] = sub['Method'].apply(lambda x: 'Wastewater' if 'Wastewater' in x else x)
            
            sub['Method'] = sub['Method'].apply(lambda x: 'Verified Carbon Standard' if 'Verified Carbon Standard' in x else x)
            
            sub['Method'] = sub['Method'].apply(lambda x: 'Facilities' if 'Initiative - Facilities' in x else x)
            return sub
    
        def grouped_analysis(self):
            df = pd.read_sql(self.query, engine)
            
            
            
            
#df['Last_Updated_ERF'] = pd.to_datetime(df['Last_Updated_ERF']).dt.date
#df = df[df.Last_Updated_ERF >= pd.to_datetime('2023-06-25')]
#df = df.drop_duplicates()

#df = self.method_rename(df)            

#df['ACCUs Total units issued'] = df['ACCUs Total units issued'].astype(int)
#df = df[['Project ID','Project Name','Method','ACCUs Total units issued','Last_Updated_ERF','Run Date']]











# [current_date, end_of_month, three_months, end_of_year]
#report = Position_Reporting(positions, 'LGC', current_date)
#report = Position_Reporting(positions, 'ACCU', current_date)
#report = Position_Reporting(positions, 'NZU', current_date)
#report = Position_Reporting(positions, 'EUA', current_date)

#report.beta_units()
#report.beta_pnl()


#report = Position_Reporting(positions, 'ACCU', end_of_month)
#report = Position_Reporting(positions, 'ACCU', end_of_year)
#report = Position_Reporting(positions, 'ACCU', one_year)
#report.prices

#p, d, value = report.forwards()
#p, d, t, v, value = report.options()#
#


#p, d, t, v, value = report.combine_frame()
#pos_values = report.current_values()
#
#sigma_pnl = report.std_moves(1,'daily')
#sigma_pnl = report.std_moves(2,'daily')
#sigma_pnl = report.std_moves(2,'weekly')
#
#price_pnl = report.price_moves([20,25,35,40])
#
#
#st_prices = report.std_moves(2,'daily')
#report.std_moves(2,'weekly')
#
#

## DEBUGGING THE IPYNB GRAPHING CODE
#def reporting_date(run_date='today'):
#    if run_date=='today':
#        current_date = dt.datetime.today().date()
#        current_year = current_date.year
#    else:
#        current_date = run_date
#        current_year = current_date.year
#    return current_date, current_year
#        
#current_date, current_year = reporting_date()    # as at today
#_, last_day = calendar.monthrange(current_date.year, current_date.month)
#end_of_month = dt.date(current_date.year, current_date.month, last_day)
#three_months = current_date + dt.timedelta(days=3*30)
#six_months = current_date + dt.timedelta(days=6*30)
#dates_names = ['today','EoM']
#dates_values = [current_date, end_of_month]
#
#euas = dict()
#for d in dates_values:
#    euas[d] = Position_Reporting(positions, 'EUA', d).combine_frame()
#    
#mkt_data = euas.copy()
#
#p = Portfolio_Performance_And_Risk(current_date)
