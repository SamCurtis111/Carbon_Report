# -*- coding: utf-8 -*-
"""
Position_Report_Calcs scratchad
"""
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
import calendar

mkts = ['ACCU','LGC','NZU','EUA', 'UKA', 'CCA','VCM']
positions = dict()
for m in mkts:
    positions[m] = pd.read_excel('Positions.xlsx', sheet_name=m)
    positions[m]['Expiry'] = pd.to_datetime(positions[m].Expiry).dt.date
    
random_date = dt.datetime.strptime('16-12-2024', '%d-%m-%Y').date()   
premiums = pd.read_excel('Positions.xlsx', sheet_name='Index') 
current_date = dt.datetime.today().date()
rates = 1.06    #### THIS IS JUST A RANDOM INPUT NUMBER

from Value_Derivatives import Derivatives
derivative = Derivatives(premiums)

from Position_Report_Calcs import Position_Reporting

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# __init__ values
mkt = 'EUA'
report = Position_Reporting(positions, mkt, current_date)
position_frame = positions.copy()
positions = position_frame[mkt]
fx = positions.FX[0]

fx_all = rates
date = current_date
FUM = premiums[premiums.Spread=='FUM']

EUA_CARRY = 1.037725

spot = positions[positions.Type=='Spot']
spot_price = spot.Price[0]

fwds = positions[positions.Type=='Fwd'].reset_index(drop=True)
fwds['Time'] = fwds.Expiry.astype('object') - date # Need to be able to alter this class to run it across multiple times
fwds['Time'] = [i.days/365 for i in fwds.Time]
fwds['Time'] = np.where(fwds.Time < 0, 0, fwds.Time)
fwds['Value'] = fwds.Price * fwds.Qty

ops = positions[positions.Type=='Option'].reset_index(drop=True)
ops['Time'] = ops.Expiry.astype('object') - date
ops['Time'] = [i.days/365 for i in ops.Time]
ops['Time'] = np.where(ops.Time < 0, 0, ops.Time)
ops['Value'] = ops.Price * ops.Qty




#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
## MKT RISK ALLOCATION; % PRICE MOVES
mkt_current = Position_Reporting(position_frame, mkt, current_date)
horizon=current_date

price_returns = [-0.15, -0.1, -0.05, 0, .05, .1, .15]

def create_returns(horizon):
    price_pnls = pd.DataFrame()
    price_ranges = pd.DataFrame()
    price_ranges['% Move'] = [int(i*100) for i in price_returns]
    for m in mkts:    #[:-1]:
        mkt_current = Position_Reporting(positions, m, horizon)
        spot = mkt_current.spot_price
        mkt_prices = [(1+i)*spot for i in price_returns]
        price_pnls[m] = round(mkt_current.price_moves(mkt_prices)['Total'],2)       #########################################
        price_ranges[m] = mkt_prices

    price_ranges = round(price_ranges,2)
    return price_ranges, price_returns, price_pnls

price_ranges, price_returns, price_pnls = create_returns(current_date)




PriceRange = mkt_prices[mkt]
def price_moves(self, PriceRange):    # to create a table and graph below the sigma moves graphs in the same style
    #mkt_prices = actual_prices.copy()
    #mkt_prices.set_index('Date', inplace=True)
    #sub = mkt_prices[mkt]
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Calc the PnLs
    ## SPOT PNL
    spots = [(i-spot_price)*spot.Qty[0]*fx for i in PriceRange]   # pnl on spot position
    
    # FWD 
    # Process:
        # we want the pnl of all the fwds at each entered price
        # at each price p, we take the pnl of each fwd i and append it to price_pnl
        # we then sum price pnl which is the pnl of all fwds at that price
        # we append that value to fwd pnls
    fwd_pnls = []
    for p in PriceRange:
        price_pnl = []
        for i in range(0, len(fwds)):
            current_fwd = fwds.iloc[i]
            current_fwd_value = current_fwd.Value
            current_fwd_rate = current_fwd.Rate            
            
            new_fwd_price = derivative.fwd_pricer(p, current_fwd_rate, current_fwd.Time, current_fwd.Subtype)
            value = new_fwd_price * current_fwd.Qty
            fwd_pnl = (value - current_fwd_value) * fx  
            
            price_pnl.append(fwd_pnl)
        fwd_pnls.append(sum(price_pnl))
        
    # OPTION PNL
    # Same as above, we loop through each of the prices and within that loop want to calculate pnl of every option at that price point then sum those values
    op_pnls = []
    for p in PriceRange:
        price_pnl = []
        option_prices = []
        option_prices_black = []
        for i in range(0, len(ops)):   # loop through each option
            current_op = ops.loc[i]
            current_op_value = current_op.Price
            current_op_type = current_op.Subtype
            current_op_rate = current_op.Rate
        
            if mkt=='EUA' or mkt=='CCA' or mkt=='OTHER':
                if current_op.Time < 0.4:
                    new_op_price = derivative.black76(current_op.Subtype, p, current_op.Strike, current_op_rate, current_op.Time, current_op.Vol)
                else:
                    new_p = p/(1+current_op_rate)**current_op.Time   # convert the EUA futures price to spot
                    new_op_price = derivative.black_scholes(current_op.Subtype, new_p, current_op.Strike, current_op_rate, current_op.Time, current_op.Vol)
                
            else:
                new_op_price = derivative.black_scholes(current_op.Subtype, p, current_op.Strike, current_op_rate, current_op.Time, current_op.Vol)
              
            value = new_op_price * current_op.Qty
            op_pnl = (value - current_op.Value) * fx
            
            price_pnl.append(op_pnl)
            option_prices.append(new_op_price)
            #option_prices_black.append(new_op_price76)
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

m='NZU'
PriceRange = changes[m]

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
## MKT RISK ALLOCATION; STD MOVES

freq='daily'
spot = position
calc_method=''

def std_moves(self, stds, freq, calc_method=''):   # alt calc_method = [neg_std_move, pos_std_move] which will be manually entered values to see what pnl happens to those specific values.
    mkt_prices = actual_prices.copy()
    mkt_prices.set_index('Date', inplace=True)
    sub = mkt_prices[mkt]
    
    if freq == 'daily':
        sub = sub.pct_change()[-126:] # last 6 months of daily prices
    elif freq=='weekly':
        sub = sub.resample('W').ffill().pct_change()[1:] # convert prices to weekly
    
    last_price = spot_price
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
    spot_profit = (price_increase - spot_price) * spot.Qty[0] * fx
    spot_loss = (price_decrease - spot_price) * spot.Qty[0] * fx
    spots = [spot_profit, spot_loss]
    
    # FWD 
    fwd_profits = []
    fwd_losses = []
    for i in range(0, len(fwds)):
        current_fwd = fwds.iloc[i]
        #current_fwd = fwds.iloc[i]
        current_fwd_value = current_fwd.Value
        current_fwd_rate = current_fwd.Rate            
        
        fwd_increase = derivative.fwd_pricer(price_increase, current_fwd_rate, current_fwd.Time, current_fwd.Subtype)
        value = fwd_increase * current_fwd.Qty
        fwd_profit = (value - current_fwd_value) * fx
        
        fwd_decrease = derivative.fwd_pricer(price_decrease, current_fwd_rate, current_fwd.Time, current_fwd.Subtype)
        value = fwd_decrease * current_fwd.Qty
        fwd_loss = (value - current_fwd_value) * fx
        
        fwd_profits.append(fwd_profit)
        fwd_losses.append(fwd_loss)
    fwd_results = [sum(fwd_profits), sum(fwd_losses)]
        
    # OPTION PNL
    op_profits = []
    op_losses = []
    for i in range(0, len(ops)):
        current_op = ops.loc[i]
        current_op_value = current_op.Price
        current_op_type = current_op.Subtype
        current_op_rate = current_op.Rate
        
        if mkt=='EUA' or mkt=='CCA' or mkt=='OTHER':
            spot_increase = price_increase/(1+current_op_rate)**current_op.Time # convert the futures price to spot
            spot_decrease = price_decrease/(1+current_op_rate)**current_op.Time # convert the futures price to spot
            
            op_price_increase = derivative.black_scholes(current_op.Subtype, spot_increase, current_op.Strike, current_op_rate, current_op.Time, current_op.Vol)
            op_price_decrease = derivative.black_scholes(current_op.Subtype, spot_decrease, current_op.Strike, current_op_rate, current_op.Time, current_op.Vol)
        else:
            op_price_increase = derivative.black_scholes(current_op.Subtype, price_increase, current_op.Strike, current_op_rate, current_op.Time, current_op.Vol)  
            op_price_decrease = derivative.black_scholes(current_op.Subtype, price_decrease, current_op.Strike, current_op_rate, current_op.Time, current_op.Vol)  
            
        op_value_increase = op_price_increase * current_op.Qty
        op_value_decrease = op_price_decrease * current_op.Qty
        
        op_profit = (op_value_increase - current_op.Value) * fx            
        op_loss = (op_value_decrease - current_op.Value) * fx
        
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


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# options(self) function
prices_opt = pd.DataFrame()
prices_opt['Price'] = report.prices

pnl_opt = pd.DataFrame()
pnl_opt['Price'] = report.prices

delta_opt = pd.DataFrame()
delta_opt['Price'] = report.prices

theta_opt = pd.DataFrame()
theta_opt['Price'] = report.prices

vega_opt = pd.DataFrame()
vega_opt['Price'] = report.prices

value_opt = pd.DataFrame()
value_opt['Price'] = report.prices

ops = report.ops   ## for debugging
i = 0    # specific option in report.ops

for i in range(0, len(report.ops)):
    current_op = report.ops.loc[i]
    current_op_value = current_op.Price
    current_op_type = current_op.Subtype
    current_op_rate = current_op.Rate
    
    pnls_opt = [] # create a list for all the option values at each price
    prices_opt_list = []
    deltas_opt = []
    thetas_opt = []
    vegas_opt = []
    values_opt = []
    
    for p in report.prices:
        if report.mkt=='EUA' or report.mkt=='CCA':  # for the EUA we need to run black-scholes against spot not futures price
            spot = p/(1+current_op_rate)**current_op.Time # convert the futures price to spot
            #if current_op.Expiry.year==2024:
            #    spot *= report.EUA_CARRY
            op_price = derivative.black_scholes(current_op.Subtype, spot, current_op.Strike, current_op_rate, current_op.Time, current_op.Vol)
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
        op_pnl = (op_value - current_op.Value) * report.fx
        op_delta = delta * current_op.Qty
        op_theta = theta * current_op.Qty
        op_vega = vega * current_op.Qty
        
        pnls_opt.append(op_pnl)
        prices_opt_list.append(op_price)
        deltas_opt.append(op_delta)
        thetas_opt.append(op_theta)
        vegas_opt.append(op_vega)
        values_opt.append(op_value * report.fx)
        
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

prices_opt['Option_Price'] = prices_opt.iloc[:,1:].sum(axis=1)    




#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# RUNNING IPYNB RERPORT CODE HERE
## Read in the position data
mkts = ['ACCU','NZU','EUA']

positions = dict()
for m in mkts:
    positions[m] = pd.read_excel('Positions.xlsx', sheet_name=m)
    positions[m]['Expiry'] = pd.to_datetime(positions[m].Expiry).dt.date




## Define the dates we want to use for reporting


_, last_day = calendar.monthrange(current_date.year, current_date.month)
end_of_month = dt.date(current_date.year, current_date.month, last_day)

three_months = current_date + dt.timedelta(days=3*30)

end_of_year = dt.datetime.strptime('10-12-2023', '%d-%m-%Y').date()

one_year = current_date + dt.timedelta(days=365)

dates_names = ['today','EoM','3 months','EoY','OneYear']
dates_values = [current_date, end_of_month, three_months, end_of_year, one_year]


dates = dict()
for i in list(range(0,len(dates_names))):
    dates[dates_names[i]] = dates_values[i]

nzus = dict()

for d in dates_values:
    nzus[d] = Position_Reporting(positions, 'NZU', d).combine_frame()
    
    
nzu_opts = dict()
for d in dates_values:
    nzu_opts[d] = Position_Reporting(positions, 'NZU', d).options()
    
    
    
    
 #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~   
 def combine_frame(self):
     pnl_fwds, delta_fwds, value_fwds = self.forwards()
     pnl_ops, delta_ops, theta_ops, vega_ops, value_ops = self.options()
     
     delta_frame = pd.DataFrame()
     delta_frame['Price'] = self.prices
     delta_frame['Spot'] = self.spot.Qty[0]
     delta_frame['Fwds'] = delta_fwds.Fwd_Delta
     delta_frame['Options'] = delta_ops.Option_Delta
     delta_frame['Total_Delta'] = delta_frame.iloc[:,1:].sum(axis=1)
     
     pnl_frame = pd.DataFrame()
     pnl_frame['Price'] = self.prices
     pnl_frame['Spot'] = ((pnl_frame.Price - self.spot.Price[0]) * self.spot.Qty[0]) * self.fx
     pnl_frame['Fwds'] = pnl_fwds.Fwd_Pnl
     pnl_frame['Options'] = pnl_ops.Option_Pnl
     pnl_frame['Total_Pnl'] = pnl_frame.iloc[:,1:].sum(axis=1)
     
     spot_value = pd.DataFrame()
     spot_value['Price'] = self.prices
     spot_value['Spot_Value'] = (spot_value['Price'] * self.spot.Qty[0]) * self.fx
     
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
 
    
derivative = Derivatives(premiums) 
derivative.black76('Call',69.56,65,0.03706,0.193,0.416)
derivative.black76('Put',69.56,50,0.03706,0.686,0.4582)





spot_price = 63.51

price_pnl = []
option_price = []
for i in range(0, len(ops)):   # loop through each option
    current_op = ops.loc[i]
    current_op_value = current_op.Price
    current_op_type = current_op.Subtype
    current_op_rate = current_op.Rate

    if mkt=='EUA' or mkt=='CCA' or mkt=='UKA' or mkt=='OTHER':
        new_op_price = derivative.black76(current_op.Subtype, p, current_op.Strike, current_op_rate, current_op.Time, current_op.Vol)
        #if current_op.Time < 0.4:  # black76 seems to wrok better for shorter dated options... counterintuitive... 
        #    new_op_price = derivative.black76(current_op.Subtype, p, current_op.Strike, current_op_rate, current_op.Time, current_op.Vol)
        #else:
        #    new_p = p/(1+current_op_rate)**current_op.Time   # convert the EUA futures price to spot
        #    new_op_price = derivative.black_scholes(current_op.Subtype, new_p, current_op.Strike, current_op_rate, current_op.Time, current_op.Vol)                    
    else:
        new_op_price = derivative.black_scholes(current_op.Subtype, p, current_op.Strike, current_op_rate, current_op.Time, current_op.Vol)
      
    value = new_op_price * current_op.Qty
    op_pnl = (value - current_op.Value) * fx
    
    price_pnl.append(op_pnl)
    option_price.append(new_op_price)