# -*- coding: utf-8 -*-
"""
Position_Report_Calcs scratchad
"""

report = Position_Reporting(positions, 'EUA', current_date)


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


for i in range(0, len(report.ops)):
#i=1  # want to look specifically at this option
#i=4   # this is one of the Dec24 options
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

    #prices = list(range(70,105))  # smaller range for debugging    
    for p in report.prices:
    #p = 100
        ## if its a dec24 product we need to convert p into dec24 price here
        #if current_op.Expiry.year == 2024:
        #    p *= 1.05
        spot = p/(1+current_op_rate)**current_op.Time # convert the futures price to spot
        if current_op.Expiry.year==2024:
            spot *= 1.04622
        op_price = derivative.black_scholes(current_op.Subtype, spot, current_op.Strike, current_op_rate, current_op.Time, current_op.Vol)
        theta = derivative.option_theta(spot, current_op.Strike, current_op_rate, current_op.Vol, current_op.Time, current_op_type)
        vega = derivative.option_vega(spot, current_op.Strike, current_op_rate, current_op.Vol, current_op.Time)/100            
        delta = derivative.option_delta(current_op.Subtype, spot, current_op.Strike, current_op_rate, current_op.Time, current_op.Vol)                    
       
        if math.isnan(op_price):
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

premiums = pd.read_excel('Positions.xlsx', sheet_name='Index') 


## Define the dates we want to use for reporting
current_date = dt.datetime.today().date()

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