#pylint: disable=all

from ib_insync import *
import pandas as pd
import os

#Connection to IBKR 
ib = IB()
ib.connect('127.0.0.1', 7497, clientId=1)

#Definition of the contracts
spy = Stock('SPY', 'SMART', 'USD')
gld = Stock('GLD', 'SMART', 'USD')

#Function do download data (20 years)
def get_weekly_data(contract, end='20250711 23:59:59', duration='20 Y'):
    bars = ib.reqHistoricalData(
        contract,
        endDateTime=end,
        durationStr=duration,
        barSizeSetting='1 week',
        whatToShow='TRADES',
        useRTH=True,
        formatDate=1
    )
    df = util.df(bars)
    df.set_index('date', inplace=True)
    df.index.name = 'Date'
    df.columns = [col.capitalize() for col in df.columns]
    return df

spy_df = get_weekly_data(spy)
gld_df = get_weekly_data(gld)

os.makedirs('../data', exist_ok=True)

#Save the data in CSV files
spy_df.to_csv('../data/SPY.csv')
gld_df.to_csv('../data/GLD.csv')

print("SPY and GLD data succesfully saved in /data")

#Disconnection to IBKR
ib.disconnect()