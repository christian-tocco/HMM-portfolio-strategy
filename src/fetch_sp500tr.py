#pylint: disable=all

import yfinance as yf
import pandas as pd
import os

#Download weekly data from 2005 to 2025
ticker = "^SP500TR"
sp500tr = yf.download(
    ticker,
    start="2005-07-22",
    end="2025-07-11",
    interval="1wk",
    auto_adjust=True
)

#Preparation of the Dataframe
sp500tr = sp500tr[['Close']].dropna()
sp500tr.index.name = 'Date'
sp500tr.columns = ['Close']

os.makedirs('../data', exist_ok=True)

#Save the data in a CSV file
sp500tr.to_csv("../data/SP500TR.csv")

print("SP500 t.r. data succesfully saved in /data")