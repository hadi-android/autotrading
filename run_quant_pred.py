#import os
#os.chdir('C:\\Users\\hadit\\OneDrive - Prophix Software, Inc\\AutoTrading')
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from yahoo_historical import Fetcher
from quant_predict import quant_pred

ticker = 'BTC-USD'
dstart = [2018,12,1]
alg = 'macd'

df = Fetcher(ticker, dstart).getHistorical()	
#df = df[['Date','Close']]
df.Date = pd.to_datetime(df.Date)
# df = df[:-1]
df.tail()

start = '2019-1-1'
end = '2020-8-1'
signals, gl_alg, gl_ref = quant_pred(df,alg, start, end)

# end='2019-12-31'
# signals, gl_alg_19, gl_ref_19 = quant_pred(df, alg, start, end)

# start = '2020-1-1'
# end = '2020-7-10'
# signals, gl_alg_20, gl_ref_20 = quant_pred(df, alg, start, end)
