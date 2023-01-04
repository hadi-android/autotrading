#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from yahoo_historical import Fetcher
import datetime
import time


# <h3> Download QQQ price data from 2018-12-1 to 2020-9-9

# In[12]:
def quant_pred(ticker, start,end):

    #ticker = 'ETH-CAD'
    #dstart = '1-1-2021' #start date of data to calculate rolling averages
    #actual_start = '1-1-2022' # actual start date where data is analyzed
    
    
    
    
    dstart = datetime.datetime.strptime(start, '%m-%d-%Y') -datetime.timedelta(days = 50) 
    dstart = time.mktime(dstart.timetuple())
    
    df = Fetcher(ticker, dstart).get_historical()
    df.Date = pd.to_datetime(df.Date)
    # df = df[:-1]
    df.head()
    
    
    rsi_lower = 40
    rsi_upper = 75
    
    #52 ema of volume
    df.Volume = df.Volume*1e-9
    df['vol_avg'] = df.Volume.ewm(span=50,adjust=False).mean()
    #df['vol_std'] = df.Volume.ewm(span=52,adjust=False).std()
    df['vol_quantile'] = df.Volume.rolling(50).quantile(0.75, interpolation='midpoint')
    
    df.iloc[50:,:].head()
    
    
    # In[2]:
    
    
    #RSI
    df['chg'] = df.Close-df.Close.shift(1)
    N=14
    df['RSI']=np.zeros(df.shape[0])
    df.RSI = np.nan
    i=0
    while((i+N) < df.shape[0]):
        chg = df.chg[i:i+N]
        pos_chg = chg[chg>0].sum()/N
        neg_chg = abs(chg[chg<0].sum())/N
        RS = pos_chg/neg_chg    
        df['RSI'][i+N] = 100-(100/(1+RS))
        i+=1
    
    df=df[(df.Date>=pd.to_datetime(start)) & (df.Date<=pd.to_datetime(end))].reset_index(drop=True)
    
    df.RSI.min()
    df.RSI.max()   
    
    thresh = df.copy()
    thresh['lower'] = rsi_lower #40
    thresh['upper'] = rsi_upper #75    
       
    plt.figure(figsize=(20,5))
    plt.plot(df.Date, df.RSI)
    plt.plot(thresh.Date, thresh.lower)
    plt.plot(thresh.Date, thresh.upper)
    plt.savefig('rsi.png')
    plt.show()
       
    # df['buy_rsi'] = df.RSI <=70
    # df['buy_rsi'] = df.RSI>=50
    # df.buy_rsi = df.buy_rsi.astype(int)
    # df.head(10)    
    
        
    state=0
    df_sig = []
    buy_rsi = []
    dates_rsi = []
    for i in range(1,df.shape[0]):
        #long
        if((df.RSI[i]<=40) & (df.RSI[i]>df.RSI[i-1]) & (state==0)):
            state = 1
            df_sig.append(df.iloc[i,:].to_frame().T)
            buy_rsi.append(1)
            start_date = df.Date[i]
        #short
        if((df.RSI[i]>=75) & (df.RSI[i]<df.RSI[i-1]) & (state==1)):
            state=0
            df_sig.append(df.iloc[i].to_frame().T)
            buy_rsi.append(0)
            end_date = df.Date[i]
            dates_rsi.append(pd.date_range(start_date,end_date))
    # added a comment        
    df_sig = pd.concat(df_sig, axis=0)
    df_sig = df_sig.reset_index(drop=True)
    df_sig['buy_rsi'] = buy_rsi
    
    df_sig['gainloss'] =100*(df_sig.Close-df_sig.Close.shift(1))/df_sig.Close.shift(1)
    df_sig1 = df_sig[df_sig.buy_rsi==0]    
    
    gainloss_rsi = 100*(np.prod(1+df_sig1.gainloss/100)-1)  
    
    import matplotlib
    cmap = matplotlib.colors.ListedColormap(["red","green"], name='from_list', N=None)
    # if(alg=='rsi'):
    plt.figure(figsize=(20,5))
    plt.title('Buy(green)/sell(red) signals based on RSI')
    plt.plot(df.Date, df.Close)
    plt.scatter(df_sig.Date, df_sig.Close, c=df_sig.buy_rsi, cmap=cmap)
    plt.savefig('buy_sell_rsi.png')
    plt.show()
        
    
    
    # In[3]:
    
    
    df.RSI.head()
    
    
    # In[4]:
    
    
    #macd
    
    exp1 = df.Close.ewm(span=12, adjust=False).mean()
    exp2 = df.Close.ewm(span=26, adjust=False).mean()
    
    #df['macd'] = exp1-exp2
    #df['signal'] = macd.ewm(span=9, adjust=False).mean()
    macd = exp1-exp2
    signal = macd.ewm(span=9, adjust=False).mean()
    df['macd'] = macd-signal
    
    
    
    # In[107]:
    
    
    plt.figure(figsize=(20,5))
    plt.plot(df.Date, macd, label='MACD', color = 'b')
    plt.plot(df.Date, signal, label='Signal Line', color='r')
    plt.legend(loc='upper left')
    #plt.show()
    plt.savefig('macd.jpg')
    plt.show()
    
    
    # In[121]:    
       
    
    
    # In[108]:
    
    
    #pd.set_option('max.rows',None)
    
      
    
    # In[122]:
    
    
    # plt.figure(figsize=(20,5))
    # horizontal = np.zeros(df.shape[0])
    # macd_norm = (df.macd-df.macd.mean())/df.macd.std()
    # price_norm = (df.Close-df.Close.mean())/(df.Close.std())
    # plt.plot(df.Date, macd_norm)
    # plt.plot(df.Date, horizontal)
    # plt.plot(df.Date, df.macd_dir)
    # plt.plot(df.Date, price_norm)
    
    
    # In[123]:
    
    
    #plt.figure(figsize=(20,5))
    #plt.plot(df.Date,df.macd)
    #plt.plot(df.Date, horizontal)
    
    
    # In[124]:
    
    
    #df.head(20)
    
    
    # In[125]:
    #Identify dates of macd reversals
    
    
    # In[5]:
    
    
    df['macd_lag'] = df.macd.shift(1)
    df['reversal'] = df.macd*df.macd_lag #if this value is negative, there is a reversal
    df.reversal = df.reversal<0
    df.reversal = df.reversal.astype(int)
    df.head(10)
    
    
    # In[6]:
    
    
    reversal_indices = df[df.reversal==1].index
    reversal_indices
    
    
    # In[7]:
    
    
    # if first signal is a sell, remove it
    if df.macd[reversal_indices[0]] < df.macd_lag[reversal_indices[0]]:
        reversal_indices = reversal_indices[1:]
    reversal_indices   
    
    
    # In[8]:
    
    
    reversal_indices.shape
    
    
    # In[9]:
    
    
    macd_reversals = df[df.index.isin(reversal_indices)].reset_index(drop=True)
    macd_reversals.head()
    
    
    # In[92]:
    
    
    macd_reversals.to_csv('macd_reversals.csv', index=False)
    
    
    # In[10]:
    
    
    for i in range(macd_reversals.shape[0]):
        if macd_reversals.macd[i]>0 and macd_reversals.macd_lag[i]<0:
            macd_reversals.reversal[i] = 1 # buy signal
        elif macd_reversals.macd[i]<0 and macd_reversals.macd_lag[i]>0:
            macd_reversals.reversal[i] = 0 #sell signal
    
    
    macd_reversals.head()
    
    
    # In[11]:
    
    
    plt.figure(figsize=(20,5))
    plt.plot(df.Date, macd, label='MACD', color = 'b')
    plt.plot(df.Date, signal, label='Signal Line', color='r')
    plt.legend(loc='upper left')
    #plt.show()
    plt.savefig('macd.jpg')
    plt.show()
    
    cmap = matplotlib.colors.ListedColormap(["red","green"], name='from_list', N=None)
    # if(alg=='rsi'):
    plt.figure(figsize=(20,5))
    plt.title('Buy(green)/sell(red) signals based on MACD')
    plt.plot(df.Date, df.Close)
    plt.scatter(macd_reversals.Date, macd_reversals.Close, c=macd_reversals.reversal, cmap=cmap)
    plt.savefig('buy_sell_rsi.png')
    plt.show()
    
    
    # In[12]:
    
    
    df.head()
    
    
    # In[13]:
    
    
    df.RSI
    
    
    # In[ ]:
    
    
    
    
    
    # In[14]:
    
    
    #df.to_csv('df.csv',index=False)
    
    
    # In[15]:
    
    
    reversal_indices.shape
    
    
    # In[16]:
    
    
    reversal_indices[0]
    
    
    # In[17]:
    
    
    df['buy_sell'] = 0
    df.head()
    
    
    # In[18]:
    
    
    j=0
    df.RSI[j] > 70 or df.Volume[j] > 20
    
    
    # In[81]:
    
    
    signals_macd_rsi = df.iloc[reversal_indices,:].reset_index(drop=True)
    signals_macd_rsi['buy_sell'] = 0
    signals_macd_rsi.loc[signals_macd_rsi.macd>signals_macd_rsi.macd_lag, 'buy_sell'] = 1   
    
    cmap = matplotlib.colors.ListedColormap(["red","green"], name='from_list', N=None)
    # if(alg=='rsi'):
    plt.figure(figsize=(20,5))
    plt.title('Buy(green)/sell(red) signals based on Algorithm')
    plt.plot(df.Date, df.Close)
    plt.scatter(signals_macd_rsi.Date, signals_macd_rsi.Close, c=signals_macd_rsi.buy_sell, cmap=cmap)
    plt.savefig('buy_sell_rsi.png')
    plt.show()
    
    
    # In[86]:
    
    
    #gain_loss_alg = 100*(np.prod(1+dates_bull.gain_loss/100)-1)  
    
    
    # In[87]:
    
    
    signals_macd_rsi = signals_macd_rsi[['Date','Close','Volume','vol_quantile','RSI','macd','macd_lag','buy_sell']]
    signals_macd_rsi['close_lag'] = signals_macd_rsi.Close.shift(1)
    signals_macd_rsi.head()
    
    
    # In[88]:
    
    
    signals_macd_rsi['gain_loss'] = (signals_macd_rsi.Close-signals_macd_rsi.close_lag)/signals_macd_rsi.close_lag
    signals_macd_rsi.loc[signals_macd_rsi.buy_sell==1,'gain_loss'] = 0
    
    signals_macd_rsi
    # In[89]:
    
    
    
    
    
    # In[90]:    
    gain_loss_alg = 100*(np.prod(1+signals_macd_rsi.gain_loss)-1)  
    gain_loss_alg
    
    gain_loss_ref = 100*(df.iloc[-1,:].Close - df.Close[0])/df.Close[0]
    #gain_loss_ref=0
    
    return df, signals_macd_rsi, gain_loss_alg, gain_loss_ref    
    
    
