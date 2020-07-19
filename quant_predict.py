
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

#ticker = 'msft'
#start = [2019,1,1]
#
#df = Fetcher(ticker, start).getHistorical()	
##df = df[['Date','Close']]
#df.Date = pd.to_datetime(df.Date)
#df.head()
#

# In[106]:

def quant_pred(df, alg):
    
    df.Date = pd.to_datetime(df.Date)


    exp1 = df.Close.ewm(span=12, adjust=False).mean()
    exp2 = df.Close.ewm(span=26, adjust=False).mean()
    
    #df['macd'] = exp1-exp2
    #df['signal'] = macd.ewm(span=9, adjust=False).mean()
    macd = exp1-exp2
    signal = macd.ewm(span=9, adjust=False).mean()
    df['macd'] = macd-signal
    
    print(signal.shape)
    print(macd.shape)
    
    
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
    
    df['macd_dir'] = df.macd>0
    df.macd_dir = df.macd_dir.astype(int)
    df['macd_chg'] = df.macd-df.macd.shift(1)
    df['macd_chg_abs'] = abs(df['macd_chg'])
    df['macd_chg_pct'] = (df.macd-df.macd.shift(1))/df.macd.shift(1)
    df['macd_chg_dir'] = df.macd_chg_pct>0
    df.macd_chg_dir = df.macd_chg_dir.astype(int)
    df['macd_chg_pct_abs'] = abs(df.macd_chg_pct)
    df=df[['Date','Close','macd','macd_dir','macd_chg','macd_chg_abs','macd_chg_pct','macd_chg_pct_abs','macd_chg_dir']]
    df.head(10)
    
    
   
    
    # In[122]:
    
    
    plt.figure(figsize=(20,5))
    horizontal = np.zeros(df.shape[0])
    macd_norm = (df.macd-df.macd.mean())/df.macd.std()
    price_norm = (df.Close-df.Close.mean())/(df.Close.std())
    plt.plot(df.Date, macd_norm)
    plt.plot(df.Date, horizontal)
    plt.plot(df.Date, df.macd_dir)
    plt.plot(df.Date, price_norm)
    
    
    # In[123]:
    
    
    #plt.figure(figsize=(20,5))
    #plt.plot(df.Date,df.macd)
    #plt.plot(df.Date, horizontal)
    
    
    # In[124]:
    
    
    #df.head(20)
    
    
    # In[125]:
    
      
    #RSI
    df['chg'] = df.Close-df.Close.shift(1)
    N=14
    df['RSI']=np.zeros(df.shape[0])
    i=0
    while((i+N) < df.shape[0]):
        chg = df.chg[i:i+N]
        pos_chg = chg[chg>0].sum()/N
        neg_chg = abs(chg[chg<0].sum())/N
        RS = pos_chg/neg_chg    
        df['RSI'][i+N] = 100-(100/(1+RS))
        i+=1
    
    df.RSI.min()
    df.RSI.max()   
    
    thresh = df.copy()
    thresh['lower'] = 30
    thresh['upper'] = 75
    
    df_nobuy = df[df.RSI>75]
    
    plt.figure(figsize=(20,5))
    plt.plot(df.Date, df.RSI)
    plt.plot(thresh.Date, thresh.lower)
    plt.plot(thresh.Date, thresh.upper)
    plt.show()
       

    # df['buy_rsi'] = df.RSI <=70
    df['buy_rsi'] = df.RSI>=50
    df.buy_rsi = df.buy_rsi.astype(int)
    df.head(10)    
    
        
    state=0
    df_sig = []
    for i in range(df.shape[0]):
        if((df.buy_rsi[i]==1) & (state==0)):
            state = 1
            df_sig.append(df.iloc[i,:].to_frame().T)
        if((df.buy_rsi[i]==0) & (state==1)):
            state=0
            df_sig.append(df.iloc[i].to_frame().T)
            
    df_sig = pd.concat(df_sig, axis=0)
    df_sig = df_sig.reset_index(drop=True)
    
    df_sig['gainloss'] =100*(df_sig.Close-df_sig.Close.shift(1))/df_sig.Close.shift(1)
    df_sig1 = df_sig[df_sig.buy_rsi==0]
    
    
    gainloss_rsi = 100*(np.prod(1+df_sig1.gainloss/100)-1)    

#macd    
    bear=0
    bull=0
    df['bull_period'] = np.empty(df.shape[0])
    df.bull_period = np.nan
    df['bear_period'] = np.empty(df.shape[0])
    df.bear_period = np.nan
    dates_bull=[]
    dates_bear=[]
    tmp=[]
    tmp2=[]
    for i in range(1,df.shape[0]):
        if((df.macd[i]>0) & (df.macd[i-1]>0)):
            if(bull==0):
                tmp.append(df.Date[i-1])
            bull+=1
        if((bull>0) & (df.macd[i]<0)):
            df.bull_period[i-1] = bull+1
            tmp.append(df.Date[i-1])
            dates_bull.append(tmp)
            bull=0
            tmp=[]
        if((i==df.shape[0]-1) & (bull>0)):
            tmp.append(df.Date[i])
            dates_bull.append(tmp)
            df.bull_period[i-1] = bull+1
            
            
        if((df.macd[i]<0) & (df.macd[i-1]<0)):
            if(bear==0):
                tmp2.append(df.Date[i-1])
            bear+=1
        if((bear>0) & (df.macd[i]>0)):
            df.bear_period[i-1] = bear+1
            tmp2.append(df.Date[i-1])
            dates_bear.append(tmp2)
            bear=0  
            tmp2=[]
        if((i==df.shape[0]-1) & (bear>0)):
            tmp2.append(df.Date[i])
            dates_bear.append(tmp2)
            df.bear_period[i-1] = bear+1             
    
            
        
    bull_periods = df.bull_period[df.bull_period.isnull()==False]
    len(bull_periods)
    bull_periods.reset_index(drop=True)  
       

    dates_bull = pd.DataFrame(dates_bull, columns=['Start','End'])
    dates_bull = dates_bull.assign(len = bull_periods.values)
    
    dates_bull['macd_max'] = np.zeros(dates_bull.shape[0])
    dates_bull['chg_max'] =  np.zeros(dates_bull.shape[0])
    dates_bull['chg_pct_max'] = np.zeros(dates_bull.shape[0])
    dates_bull['close_start'] = np.zeros(dates_bull.shape[0])
    dates_bull['close_end'] = np.zeros(dates_bull.shape[0])
    dates_bull['gain_loss'] = np.zeros(dates_bull.shape[0])
    macd_act = []
    for i in range(dates_bull.shape[0]):
        macd_act.append(df[df.Date==dates_bull.Start[i]])
        macd_act.append(df[df.Date==dates_bull.End[i]])        
        subset = df[(df.Date>=dates_bull.Start[i]) & (df.Date<=dates_bull.End[i])]
        dates_bull.macd_max[i] = subset.macd.max()
        dates_bull.chg_max[i] = subset.macd_chg.max()
        dates_bull.chg_pct_max[i] = subset.macd_chg_pct.max()
        dates_bull.close_start[i] = subset.Close.head(1)
        dates_bull.close_end[i] = subset.Close.tail(1)
        dates_bull.gain_loss[i] = 100*(float(subset.Close.tail(1))-float(subset.Close.head(1)))/float(subset.Close.head(1))
    
    macd_act = pd.concat(macd_act, axis=0)
    
    plt.figure(figsize=(20,5))
    plt.plot(df.Date, df.Close)
    plt.plot(macd_act.Date, macd_act.Close, 'o')
    plt.savefig('buy_sell_points.png')
    plt.show()

    #combination   
         
            
    if (alg=='macd'):
        gain_loss_alg = 100*(np.prod(1+dates_bull.gain_loss/100)-1)  
        signals = dates_bull
    elif (alg=='rsi'):
        gain_loss_alg = gainloss_rsi
        signals = df_sig
        
    gain_loss_ref = 100*(float(df.Close.tail(1))-float(df.Close.head(1)))/float(df.Close.head(1))    

    
    return (signals, gain_loss_alg, gain_loss_ref)


# In[ ]:




