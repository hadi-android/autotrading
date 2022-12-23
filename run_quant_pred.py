#!/usr/bin/env python
# coding: utf-8

# <h3> Import required libraries

# In[7]:


import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from yahoo_historical import Fetcher
from quant_predict import quant_pred
import datetime
import time


# <h3> Download QQQ price data from 2018-12-1 to 2020-9-9

# In[12]:


ticker = 'ETH-CAD'
dstart = '1-1-2022'

dstart = time.mktime(datetime.datetime(2022,1,1).timetuple())
df = Fetcher(ticker, dstart).get_historical()
df.Date = pd.to_datetime(df.Date)
# df = df[:-1]
df.head()


# <h3> compute trade signals and returns on investment and set the signal source to macd

# In[13]:


df.tail()


# In[14]:


start = '2022-1-1'
end = '2022-12-23'
alg = 'macd'
# signals, signal2, gl_alg, gl_ref = quant_pred(df,alg, start, end)


# <h3> Return (% gains) using the signals

# In[15]:


# print(round(gl_alg))


# # <h3>Return without using the singnals (buy at start, sell at end)

# # In[16]:


# print(round(gl_ref))


# # In[17]:


# signals


# # In[18]:


# 100*(np.prod(1+signals.gain_loss/100)-1)


# # In[21]:


# winrate = 100*len(signals[signals.gain_loss>0])/len(signals)
# winrate


# In[ ]:





# In[23]:


start = '2022-1-1'
end = '2022-12-23'
alg = 'rsi'
signals, signal2, gl_alg, gl_ref = quant_pred(df,alg, start, end)


# In[ ]:




