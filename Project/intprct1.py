#!/usr/bin/env python
# coding: utf-8

# In[18]:


get_ipython().system('pip install matplotlib')
get_ipython().system('pip install pandas')


# In[5]:


import matplotlib.pyplot as plt


# In[7]:


import numpy as np
xpts=np.array([0,4])
ypts=np.array([0,100])
plt.plot(xpts,ypts)
plt.show()


# In[6]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from statsmodels.tsa.stattools import adfuller


# In[9]:


df=pd.read_csv('C:\\Users\\alsaf\\Downloads\\dmart.csv')
 print(df.head())


# In[38]:


df.iloc[:,[2,3,4]]


# In[39]:


df['Date']=pd.to_datetime(df['Date'])
print(df.head())
print(df.tail())
df.set_index(['Date'],inplace=True)  #we need data based on Month so we are making it as an index for easy use 
print(df.head())

df.plot()


# In[58]:


ts= df['Weekly_Sales']
ts.rolling(12).mean().plot()


# In[59]:


ts.rolling(12).std().plot()


# In[ ]:


pvalue=adfuller(ts)
print(pvalue)
# # shifting coz pvalue is greater than 0.02 
df['Weekly_Sales']=df['Weekly_Sales'].rolling(12).mean().shift(4)
print(df['Weekly_Sales'])


# In[ ]:




