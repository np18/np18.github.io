
# coding: utf-8

# In[1]:

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[2]:

get_ipython().magic('matplotlib inline')


# In[3]:

print('Python version ' + sys.version)
print('Pandas version ' + pd.__version__)
print('Numpy version ' + np.__version__)


# In[4]:

print("Test Notebook")


# In[5]:

x = np.arange(0, 5, 0.1);
plt.plot(x, np.sin(x));


# In[6]:

df = pd.DataFrame({'c1':[1,2,3], 'c2':[4,5,6]})
df


# In[7]:

# <!-- collapse=True --> 
print('Testing Collapsible code')

