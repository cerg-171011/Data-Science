
# coding: utf-8

# In[33]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import ExtraTreesClassifier
from pandas.plotting import scatter_matrix


# In[34]:


a = pd.read_csv('/home/hacktone/Рабочий стол/виндовс/test_python/housing.csv')


# In[35]:


a.head(), a.info(), a.shape


# In[36]:


a[~pd.notna(a['total_bedrooms'])]


# In[37]:


b = a.index[~pd.notna(a['total_bedrooms'])]
b.values


# In[38]:


median = a['total_bedrooms'].median()
median


# In[39]:


a['total_bedrooms'].fillna(median, inplace = True)
a.info()


# In[40]:


corr_matrix = a.corr()
corr_matrix


# In[43]:


scatter_matrix(['longitude, latitude', 'housing_median_age'])

