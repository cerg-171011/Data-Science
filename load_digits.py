
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[27]:


from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import RidgeCV
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.datasets import load_digits


# In[29]:


im = load_digits()
data = im.data
target = im.target
data.shape, target.shape


# In[60]:


x_test, y_test, x_train, y_train = data[:-10], target[:-10], data[-10:], target[-10:]
x_test.shape, y_test.shape, x_train.shape, y_test.shape


# In[33]:


ESTIMATORS = {
    'SVC': SVC(gamma = 0.0001),
    'Tree': DecisionTreeClassifier(),
    'Neighbor': KNeighborsClassifier(),
    'Ridge': RidgeCV(),
    'tree_regressor': ExtraTreesRegressor()
}


# In[48]:


name_dict = []
for name, est in ESTIMATORS.items():
    est.fit(x_test, y_test)
    name_dict.append(name)
y_pred = dict()
for name, est in ESTIMATORS.items():
    y_pred[name] = est.predict(x_train)


# In[49]:


data_pd = pd.DataFrame(y_pred, columns = name_dict)
data_pd.head()


# In[77]:


for i in range(5):
    if i:
        sub = plt.subplot(5,5, i * 5 +1, title = y_train[i])
        sub.axis('off')
        sub.imshow(x_train[i].reshape(8,8), cmap = plt.cm.gray)
    else:
        sub = plt.subplot(5,5, i * 5 +1, title = 'true_digit')
        sub.axis('off')
        sub.imshow(x_train[i].reshape(8,8), cmap = plt.cm.gray)
        for j, est in enumerate(ESTIMATORS):
            if i:
                sub = plt.subplot(5,5,i*5+2+j)
                sub.imshow(x_train[i], y_pred[est][i])

