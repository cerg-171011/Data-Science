#something_new
# coding: utf-8

# In[8]:


import numpy as np
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from sklearn.metrics import accuracy_score


# In[9]:


a = pd.read_csv(r'/home/hacktone/Рабочий стол/виндовс/test_python/titanic.csv')


# In[10]:


##  РАЗДЕЛЕНИЕ ЦЕЛЕВОГО ПРИЗНАКА ДАННЫХ (МЕТОК) ОТ ОСТАЛЬНЫХ ПРИЗНАКОВ.
##  МЕТКИ - КОНЕЧНЫЙ РЕЗУЛЬТАТ
x = a.drop('Survived', axis = 1)
y = a['Survived']


# In[20]:


##  РАЗДЕЛЕНИЕ ДАННЫХ НА ТРЕНИРОВОЧНЫЕ И ТЕСТОВЫЕ В ПРОПОРЦИИ 70%/30%
x_train, x_valid, y_train, y_valid = train_test_split(x,y, test_size = 0.3, random_state = 42)
x_train.shape, y_train.shape, x_valid.shape, y_valid.shape


# In[12]:


##  СОЗДАЕМ МОДЕЛЬ ОБУЧЕНИЯ
tree = DecisionTreeClassifier(random_state = 42)


# In[13]:


##  1. МЕТОД. ОБУЧЕНИЕ МОДЕЛИ НА ТЕСТОВЫХ ДАННЫХ, cv=5 - это кол-во эпох обучений 
cross_val_score(tree, x_train, y_train, cv = 5)


# In[21]:


##  2. МЕТОД (РЕШЕТЧАТЫЙ ПОИСК). СОЗДАНИЕ НОВОЙ МОДЕЛИ ОБУЧЕНИЯ КОТОРАЯ НАЙДЕТ НАМ НАИЛУЧШИЕ ПАРАМЕТРЫ ДЛЯ ОБУЧЕНИЯ КОТОРЫЕ МЫ ЗАДАЛИ В СЛОВАРЕ params
params = {'max_depth': np.arange(1,11), 'max_features': np.arange(0.25, 1, 0.25)}
grid_tree = GridSearchCV(tree, params, cv = 5, n_jobs = -1)

#  ПЕРЕДАЧА ДАННЫХ В НАШУ МОДЕЛЬ ДЛЯ ОБУЧЕНИЯ
grid_tree.fit(x_train, y_train)

##  СМОТРИМ ЛУЧШИЕ ДАННЫЕ ДЛЯ ОБУЧЕНИЯ
grid_tree.best_score_   ## - ЛУЧШИЙ ПРОЦЕНТ ОБУЧЕНИЯ
grid_tree.best_params_  ## - ЛУЧШИЕ ПАРАМЕТРЫ МОДЕЛИ
grid_tree.best_estimator_   ## - ЛУЧШАЯ МОДЕЛЬ ДЛЯ ОБУЧЕНИЯ


# In[15]:


##  ПОКАЗЫВАЕТ РЕЗУЛЬТАТ МАССИВОМ МЕТОК (0 - 1) ДЛЯ ТЕСТОВОГО НАБОРА 
grid_valid = grid_tree.predict(x_valid)

##  ПОКАЗЫВАЕТ ТОЧНОСТЬ ПРАВИЛЬНЫХ ОТВЕТОВ НАШЕЙ МОДЕЛИ ПУТЕМ СРАВНЕНИЯ ПОЛУЧЕННЫХ ДАННЫХ С ДАННЫМИ КОТОРЫЕ БЫЛИ    
accuracy_score(y_valid, grid_valid)

