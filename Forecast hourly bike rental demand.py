#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn


# In[3]:


train = pd.read_csv(r"C:\Users\Amisha\Desktop\train.csv")


# In[4]:


train.head()


# In[5]:


train.columns


# In[6]:


test = pd.read_csv(r"C:\Users\Amisha\Desktop\test.csv")


# In[7]:


test.head()


# In[8]:


test.columns


# In[9]:


## count variable is missing in test but is present in train so this is what the missing variable is


# In[10]:


train.dtypes


# In[11]:


test.dtypes


# In[12]:


## date time is the only non numerical data type in both test and train we will eventually have to get rid of it to be able to
## use scikit learn


# In[13]:


## now we have identified the target variable and the independent variables


# In[14]:


## now we will do univariate and bivariate analysis, univariate analysis can be skipped


# In[15]:


## univariate analysis of the target variable


# In[16]:


sn.distplot(train["count"])


# In[17]:


## right skewed


# In[18]:


## We can skip univariate analysis


# In[19]:


corr = train[["temp","atemp","casual","registered","humidity","windspeed","count"]].corr()


# In[20]:


corr


# In[21]:


## either of temp or atemp will have to be dropped


# In[22]:


## preparing for model 


# In[23]:


train.isnull().sum()


# In[24]:


test.isnull().sum()


# In[25]:


train.shape


# In[26]:


training = train[0:10000]


# In[27]:


validation = train[10001:12980]


# In[45]:


## dropping non numerical values and one of the two independent variables which are strongly correlated


# In[28]:


training = training.drop(['datetime','atemp'],axis=1)


# In[29]:


validation = validation.drop(['datetime','atemp'],axis=1)


# In[30]:


train = train.drop(['datetime','atemp'],axis=1)


# In[31]:


test = test.drop(['datetime','atemp'],axis=1)


# #  MODEL BUILDING

# ### LINEAR REGRESSION

# In[32]:


from sklearn.linear_model import LinearRegression


# In[33]:


lModel = LinearRegression()


# In[37]:


x_train = training.drop('count', 1)
y_train = training['count']
x_val = validation.drop('count', 1)
y_val = validation['count']


# In[38]:


x_train.shape


# In[40]:


x_val.shape


# In[41]:


lModel.fit(x_train,y_train)


# In[43]:


prediction = lModel.predict(x_val)


# In[44]:


# defining a function which will return the rmsle score
def rmsle(y, y_):
    log1 = np.nan_to_num(np.array([np.log(v + 1) for v in y]))
    log2 = np.nan_to_num(np.array([np.log(v + 1) for v in y_]))
    calc = (log1 - log2) ** 2
    return np.sqrt(np.mean(calc))


# In[45]:


rmsle(y_val,prediction)


# In[62]:


test_prediction1 = lModel.predict(test)


# In[65]:


submission1 = pd.DataFrame()


# In[67]:


submission1['count'] = test_prediction1


# In[81]:


submission1.to_csv('submission1.csv', header=True, index=False)


# In[59]:


## let us see if we can find an algorithm with a smaller rmsle value ## LR seems to be an overfit


# ### DECION TREE MODEL

# In[49]:


from sklearn.tree import DecisionTreeRegressor


# In[52]:


dectree = DecisionTreeRegressor(max_depth=7)


# In[54]:


dectree.fit(x_train,y_train)


# In[55]:


dectree.fit(x_val,y_val)


# In[57]:


predict2=dectree.predict(x_val)


# In[60]:


rmsle(y_val, predict2)


# In[63]:


test_prediction2 = dectree.predict(test)


# In[72]:


submission2 = pd.DataFrame()


# In[73]:


submission2['count'] = test_prediction2


# In[82]:


submission2.to_csv('submission2.csv', header=True, index=False)


# In[83]:


submission2.head()


# In[84]:


submission1.head()


# In[ ]:




