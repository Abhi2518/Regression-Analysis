#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


# In[2]:


df = pd.read_csv("C:\\Users\\Abhishek\\Downloads\\china_gdp.csv")
df.head()


# In[3]:


x = df.drop(['GDP value of China'],axis=1).values
y = df['GDP value of China'].values


# In[4]:


print(x)


# In[5]:


print(y)


# In[6]:


plt.scatter(x,y)
plt.xlabel('Year')
plt.ylabel('GDP value of China')
plt.title('Non linear data of GDP of china year wise')


# In[7]:


poly_features = PolynomialFeatures(degree = 5, include_bias = False)
x_poly = poly_features.fit_transform(x)
x_poly


# In[8]:


lin_reg = LinearRegression()
lin_reg.fit(x_poly, y)
print('Coefficients of x are', lin_reg.coef_)
print('Intercept is', lin_reg.intercept_)


# In[9]:


y_pred = lin_reg.predict(x_poly)
plt.scatter(x,y,color='blue')
plt.scatter(x,y_pred,color='red')
plt.xlabel('year')
plt.ylabel('GDP of China')
plt.title('non linear GDP plot of china')


# In[10]:


from sklearn.metrics import r2_score
r2_score(y,y_pred)

