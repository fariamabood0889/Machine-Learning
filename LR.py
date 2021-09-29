#!/usr/bin/env python
# coding: utf-8

# In[22]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[28]:


data=pd.read_csv('data.csv',names = ['x','y'])
data


# In[29]:


data.shape


# In[31]:


x = np.array(data.x)   # stored in the form of array and name the col as x
y = np.array(data.y)
x,y


# In[33]:


plt.scatter(x,y)
plt.xlabel("year Experience")
plt.ylabel("Salary")
plt.show


# In[39]:


x_mean=np.mean(x)
y_mean=np.mean(y)
print("Mean for Value X:",x_mean,"\n\nMean for Value Y:",y_mean)


# To Calculate Value of Coefficient and intercept

# In[53]:


numr=0
denr=0
for i in range(len(x)):
        numr+=(x[i]-x_mean)*(y[i]-y_mean)
        denr+=(x[i]-x_mean)**2
b1=numr/denr       
b0=y_mean-(b1*x_mean)
print("Intercept value is:",b0,"\n\nCoeficient is",b1)


# In[54]:


#predicted values for y
y_pred=b0+b1*x
y_pred
#for each value of x


# In[61]:


plt.scatter(x,y,c="red")
plt.xlabel("year Experience")
plt.ylabel("Salary")
plt.plot(x,y_pred,c="blue")
plt.show


# In[63]:


num_r=0
den_r=0
for i in range(len(x)):
        num_r+=(y[i]-y_pred[i])**2
        den_r+=(y[i]-y_mean)**2
r_sqr=1-(num_r/denr)
print("The Value of Coefficient of determination R_Square is",r_sqr)


# In[ ]:




