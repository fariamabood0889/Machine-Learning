#!/usr/bin/env python
# coding: utf-8

# In[12]:


import numpy as np
import matplotlib.pyplot as plt


# In[13]:


X = np.array([.50,1.50,2.00,4.25,3.25,5.15,5.30,6.45,6.70], ndmin=2).reshape((9,1))   
y = np.array([0,0,0,0,1,1,1,1,1])     
X_mean = np.mean(X)
y_mean = np.mean(y)         
n = len(X)
plt.scatter(X,y)
plt.show()


# In[14]:


class LogisticRegression:
    b0 = 0    
    b1 = 0
    sigmoid = np.array([])
    loss = np.array([])
    epoch = 100
    alpha = 0.001

    def fit(self, X, y):
        upward_function = 0
        downward_function = 0
        for i in range(n):
            upward_function += (X[i]-X_mean)*(y[i]-y_mean)
            downward_function += (X[i]-X_mean)**2
        self.b1 = upward_function / downward_function
        self.b0 = y_mean - (self.b1*X_mean)
        return self.b0, self.b1
    
    
    def predict(self, Xi):
        z = self.b0 + (self.b1*Xi)
        self.sigmoid = np.append(self.sigmoid, [1/(1 + np.exp(-z))])
        return self.sigmoid
    
    def log_loss(self):
        for i in range(n):
            self.loss = np.append(self.loss, [-y[i]*np.log(self.sigmoid[i])-(1-y[i])*np.log(1-self.sigmoid[i])])
        print(self.loss)
        log_loss = np.mean(self.loss)
        return log_loss, self.sigmoid
    def loss_optimization(self, Xi):
        for i in range(self.epoch):
            dertivate_sigmoid = (1/n)*np.sum(self.loss*Xi)
            self.b0 = self.b0 - self.alpha*dertivate_sigmoid
            self.b1 = self.b1 - self.alpha*dertivate_sigmoid
        print('Constant : ',self.b0,'\nRegresssion Coeff : ',self.b1)
        
model = LogisticRegression()
print(model.fit(X, y))
print(model.predict(X))
print(model.log_loss())
model.loss_optimization(X)
model.log_loss()


# In[15]:


def mserror(y,y_cap):
    error=[]
    se=[]
    for i in range(0,len(y)):
        error.append(y[i]-y_cap[i])
        se.append(error[i]**2)
        mse= np.sum(se)/len(y)
        
    return mse


# In[16]:


def error(y,y_cap):
    error=[]
    for i in range(0,n):
        error.append(y[i]-y_cap[i])
    return error


# In[17]:


plt.scatter(X,y)
plt.plot(X,y,color='red')
plt.show()


# In[ ]:





# In[ ]:




