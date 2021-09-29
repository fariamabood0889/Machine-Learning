#!/usr/bin/env python
# coding: utf-8

# In[25]:


import numpy as np
import matplotlib.pyplot as plt


# In[26]:


x= np.arange(0,2*np.pi,0.3) #signpi 
y =np.sin(x)
e=np.random.normal(0,0.2,len(x))
y=y+e
n=len(x)
print(x)
print(y)


# In[27]:


plt.scatter(x,y)
plt.show()


# In[28]:


d=6


# In[29]:


ta=[]
for i in range(1,2*d+1):
    s=0
    for j in range(0,len(x)):
        s=s+x[j]**i
    ta.append(s)
print(ta)
ar=[]
temp=[n]
for i in range(0,d):
    temp.append(ta[i])
ar.append(temp)
t=[]
vt=0
for i in range (0,d):
    temp=[]
    vte=vt
    for j in range(0,d+1):
        temp.append(ta[vte])
        vte=vte+1
    ar.append(temp)
    vt=vt+1
    
A=ar
B=[]
for i in range(0,d+1):
        s=0
        for j in range(0,len(x)):
            s=s+(x[j]**i)*y[j]
        B.append(s)
print("A matrix is: ")
print(A,"\n")
print("B matrix is: ")
print(B)


# In[30]:


c = np.linalg.solve(A, B)
print(c)


# In[31]:


y_predicted = np.zeros(len(x))
for i in range(len(x)):
    for j in range(len(c)):
        y_predicted[i] +=c[j] * (x[i]**j)
print(y_predicted)


# In[32]:


plt.scatter(x,y)
plt.plot(x,y_predicted,color='r')
plt.show()


# In[33]:


def grad(x,c,d):
    lr=0.020
    ep=15
    ec=[]
    err1=err(y,y_predicted)
    for i in range(0,d+1):
        for j in range(0,ep):
            t=c[i]-((-2/n)*lr)*np.sum((x**1)*err1) #meaan square error ko diff kiyahai
        ec.append(t)
    return ec
def err(y,y_predicted):
    error=[]
    error.append(y-y_predicted)
    return error


# In[34]:


ec=grad(x,c,d)
print(ec)
y_predicted1 = np.zeros(len(x))
for i in range(len(x)):
    for j in range(len(ec)):
        y_predicted1[i] += ec[j] * (x[i]**j)
plt.scatter(x,y)
plt.plot(x,y_predicted1,color='r')
plt.show()


# In[35]:


def mse(y,y_cap):
    error = []
    se=[]
    for i in range(0,len(y)):
        error.append((y[i]-y_cap[i]))
        se.append((error[i])**2)
    mse=np.sum(se)/len(y)
    return mse
print("mse before gradient descent : " ,mse(y,y_predicted))
print("mse after gradient descent: ",mse(y,y_predicted1))


# In[ ]:




