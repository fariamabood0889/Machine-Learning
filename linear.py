#!/usr/bin/env python
# coding: utf-8

# ## linearregression using gradient descent

# In[47]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# In[37]:


def plotPredictions(x,y,X,M,flag=False):
    ypred = []
    for i in range(len(x)):
        yi = float(X[0])
        for j in range(1,2):
            yi += float(X[j]) * (x[i] ** j)
        ypred.append(yi)
        
    # plot original curve and estimated curve
    
    if flag:
        plt.plot(x,ypred,color='red')
        plt.scatter(x,y,color='green')
        
    # calculate MSE
    mse = 0
    for i in range(len(x)):
        sqerr = (y[i] - ypred[i]) ** 2
        mse += sqerr

    mse = mse/len(x)
    return mse


# In[3]:


def optimizeParameters(A,B,X,M,lr):

    mult = np.dot(A,X)

    subt = np.subtract(mult,B)

    slope = subt * (2/M)

    change = np.dot(slope,lr)

    X = np.subtract(X,change)

    return X,np.min(change)


# In[4]:


def calculateSumMatrices(x,y,M):
    coeffArr = [M] # stores M Sum(X) Sum(X^2) ...
    highestPowerX = 2*M
    for i in range(1,highestPowerX+1):
        summ = 0
        for j in range(len(x)):
            summ += x[j] ** i
        coeffArr.append(summ)
        
    #print('coeffArr',coeffArr)
    
    coeffArrXY = [] # stores Sum(Y) Sum(X*Y) ...
    for i in range(M+1):
        summ = 0
        for j in range(len(x)):
            summ += y[j] * (x[j] ** i)
        coeffArrXY.append(summ)
        
    #print('coeffArrXY',coeffArrXY)
    
    k = 0 
    # creating matrix A & B to solve  "AX = B"
    
    A = [] 
    for i in range(M+1):
        k = i
        a = []
        for j in range(M+1):
            a.append(coeffArr[k])    
            k+=1
        A.append(a)
    A = np.array(A)
    
    #print('A',A)
    
    B = np.array(coeffArrXY)
    B = B.reshape((M+1,1))
    
    #print('B',B)
    
    X = np.random.rand(M+1,1)
    return A,B,X


# In[5]:


# generate input data to be fit

#x = np.linspace(0,4)
#y = np.sin(x) * 2 + np.random.normal(scale=0.1,size=len(x))

x = [i for i in range(1,11)]
y = [12,34,45,56,23,56,67,98,88,102]
#print(x)
#print(y)
#plt.plot(x,y)


# In[9]:


M = 1
lr = 0.00000025
A,B,X = calculateSumMatrices(x,y,M)
print(A,B,X,sep='\n')


# In[10]:


i = 1
mseArr = []
#mse = 0
step = 1
#print('The mean square error for M = %d is %.7f '%(M,mse))
while(abs(step)>0.000001 and i < 10000):
    
    #print('Iteration %d'%(i))
    
    X,step = optimizeParameters(A,B,X,M,lr)
    
    #print(X)
    #print('Step = %f '%(step))
    #time.sleep(1)
    
    mse = plotPredictions(x,y,X,M)
    mseArr.append(mse)
    
    #print('The mean square error for M = %d is %.7f '%(M,mse))
    
    i+=1

mse = plotPredictions(x,X,M,True)
print('The mean square error for M = %d is %.7f after %d iterations'%(M,mse,i))


# In[ ]:


plt.plot(range(1,i),mseArr)


# X = [[0.3509782 ]
#  [0.82707418]
#  [0.52975741]
#  [0.12360296]]
# 
# x = [0.         0.08163265 0.16326531 0.24489796 0.32653061 0.40816327
#  0.48979592 0.57142857 0.65306122 0.73469388 0.81632653 0.89795918
#  0.97959184 1.06122449 1.14285714 1.2244898  1.30612245 1.3877551
#  1.46938776 1.55102041 1.63265306 1.71428571 1.79591837 1.87755102
#  1.95918367 2.04081633 2.12244898 2.20408163 2.28571429 2.36734694
#  2.44897959 2.53061224 2.6122449  2.69387755 2.7755102  2.85714286
#  2.93877551 3.02040816 3.10204082 3.18367347 3.26530612 3.34693878
#  3.42857143 3.51020408 3.59183673 3.67346939 3.75510204 3.83673469
#  3.91836735 4.        ]
# y = [-0.08903725  0.23962717  0.04740177  0.40982837  0.82093915  0.81823549
#   0.95509846  1.06767663  1.06684443  1.27841417  1.52911016  1.49887695
#   1.65065241  1.82798101  1.69523876  2.00737336  1.98526324  1.83140126
#   2.10213821  1.89691495  1.98107874  1.92409138  2.03840921  1.83093469
#   1.84772476  1.88316445  1.90411325  1.71631624  1.40879329  1.30405746
#   1.37058593  1.01271808  1.02713292  0.81809013  0.7466398   0.69422469
#   0.24274845  0.08782341 -0.05827118 -0.08699947 -0.18671066 -0.53014852
#  -0.58620177 -0.8561655  -0.67307204 -0.82624796 -1.1131808  -1.41345815
#  -1.41108144 -1.26982713]
# 
# M = 3
# lr = 0.0000025

# In[73]:


def sample(x,y,testx,testy,splitratio,sampleratio):
    
    samplesize = int(len(x) * sampleratio)
    
    samplex = []
    sampley = []
    
    for i in range(samplesize):
        samplex.append(x[i])
        sampley.append(y[i])
        
    print(samplex,sampley)
    
    lentr = len(samplex)
    #lents = int(lentr*splitratio)
    lents = len(samplex)
    #testx = []
    #testy = []
    trainx = []
    trainy = []
    #for i in range(lents,lentr,1):
    #    testx.append(samplex[i])
    #    testy.append(sampley[i])
        
    for i in range(lents):
        trainx.append(samplex[i])
        trainy.append(sampley[i])
    
    print(trainx,trainy)
    print(testx,testy)
    
    M = 1 # linear regression
    lr = 0.00000025
    A,B,X = calculateSumMatrices(trainx,trainy,M)
    print(A,B,X,sep='\n')
    
    i = 1
    mseArr = []
    #mse = 0
    step = 1
    #print('The mean square error for M = %d is %.7f '%(M,mse))
    while(abs(step)>0.000001 and i < 10000):

        #print('Iteration %d'%(i))

        X,step = optimizeParameters(A,B,X,M,lr)

        #print(X)
        #print('Step = %f '%(step))
        #time.sleep(1)

        mse = plotPredictions(trainx,trainy,X,M)
        mseArr.append(mse)

        #print('The mean square error for M = %d is %.7f '%(M,mse))

        i+=1

    mse = plotPredictions(testx,testy,X,M,False)
    test_mse = mse
    return test_mse


# In[50]:


sample(x,y,0.7,0.9)


# In[52]:





# In[ ]:


mseArr = []
mse = 0
sampleArr = [0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
for i in sampleArr:
    mse = sample(x,y,testx,testy,0.7,i)
    mseArr.append(mse)
    
plt.plot(sampleArr,mseArr)


# In[64]:


data = pd.read_csv('Linear Regression train.csv')
print(data.head())


# In[65]:


x = data['Age(X)']
y=data['EstimatedSalary(Y)']


# In[66]:


x = list(x)
y = list(y)


# In[67]:


print(x,y)


# In[69]:


testdata = pd.read_csv('Linear Regression Test.csv')
testx = testdata['Age(X)']
testy=testdata['EstimatedSalary(Y)']
testx = list(testx)
testy = list(testy)
print(testx,testy)


# In[74]:


mseArr = []
mse = 0
sampleArr = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
for i in sampleArr:
    mse = sample(x,y,testx,testy,0.7,i)
    mseArr.append(mse)
    
plt.plot(sampleArr,mseArr)


# In[ ]:




