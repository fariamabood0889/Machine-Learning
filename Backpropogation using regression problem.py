#!/usr/bin/env python
# coding: utf-8

# In[126]:


import numpy
import matplotlib.pyplot


# In[127]:


x1=numpy.random.rand()
x2=numpy.random.rand()


target = 0.7

learning_rate = 0.5

w1=numpy.random.rand()
w2=numpy.random.rand()



predicted_output = []
network_error = []
MSE = []


# In[128]:


def sigmoid(s):
    return 1.0/(1+numpy.exp(-1*s))

def error(predicted, target):
    return numpy.power(predicted-target, 2)

def error_predicted_deriv(predicted, target):
    return 2*(predicted-target)

def sigmoid_deriv(s):
    return sigmoid(s)*(1.0-sigmoid(s))

def w_deriv(x):
    return x


def update_w(w, grad, learning_rate):
    return w - learning_rate*grad


# In[129]:


for k in range(30): #epoch =1000
    # Forward Pass
    square_error =0.0
    mean_square_error =0.0
    
    y = w1*x1 + w2*x2
    predicted = sigmoid(y)
    err = error(predicted, target)
    
    predicted_output.append(predicted)
    network_error.append(err)

    # Backward Pass
    g1 = error_predicted_deriv(predicted, target)

    g2 = sigmoid_deriv(y)
    
    g3w1 = w_deriv(x1)
    g3w2 = w_deriv(x2)
    
    gradw1 = g3w1*g2*g1
    gradw2 = g3w2*g2*g1

    w1 = update_w(w1, gradw1, learning_rate)
    w2 = update_w(w2, gradw2, learning_rate)
    
    square_error = err* err
    mean_square_error = numpy.mean(square_error)
    MSE.append(mean_square_error)


# In[130]:


y = w1*x1 + w2*x2
predicted = sigmoid(y)
err = error(predicted, target)
    
predicted_output.append(predicted)
network_error.append(err)


# In[131]:


g1 = error_predicted_deriv(predicted, target)
g2 = sigmoid_deriv(y)
    
g3w1 = w_deriv(x1)
g3w2 = w_deriv(x2)
    
gradw1 = g3w1*g2*g1    #derivatives of the error wrt to the weights 
gradw2 = g3w2*g2*g1

w1 = update_w(w1, gradw1, learning_rate)   #weights are updated in w1 and w2
w2 = update_w(w2, gradw2, learning_rate)


# In[132]:


matplotlib.pyplot.figure()
matplotlib.pyplot.plot(MSE)
matplotlib.pyplot.title("Epochs vs MSE")
matplotlib.pyplot.xlabel("Epochs")
matplotlib.pyplot.ylabel("MSE")


# In[ ]:




