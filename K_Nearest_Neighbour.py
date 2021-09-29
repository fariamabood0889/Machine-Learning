#!/usr/bin/env python
# coding: utf-8

# In[25]:


arr=[]
#with open('data.csv') as file:
  #  data = file.read()
  #  temp = data.split('\n')
   # for i in temp:
    #    arr.append(i.split(','))
#print(arr)


# In[26]:


def dist(x1,x2):
    d = 0
    for i in range(len(x1)):
        d = d + (x1[i]-x2[i])**2
    return d ** 0.5


# In[27]:


def knn(x_train, test , y_train,k=5):
    
    dis = []
    
    for i in range(len(x_train)):
         print(test)
        
    d = dist(x_train[i],test)
        
    temp =[]
    temp.append(d)
    print(d)
    dis.append((d,y_train[i][0]))
    dis = sorted(dis)
    return dis
    


# In[28]:


x_train=[]
y_train=[]

for x in arr:
#     print(x)
    x_train.append(x[:3])
    y_train.append(x[3:])
    
for i in range(len(x_train)):
    for j in range(len(x_train[0])):
        x_train[i][j] = float(x_train[i][j])

# x_train.append([1,2,3])
# x_train.append([1,2,7])
print(x_train)
print(y_train)


# In[ ]:


res = knn(x_train,[1,2,3],y_train)
print(res)


# In[ ]:


sorted(res)


# In[ ]:


test = input()
print(test.split())
print(test)
test = test.split()
test1 =[]
for i in range(len(test)):
    test1.append(float(test[i]))
    
print(test1)


# In[ ]:


test = input()
test = test.split()
test1 = []
for i in range(len(test)):
    test1.append(float(test[i]))
k = int(input())
res = knn(x_train,test1,y_train)
res=sorted(res)
print(res[:k])


# In[ ]:




