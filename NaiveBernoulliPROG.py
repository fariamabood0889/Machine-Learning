#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
dataset = [["Chinese Beijing Chinese", 0],
           ["Chinese Chinese Chinese Shanghai", 0],
           ["Chinese Macao", 0],
           ["Tokyo Japan Chinese", 1]]
  
dataset = pd.DataFrame(dataset)
dataset.columns = [["Words", "Class"]]
print(dataset)


# In[2]:



ss=[] #Empty list creation and splitting of words
for i in dataset.iloc[:,0].values:
    x=list(i.split(" "))
    ss=ss+x
print(ss)


# In[3]:


s2=list(dataset.iloc[:,1].values)
print(s2)
c1=0
for i in s2:
    if i==0:
        c1=c1+1
print("class 0:",c1)  


c2=0
for i in s2:
    if i==1:
        c2=c2+1
print("class 1:",c2) 


# In[4]:


p1=c1/len(s2) %prior probability 
p2=c2/len(s2)
p1


# In[5]:


p2


# In[6]:


s= list(dataset.iloc[:,0].values)
snew1=[]  #distinct for class 0
for i in range (0,4):
    if (s2[i]==0):
        c11=list(s[i].split(" "))
        snew1=snew1+c11
print("class0:",snew1)
l1=len(snew1)
print(l1)


# In[7]:


s= list(dataset.iloc[:,0].values)
snew2=[]
for i in range (0,4):
    if (s2[i]==1):
        c11=list(s[i].split(" "))
        snew2=snew2+c11
print("class1:",snew2)
l2=len(snew2)
print(l2)


# In[8]:


#unique list konse word hai list me
def unique(list1):   
    ulist1=[] 
    for x in list1:
        if x not in ulist1:
            ulist1.append(x)
    return ulist1
u=unique(ss)
print(u)


# In[9]:


cn0=[]   #occurrance of word in class 0
for i in u:
    a=0
    a=a+snew1.count(i)
    cn0.append(a)
print(cn0)


# In[10]:


cn1=[] 
for i in u:
    a=0
    a=a+snew2.count(i)
    cn1.append(a)
print(cn1)


# In[11]:


ulen=len(u)
ulen


# In[12]:


prob_0=[]
temp=0
for i in range(0,ulen):
    temp=(cn0[i]+1)/(l1+2)  #+1 laplace smoothing avoiding prob  0,+2 binary(Vocab size) for likelihood of w in givn class
    prob_0.append(temp)
print(prob_0)


# In[13]:


prob_1=[]
temp=0
for i in range(0,ulen):
    temp=(cn1[i]+1)/(l2+2)
    prob_1.append(temp)
print(prob_1)


# In[14]:


test=['Chinese','Shanghai']
print("test data:",test)
usnew1=unique(snew1)
usnew2=unique(snew2)
print("unique word in class 0;",usnew1)
print("unique word in class 1;",usnew2)


# In[15]:


p=[]
np=[]
for i in test:
    if i in usnew1:
        p.append(i)
    else:
        np.append(i)
print(p)
print(np)
test_prob_0=p1
print(test_prob_0)


# In[16]:


for i in range(0,len(p)):
    j=u.index(p[i])
    test_prob_0=test_prob_0*(prob_0[j])
print("probability class 0:",test_prob_0)


# In[17]:


for i in range(0,len(np)):
    j=u.index(np[i])
    test_prob_0=test_prob_0*(1-prob_0[j])
print("probability class 0:",test_prob_0)


# In[18]:


pp=[]
npp=[]
for i in test:
    if i in usnew2:
        pp.append(i)
    else:
        npp.append(i)
print(pp)
print(npp)
test_prob_1=p2
print(test_prob_1)
for i in range(0,len(pp)):
    j=u.index(pp[i])
    test_prob_1=(test_prob_1)*(prob_1[j])
for i in range(0,len(npp)):
    j=u.index(npp[i])
    test_prob_1=(test_prob_1)*(1-prob_1[j])
print(test_prob_1)    
    
    


# In[19]:


if test_prob_1<test_prob_0:
    print("this text belong to class 0")
else:
    print("this text belong to class 1") 


# In[ ]:





# In[ ]:




