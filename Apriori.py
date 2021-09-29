#!/usr/bin/env python
# coding: utf-8

# In[6]:


import csv


# In[8]:


Data_List=[]

with open ("SMSSpamCollection.txt",'r',errors='ignore') as df:
    df=csv.reader(df)
    #print(df)
    for i in df:
        Data_List.append(i)   #append the row of df into data_list
        
print(Data_List)


# In[ ]:


def createL1(db):                               # it will create a function  that contain 
                                                #The frequency of each itemset individually in the dataset 
    L1={}
    for tranid in db:
        tran = db[tranid]           
#         print(tran)
        for item in tran:
            #print(item)
            if item in L1:                             #increament the number if its present 
                L1[item]+=1
                #print(item,L1[item])
            else:                                              
                L1[item] = 1
    return L1

L1=createL1(db)
print(L1)


# In[15]:


def compare(itemset1,itemset2,sizeafterjoin):        #compare the itemset1 and itemset2 size after their join                                                 
    matchCount = 0 
    for item in itemset1:
        if item in itemset2:                         
            matchCount += 1
    return(matchCount >= (sizeafterjoin-2))
compare(['1','2'],['2','4'],3)


# In[16]:


def createNewitemset(itemset1,itemset2):
    for item in itemset1:
        if item not in itemset2:    #if item not in itemset2 the append the item in itemset2                     
            itemset2.append(item)
    itemset2.sort()                  #sort the itemset2
    print(itemset2)
    return ",".join(itemset2)              #join with comma seperation
createNewitemset(['1','2'],['2','4'])


# In[17]:


def createL(itemsetlist,db):
    L = {}                                        
    for itemset1 in itemsetlist:
        itemset2 = itemset1.split(",")              #make pair of  every item
        count = 0
        for tranid in db:                            # calculate the frequency  from the transaction table
            tran=db[tranid]
            flag=True
            for item in itemset2:
                if item not in tran:                    
                    flag = False
                    continue
            if(flag):
                count+=1
        L[itemset1]= count
    return L
createL(['1,2','1,3','2,3'],database)


# In[18]:


def join(c,db,k):
    print("C:",c,"k:",k)                                 #Join the frequent itemsets to form sets of size k + 1                                                                                             
    itemsetlist = [*c.keys()]                              
    print(itemsetlist)                                    
    itemsetlist.sort()
    print("Sorted itemset list:",itemsetlist)
    newitemsetlist=[]
    length=len(itemsetlist)
    print(length)
    
    for i in range(0,length):                                              
        startitemset=itemsetlist[i]
        startitemset1=startitemset.split(",")
        for j in range(i+1,length):
            nextitemset=itemsetlist[j]
            nextitemset1=nextitemset.split(",")
            if(compare(startitemset1,nextitemset1,k)):
                newitemset = createNewitemset(startitemset1,nextitemset1)
                if newitemset not in newitemsetlist:
                    newitemsetlist.append(newitemset)
                    
    for itemset10 in itemsetlist:                                             #and repeat the above sets until no
        itemset100 = itemset10.split(',')                                     # more itemsets can be formed.
        for itemset20 in itemsetlist:
            itemset200 = itemset20.split(",")
            
            if(compare(itemset100,itemset200,k)):
                newitemset = createNewitemset(itemset100,itemset200)
                
                if newitemset not in newitemsetlist:
                    newitemsetlist.append(newitemset)
    l=createL(newitemsetlist,db)
    return l
join({'1,3,2':2,'1,3,4':3},database,4)
    


# In[19]:


def prune(l,minSup):
    keysToDelete=[]         #prune the itemsets by deleting the element which is l ess than the  minsup
    for key in l:
        if(l[key]<minSup):
            keysToDelete.append(key)
    for key in keysToDelete:
        del(l[key])
    return l


# In[21]:


def apriori(data,L1,minSup):
    kTables={}
    k=2
    print("l1",L1)
    c = prune(L1,minSup)
    print("c1:",c)
    kTables[1] = c
    while(True):
        l = join(c,data,k)
        print("l"+str(k)+":",1)
        c=prune(l,minSup)
        print("c"+str(k)+":",c)
        if(len(c)==0):
            break
        kTables[k]=c
        k+=1
    print("\nFinal Answer:")
    print(kTables[k-1])
apriori(database,L1,2)


# In[ ]:





# In[ ]:




