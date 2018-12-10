
# coding: utf-8

# In[28]:


import pandas as pd


# In[29]:


data=pd.read_csv("train1.csv",header=None)
print(data)


# In[30]:


train=data.sample(frac=0.5,random_state=None)
print(train)


# In[31]:


test=data.drop(train.index)
print(test)


# In[32]:


l=len(train.columns)
l=l-1 #because we are not taking the last column
h=pd.Series(['-']*l)
print(h)


# In[33]:


#r will return rows
for (i,r) in train.iterrows():
    print(r)
    if r.iloc[-1] == 1:
        for j in range(l):
            if h.iloc[j] == r.iloc[j]:
                continue
            else:
                if (h.iloc[j]=='-'):
                    h.iloc[j]=r.iloc[j]
                else:
                    h.iloc[j]='?'

print(h)


# In[34]:


#to test the data
def testh(h,sample):
    index=h!='?' #whatever values are not '?' are taken into h
    if (h[index]==sample[index]).all(): 
        return 1
    else:
        return 0


# In[35]:


#determines accuracy
def ac(testset):
    a=0
    for (i,r) in testset.iterrows():
        t=testh(h,r.iloc[:-1])
        if t==r.iloc[-1]:
            a=a+1
        print("\n Test: ",list(r))
        print("Predicted: ",t,"\tActual: ",r.iloc[-1])
        
    return (a/len(testset))*100


# In[36]:


print("Accuracy = ",ac(test),"%")

