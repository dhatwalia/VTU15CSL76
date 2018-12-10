
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# In[2]:


tou = 0.5
X_train = np.array(list(range(3,33)) + [3.2,4.2])
X_train


# In[3]:


X_train = X_train[:,np.newaxis]
X_train


# In[4]:


y_train = np.array([1,2,1,2,1,1,3,4,5,4,5,6,5,6,7,8,9,10,11,11,12,11,11,12,13,16,17,19,18,34,21,22])
X_test = np.array([i/10. for i in range(400)])

X_test = X_test[:,np.newaxis]
y_test = []


# In[5]:


count = 0
for r in range(len(X_test)):
    try:
        wts = np.exp(-np.sum((X_train - X_test[r])**2,axis = 1)/(2*tou)**2)
        W = np.diag(wts)
        factor1 = np.linalg.inv(X_train.T.dot(W).dot(X_train))
        parameters = factor1.dot(X_train.T).dot(W).dot(y_train)
        prediction = X_test[r].dot(parameters)
        y_test.append(prediction)
        count = count + 1
    except:
        pass
    
y_test = np.array(y_test)
plt.plot(X_train.squeeze(),y_train,'o')
plt.plot(X_test.squeeze(),y_test,'*')
plt.show()
y_test

