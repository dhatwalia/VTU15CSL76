
# coding: utf-8

# In[57]:


from random import seed
import pandas as pd
import numpy as np


# In[58]:


def split_dataset(dataset,train_perc = 0.8):
    data_len = len(dataset)
    print("length of dataset"+str(data_len))
    train_index = int(data_len*train_perc)
    
    train = dataset[:train_index,:]
    
    test = dataset[train_index:,:]
    
    return(train,test)


# In[59]:


def sigmoid(activation):
    return 1.0/(1.0 + np.exp(-activation))


# In[60]:


def compute_loss(prediction,actual):
    return 0.5*np.sum((actual.T-prediction)*(actual.T-prediction))


# In[61]:


def back_prop(train_X,W1,W2,layer1_output,layer2_output,actual_output):
    difference = actual_output.T - layer2_output
    delta_output = layer2_output*(1-layer2_output)*difference
    delta_hidden = layer1_output*(1-layer1_output)*W2.T.dot(delta_output)
    deltaW2 = lr*(delta_output.dot(layer1_output.T)/n_train)
    deltaW1 = lr*(delta_hidden.dot(train_X)/n_train)
    
    return (deltaW1,deltaW2)


# In[62]:


def train_network(train_X,train_Y):
    n_input = train_X.shape[1]
    W1 = np.random.random((n_hidden,n_input))
    W2 = np.random.random((num_classes,n_hidden))
    
    for epoch in range(n_epoch):
        layer1_output = sigmoid(W1.dot(train_X.T))
        layer2_output = sigmoid(W2.dot(layer1_output))
        
        (deltaW1,deltaW2) = back_prop(train_X,W1,W2,layer1_output,layer2_output,train_Y)
        
        W2 = W2+deltaW2
        W1 = W1+deltaW1
        
        if epoch%100 == 0:
            loss = compute_loss(layer2_output,train_Y)
            print(str.format('loss in {0}th epoch is {1}',epoch,loss))
            
    return (W1,W2)


# In[63]:


def evaluate(test_X,test_y,params):
    (W1,W2) = params
    layer1_output = sigmoid(W1.dot(test_X.T))
    final = sigmoid(W2.dot(layer1_output))
    
    prediction = final.argmax(axis = 0)
    return np.sum(prediction == test_y)/len(test_y)


# In[64]:


def convert_to_OH(data,num_classes):
    
    one_hot = np.zeros((len(data),num_classes))  #ask here
    
    one_hot[np.arange(len(data)),data] = 1    #ask here
    print(one_hot)
    return one_hot


# In[66]:


filename = 'train4.csv'

df = pd.read_csv(filename,dtype = np.float64,header = None)
dataset = np.array(df)

(train,test) = split_dataset(dataset)
n_train = len(train)
print('ntrain = '+str(n_train))
n_test = len(test)
print('ntest = '+str(n_test))

lr = 1
n_epoch = 2000

num_classes = len(np.unique(dataset[:,-1]))
print('no of classes:'+str(num_classes))
train_one_hot = convert_to_OH(train[:,-1].astype(int),num_classes)  #ask here

n_hidden = 20

params = train_network(train[:,:-1],train_one_hot)
accuracy = evaluate(test[:,:-1],test[:,-1],params)*100
print('Mean Accuracy: %.3f%%'%accuracy)

