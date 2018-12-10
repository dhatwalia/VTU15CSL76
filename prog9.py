
# coding: utf-8

# In[ ]:


#load the iris dataset as an example
from sklearn.datasets import load_iris
import numpy as np
iris=load_iris()
print(iris)


# In[ ]:


print(iris.DESCR)


# In[ ]:


#store the feature matrix(X) and response vector (y)
X=iris.data
y=iris.target

print("X ",X[:5])
print("y ",y[:5])


# In[47]:


#splitting X and y into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.4,random_state=1)

print(iris.data.shape)
print(len(X_train))
print(len(y_test))


# In[10]:


#training the model on training set
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3) #should be odd, usually equal to no. of class labels
knn.fit(X_train,y_train)


# In[11]:


#making predictions on the testing set
y_pred=knn.predict(X_test)
print(y_pred)


# In[13]:


#comparing actual response values (y_test) with predicted response values (y_pred)
from sklearn import metrics
print("Accuracy = ",metrics.accuracy_score(y_test,y_pred))


# In[46]:


#printing output of entire test_dataset
preds=knn.predict(X_test)
pred_species=[iris.target_names[p] for p in preds]
pred_species1=[iris.target_names[p] for p in y_test]

print("\tPredicted","\t\tActual","\t\tAnswer")
for i in range(0,len(pred_species)):
    print(i,":",pred_species[i],"\t\t",pred_species1[i],"\t\t",end='\t')    
    if(pred_species[i]==pred_species1[i]):
        print("Correct")
    else:
        print("Incorrect!!!")


# In[43]:


#checking for sample: wrong prediction
sample=[X_test[42]]
print(sample)
ps=knn.predict(sample)
print(iris.target_names[ps])

