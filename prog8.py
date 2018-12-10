
# coding: utf-8

# In[43]:


import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans
import sklearn.metrics as sm
import pandas as pd
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')


# In[44]:


# import some data to play with
iris = datasets.load_iris()

#print("\n IRIS DATA :",iris.data)
#pint("\n IRIS FEATURES :\n",iris.feature_names)
#print("\n IRIS TARGET :\n",iris.target)
#print("\n IRIS TARGET NAMES:\n",iris.target_names)

# Store the inputs as a Pandas Dataframe and set the column names
X = pd.DataFrame(iris.data)

#print(X)
X.columns = ['Sepal_Length','Sepal_Width','Petal_Length','Petal_Width']
#print(X.column)
#print("X:",x)
#print("Y:",y)

y = pd.DataFrame(iris.target)
#print(y)
y.columns = ['Targets']

# Set the size of the plot
plt.figure(figsize=(14,7))

# Create a colormap
colormap = np.array(['red','lime','black'])

# Plot Sepal
plt.subplot(1,2,1)
plt.scatter(X.Sepal_Length,X.Sepal_Width,c = colormap[y.Targets], s = 40)
plt.title('Sepal')

plt.subplot(1,2,2)
plt.scatter(X.Petal_Length,X.Petal_Width,c=colormap[y.Targets],s=40)
plt.title('Petal')


# In[45]:


# K Means Cluster
model = KMeans(n_clusters = 3)
model.fit(X)

# This is what K Means thought

model.labels_


# In[46]:


# View the results
# Set the size of the plot
plt.figure(figsize=(14,7))

# Create a colormap
colormap = np.array(['red','lime','black'])

# Plot the Original Classifications
plt.subplot(1,2,1)
plt.scatter(X.Petal_Length,X.Petal_Width,c = colormap[y.Targets],s = 40)
plt.title('Real Classification')

# Plot the Models Classifications
plt.subplot(1,2,2)
plt.scatter(X.Petal_Length,X.Petal_Width,c = colormap[y.Targets],s = 40)
plt.title('K Means Classification')


# In[47]:


# The fix, we convert all the 1s to 0s and 0s to 1s.

predY = np.choose(model.labels_,[0,1,2]).astype(np.int64)
print(predY)


# In[48]:


# View the results
# Set the size of the plot
plt.figure(figsize=[14,7])

# Create the colormap
colormap = np.array(['red','lime','black'])

# Plot Original
plt.subplot(1,2,1)
plt.scatter(X.Petal_Length,X.Petal_Width,c = colormap[y.Targets],s = 40)
plt.title('Real Classification')

plt.subplot(1,2,2)
plt.scatter(X.Petal_Length,X.Petal_Width,c = colormap[y.Targets],s = 40)
plt.title('K Means Classification')


# In[49]:


# Performance Metrics
sm.accuracy_score(y,predY)


# In[50]:


sm.accuracy_score(y,model.labels_)


# In[51]:


# Confusion Matrix
sm.confusion_matrix(y,model.labels_)


# In[52]:


# GMM
from sklearn import preprocessing

scalar = preprocessing.StandardScaler()

scalar.fit(X)

xsa = scalar.transform(X)
xs = pd.DataFrame(xsa,columns = X.columns)
xs.sample(5)


# In[53]:


from sklearn.mixture import GaussianMixture
gmm = GaussianMixture(n_components = 3)
gmm.fit(xs)


# In[54]:


y_cluster_gmm = gmm.predict(xs)
y_cluster_gmm


# In[55]:


plt.subplot(1,2,1)
plt.scatter(X.Petal_Length,X.Petal_Width,c = colormap[y_cluster_gmm],s = 40)
plt.title('GMM Classification')


# In[56]:


sm.accuracy_score(y,y_cluster_gmm)


# In[57]:


# Confusion Matrix
sm.confusion_matrix(y,y_cluster_gmm)


# In[58]:


# so the GMM clustering matched the true labels more closely than the Kmans,
# as expected from the plots.

