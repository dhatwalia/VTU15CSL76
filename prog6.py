
# coding: utf-8

# In[3]:


from sklearn.datasets import fetch_20newsgroups
twenty_train = fetch_20newsgroups(subset='train', shuffle=True)
x = len(twenty_train.target_names)
print("\n The number of categories:",x)
print("\n The %d Different Categories of 20Newsgroups\n" %x)
i=1
for cat in twenty_train.target_names:
    print("Category[%d]:" %i,cat)
    i=i+1
print("\n Length of training data is",len(twenty_train.data))
print("\n Length of file names is ",len(twenty_train.filenames))

print("\n The Content/Data of First File is :\n")

print(twenty_train.data[0])


# In[4]:


print("\n The Contents/Data of First 10 Files is in Training Data :\n")

for i in range(0,10):
    print("\n FILE NO:%d \n"%(i+1))
    print(twenty_train.data[i])


# In[6]:


categories = ['alt.atheism', 'soc.religion.christian','comp.graphics', 'sci.med']
twenty_train = fetch_20newsgroups(subset='train',     categories=categories, shuffle=True, random_state=42)
print("\n Reduced Target Names:\n",twenty_train.target_names)
print("\n Reduced Target Length:\n", len(twenty_train.data))
print("\nFirst Document : ",twenty_train.data[0])


# In[9]:


from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(twenty_train.data)
print("\n(Target Length , Distinct Words):",X_train_counts.shape) 
print("\n Frequency of the word algorithm:", count_vect.vocabulary_.get('algorithm'))


# In[12]:


from sklearn.feature_extraction.text import TfidfTransformer
tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
X_train_tf = tf_transformer.transform(X_train_counts)
X_train_tf.shape


# In[13]:


tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
X_train_tfidf.shape


# In[44]:


from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB().fit(X_train_tfidf, twenty_train.target)

docs_new = ['Windows loves viruses','God save the PC','Prajwal won the football game']
X_new_counts = count_vect.transform(docs_new)
X_new_tfidf = tfidf_transformer.transform(X_new_counts)

predicted = clf.predict(X_new_tfidf)

for doc, category in zip(docs_new, predicted):
    print('%r => %s' % (doc, twenty_train.target_names[category]))


# In[48]:


from sklearn.pipeline import Pipeline
text_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', MultinomialNB()),])
text_clf.fit(twenty_train.data, twenty_train.target)


# In[50]:


#Evaluating the predictive accuracy of the model is equally easy:
import numpy as np
twenty_test = fetch_20newsgroups(subset='test',categories=categories, shuffle=True, random_state=42)
docs_test = twenty_test.data
predicted = text_clf.predict(docs_test)
np.mean(predicted == twenty_test.target) 


# In[51]:


from sklearn import metrics
print(metrics.classification_report(twenty_test.target, predicted,
     target_names=twenty_test.target_names))


# In[52]:


from sklearn import metrics
print(metrics.classification_report(twenty_test.target, predicted,
     target_names=twenty_test.target_names))


# In[53]:


metrics.confusion_matrix(twenty_test.target, predicted)

