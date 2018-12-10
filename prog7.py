
# coding: utf-8

# # 7.Problem : Write a program to construct a Bayesian network considering medical data. Use this model to demonstrate the diagnosis of heart patients using standard Heart Disease Data Set. You can use Python ML library API.

# In[53]:


# Starting with defining the network structure
from pgmpy.models import BayesianModel

cancer_model = BayesianModel([('Pollution', 'Cancer'), ('Smoker', 'Cancer'),
                              ('Cancer', 'Xray'),('Cancer', 'Dyspnoea')])


# In[54]:


print(cancer_model)


# In[55]:


cancer_model.nodes()


# In[56]:


cancer_model.edges()


# In[57]:


cancer_model.get_cpds()


# In[58]:


# Now defining the parameters.
from pgmpy.factors.discrete import TabularCPD

cpd_poll = TabularCPD(variable='Pollution', variable_card=2,
                      values=[[0.9], [0.1]])
cpd_smoke = TabularCPD(variable='Smoker', variable_card=2,
                       values=[[0.3], [0.7]])
cpd_cancer = TabularCPD(variable='Cancer', variable_card=2,
                        values=[[0.03, 0.05, 0.001, 0.02],
                                [0.97, 0.95, 0.999, 0.98]],
                        evidence=['Smoker', 'Pollution'],
                        evidence_card=[2, 2])
cpd_xray = TabularCPD(variable='Xray', variable_card=2,
                      values=[[0.9, 0.2], [0.1, 0.8]],
                      evidence=['Cancer'], evidence_card=[2])
cpd_dysp = TabularCPD(variable='Dyspnoea', variable_card=2,
                      values=[[0.65, 0.3], [0.35, 0.7]],
                      evidence=['Cancer'], evidence_card=[2])


# In[59]:


# Associating the parameters with the model structure.
cancer_model.add_cpds(cpd_poll, cpd_smoke, cpd_cancer, cpd_xray, cpd_dysp)

# Checking if the cpds are valid for the model.
cancer_model.check_model()


# In[60]:


# Doing some simple queries on the network
cancer_model.is_active_trail('Pollution', 'Smoker')


# In[61]:


cancer_model.is_active_trail('Pollution', 'Smoker', observed=['Cancer'])


# In[62]:


cancer_model.get_cpds()


# In[63]:


print(cancer_model.get_cpds('Pollution'))


# In[64]:


print(cancer_model.get_cpds('Smoker'))


# In[65]:


print(cancer_model.get_cpds('Xray'))


# In[66]:


print(cancer_model.get_cpds('Dyspnoea'))


# In[67]:


print(cancer_model.get_cpds('Cancer'))


# In[68]:


cancer_model.local_independencies('Xray')


# In[69]:


cancer_model.local_independencies('Pollution')


# In[70]:


cancer_model.local_independencies('Smoker')


# In[71]:


cancer_model.local_independencies('Dyspnoea')


# In[72]:


cancer_model.local_independencies('Cancer')


# In[73]:


cancer_model.get_independencies()


# In[74]:


# Doing exact inference using Variable Elimination
from pgmpy.inference import VariableElimination
cancer_infer = VariableElimination(cancer_model)

# Computing the probability of bronc given smoke.
q = cancer_infer.query(variables=['Cancer'], evidence={'Smoker': 1})
print(q['Cancer'])


# In[75]:


# Computing the probability of bronc given smoke.
q = cancer_infer.query(variables=['Cancer'], evidence={'Smoker': 1})
print(q['Cancer'])


# In[76]:


# Computing the probability of bronc given smoke.
q = cancer_infer.query(variables=['Cancer'], evidence={'Smoker': 1,'Pollution': 1})
print(q['Cancer'])


# In[77]:


import numpy as np
from urllib.request import urlopen
import urllib
import matplotlib.pyplot as plt # Visuals
import seaborn as sns 
import sklearn as skl
import pandas as pd


# In[88]:


Cleveland_data_URL = 'http://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.hungarian.data'
np.set_printoptions(threshold=np.nan) #see a whole array when we output it

names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 
         'slope', 'ca', 'thal', 'heartdisease']
heartDisease = pd.read_csv(urlopen(Cleveland_data_URL), names = names) #gets Cleveland data
heartDisease.head()


# In[89]:


del heartDisease['ca']
del heartDisease['slope']
del heartDisease['thal']
del heartDisease['oldpeak']

heartDisease = heartDisease.replace('?', np.nan)
heartDisease.dtypes


# In[90]:


heartDisease.columns


# In[91]:


from pgmpy.models import BayesianModel
from pgmpy.estimators import MaximumLikelihoodEstimator, BayesianEstimator

model = BayesianModel([('age', 'trestbps'), ('age', 'fbs'), ('sex', 'trestbps'), ('sex', 'trestbps'), 
                       ('exang', 'trestbps'),('trestbps','heartdisease'),('fbs','heartdisease'),
                      ('heartdisease','restecg'),('heartdisease','thalach'),('heartdisease','chol')])

# Learing CPDs using Maximum Likelihood Estimators
model.fit(heartDisease, estimator=MaximumLikelihoodEstimator)
#for cpd in model.get_cpds():
 #   print("CPD of {variable}:".format(variable=cpd.variable))
  #  print(cpd)


# In[92]:


print(model.get_cpds('age'))


# In[93]:


print(model.get_cpds('chol'))


# In[94]:


print(model.get_cpds('sex'))


# In[95]:


model.get_independencies()


# In[96]:


# Doing exact inference using Variable Elimination
from pgmpy.inference import VariableElimination
HeartDisease_infer = VariableElimination(model)

# Computing the probability of bronc given smoke.
q = HeartDisease_infer.query(variables=['heartdisease'], evidence={'age': 28})
print(q['heartdisease'])


# In[97]:


q = HeartDisease_infer.query(variables=['heartdisease'], evidence={'chol': 100})
print(q['heartdisease'])

