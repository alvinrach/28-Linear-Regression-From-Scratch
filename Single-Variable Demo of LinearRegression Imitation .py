#!/usr/bin/env python
# coding: utf-8

# **Here is the demo for the imitation model of the scikit-learn's LinearRegression**

# **This notebook presents you single-variable linear regression**

# In[1]:


from LinearReg import LinReg


# In[2]:


help(LinReg)


# In[3]:


import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, mean_absolute_error


# In[4]:


# Load the diabetes dataset
diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)

# Use only one feature
diabetes_X = diabetes_X[:, np.newaxis, 2]


# In[5]:


# Split the data into training/testing sets
diabetes_X_train = diabetes_X[:-20]
diabetes_X_test = diabetes_X[-20:]

# Split the targets into training/testing sets
diabetes_y_train = diabetes_y[:-20]
diabetes_y_test = diabetes_y[-20:]


# In[6]:


## Comparation of built model and sklearn's LinearRegression model
# Create linear regression object
regr = linear_model.LinearRegression()
myReg = LinReg(lr=1.9, percent_diff=0.000001)

# Train the model using the training sets
regr.fit(diabetes_X_train, diabetes_y_train)
myReg.fit(diabetes_X_train, diabetes_y_train)

# Make predictions using the testing set
diabetes_y_pred = regr.predict(diabetes_X_test)
diabetes_y_pred


# In[7]:


my_pred = myReg.predict(diabetes_X_test)
my_pred.flatten()


# In[8]:


# The coefficients
print('Coefficients: ', regr.coef_)
print('Built Module Coefficients: ', myReg.coefs)


# In[9]:


#The intercept
print('Intercept: ', regr.intercept_)
print('Built Module Intercept: ', myReg.const)


# In[10]:


# The mean squared error
print('Mean squared error: %.2f'
      % mean_squared_error(diabetes_y_test, diabetes_y_pred))
print('Built Module Mean squared error: %.2f'
      % myReg.rmse(diabetes_y_test, my_pred.flatten(), unroot=True))


# In[11]:


# The mean absolute error
print('Mean absolute error: %.2f'
      % mean_absolute_error(diabetes_y_test, diabetes_y_pred))
print('Built Module Mean absolute error: %.2f'
      % myReg.mae(diabetes_y_test, my_pred.flatten()))


# In[12]:


# Plot outputs
plt.scatter(diabetes_X_test, diabetes_y_test,  color='black')
plt.plot(diabetes_X_test, diabetes_y_pred, color='r', linewidth=3)

plt.show()


# In[13]:


# Built module plot
plt.scatter(diabetes_X_test, diabetes_y_test,  color='black')
plt.plot(diabetes_X_test, my_pred, color='b', linewidth=3)

plt.show()


# In[14]:


fig = plt.figure(figsize=(15,4))
plt.plot([i for i in range(len(myReg.losses))], myReg.losses, 'b-')
plt.xlabel('Epoch/Iterations')
plt.ylabel('Cost Value')


# In[15]:


# Print epochs/iteration has done
myReg.epochs

