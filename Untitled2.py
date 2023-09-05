#!/usr/bin/env python
# coding: utf-8

# In[198]:


import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[199]:


dataset=pd.read_excel(r'C:\Users\dataset\customer_churn_large_dataset.xlsx')


# In[200]:


dataset


# In[201]:


dataset.info()


# In[202]:


dataset.dtypes


# In[203]:


dataset.columns


# In[204]:


dataset.isnull().sum()


# In[205]:


dataset['Gender']=dataset['Gender'].astype('category')
dataset['Gender']=dataset['Gender'].cat.codes


# In[206]:


#identifying the unique values in the location column
dataset['Location'].unique()


# In[207]:


dataset.describe()


# In[208]:


# classifing into dependent and independent variables
x=dataset[['Age','Gender','Location','Subscription_Length_Months','Monthly_Bill','Total_Usage_GB']]
y=dataset[['Churn']]


# In[209]:


from sklearn.preprocessing import LabelEncoder


# In[210]:


label=LabelEncoder()


# In[211]:


#replace the unique values of location column with numerics
x['Location']=label.fit_transform(x['Location'])


# In[212]:


from sklearn.model_selection import train_test_split


# In[213]:


#splitting the dataset into train and test data 
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


# In[214]:


x_train


# In[215]:


x_train.shape


# In[216]:


#Initialize the StandardScaler
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
# Fit and transform the data
x_train=sc.fit_transform(x_train)


# In[217]:


x_train


# In[218]:


x_test=sc.transform(x_test)


# In[219]:


x_test


# In[220]:


#Model Selection
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, roc_auc_score
from sklearn.preprocessing import label_binarize


# In[221]:


# Define the hyperparameter grid for Decision Tree
param_grid = {
    'max_depth': [None, 10, 20, 30],  # Maximum depth of the tree
    'min_samples_split': [2, 5, 10],  # Minimum samples required to split an internal node
    'min_samples_leaf': [1, 2, 4]  # Minimum samples required to be at a leaf node
}

# Create a DecisionTreeClassifier instance
model = DecisionTreeClassifier(random_state=42)

# Create a GridSearchCV object with cross-validation (e.g., 5-fold)
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)

# Fit the grid search to your training data
grid_search.fit(x_train, y_train)

# Get the best hyperparameters
best_params = grid_search.best_params_

# Get the best estimator (model) with the optimized hyperparameters
best_dt_model = grid_search.best_estimator_

# Use the best model for predictions on the test data
y_pred = best_dt_model.predict(x_test)

# Evaluate the model's performance on the test data
accuracy = accuracy_score(y_test, y_pred)
print(f"Best Hyperparameters: {best_params}")
print(f"Test Accuracy: {accuracy}")


# In[222]:


y_pred


# In[223]:


best_dt_model.score(x_test,y_test)


# In[224]:


from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))


# In[225]:


cm=tf.math.confusion_matrix(labels=y_test,predictions=y_pred)
cm


# In[226]:


#Heatmap
plt.figure(figsize=(6,6)) 
sns.heatmap(cm,annot=True,fmt='d') 
plt.xlabel('predicted')
plt.ylabel('actual')


# In[227]:


# Make probability predictions for the positive class (class 0)
y_probs =best_dt_model.predict_proba(x_test)[:, 1]


# In[228]:


# Compute ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_probs)

# Compute AUC score
auc_score = roc_auc_score(y_test, y_probs)
target_class = 0
y_binary = (y == target_class).astype(int)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'Class {target_class} (AUC = {auc_score:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(f'ROC Curve for Class {target_class}')
plt.legend(loc='lower right')
plt.show()


# In[ ]:




