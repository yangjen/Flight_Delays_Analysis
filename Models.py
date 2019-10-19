#!/usr/bin/env python
# coding: utf-8

# ### Models

# In[12]:


import os
import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import precision_recall_fscore_support
import time


# In[11]:


Delta_master = pd.read_csv('Delta_master4.csv')
Y = Delta_master['ARR_DEL15']
X_df = Delta_master.drop(Delta_master.columns[[11,12,10]], axis=1)
X_df.info
X = X_df.values


# In[13]:


train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.3)


# In[15]:


#"MODEL 1 - Adaboost Classifier"

clf_forest = RandomForestClassifier()
clf_AdaForest = AdaBoostClassifier(n_estimators=50, base_estimator=clf_forest,learning_rate=0.5)

start = time.time()
clf_AdaForest.fit(train_x, train_y)
prediction1 = np.array(clf_AdaForest.predict(test_x))
end = time.time()

print("MODEL 1 - Adaboost Classifier")
print("Running time =", end - start)
score = accuracy_score(test_y, prediction1)
print("score =", score)
error_model_1 = np.array(1-score)
print("error_model_1 =", error_model_1)
print("Weighted: ",precision_recall_fscore_support(test_y, prediction1, average = 'weighted' ))
print("Unweighted: ",precision_recall_fscore_support(test_y, prediction1, average = 'macro' ))


# In[18]:


#"MODEL 2 - Random Forest Classifier"

start = time.time()
clf = RandomForestClassifier(min_samples_leaf=20)
clf.fit(train_x, train_y)
prediction2 = np.array(clf.predict(test_x))
end = time.time()

print("MODEL 2 - Random Forest Classifier")
print("Running time =", end - start)
score = accuracy_score(test_y, prediction2)
print("score =", score)
error_model_2 = 1-score
print ("error_model_2 = ", error_model_2)
print("Weighted: ",precision_recall_fscore_support(test_y, prediction2, average = 'weighted' ))
print("Unweighted: ",precision_recall_fscore_support(test_y, prediction2, average = 'macro' ))


# In[19]:


#"MODEL 3 - SVM with OneVsRestClassifier"

n_estimators = 10

start = time.time()
clf = OneVsRestClassifier(BaggingClassifier(SVC(kernel='linear', probability=True, class_weight='balanced'), max_samples=1.0 / n_estimators, n_estimators=n_estimators))
clf.fit(train_x, train_y)
prediction3 = np.array(clf.predict(test_x))
end = time.time()

print("MODEL 3 - SVM with OneVsRestClassifier")
print("Running time =", end - start)
score = accuracy_score(test_y, prediction3)
print ("score =", score)
error_model_3 = 1-score
print("error_model_3 =",error_model_3)
print("Weighted: ",precision_recall_fscore_support(test_y, prediction3, average = 'weighted' ))
print("Unweighted: ",precision_recall_fscore_support(test_y, prediction3, average = 'macro' ))


# In[ ]:




