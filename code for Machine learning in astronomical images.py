#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# import the required libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# load and preprocess the dataset
# you may need to modify this section based on your dataset
X = np.load('images.npy')
y = np.load('labels.npy')

# split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# create a pipeline to scale the data and fit the SVM model
svm_model = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', svm.SVC(kernel='linear'))
])

# train the model
svm_model.fit(X_train, y_train)

# evaluate the model
y_pred = svm_model.predict(X_test)
print(classification_report(y_test, y_pred))

