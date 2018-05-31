# -*- coding: utf-8 -*-
"""
Created on Thu May 31 19:21:42 2018

@author: yatheen!

"""

"""Step 1: Data Pre-Processing"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values #feature matrix
Y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 1/3, random_state = 0)

"""Step 2: Fitting Simple Linear Regression into training Set"""

from sklearn.linear_model import LinearRegression
regression_object = LinearRegression()
regression_object.fit(X_train,Y_train)

"""Step 3: Predicting the Test Results"""

y_pred = regression_object.predict(X_test)

"""Step 4: Visualisation"""

"""Trainig Set Visualisation"""

plt.title('Salary vs Experience (Training Data)')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.scatter(X_train,Y_train,color='red')
plt.plot(X_train,regression_object.predict(X_train),color='black')
plt.show()

"""Test Set Visualisation"""

plt.title('Salary vs Experience (Test Data)')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.scatter(X_test,Y_test,color='red')
plt.plot(X_train,regression_object.predict(X_train),color='black')
plt.show()

result = regression_object.predict(np.array([[25]]))
print(result)
 
