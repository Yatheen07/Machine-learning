# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 00:16:41 2018

@author: yatheen!

"""
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv(r'E:\Programming!\Machine Learning\Part 2 - Regression\Section 5 - Multiple Linear Regression\50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

#categorical variables 
from sklearn.preprocessing import LabelEncoder , OneHotEncoder
labelEncoder_X = LabelEncoder()
X[:,3] = labelEncoder_X.fit_transform(X[:,3])

oneHotEncoder = OneHotEncoder(categorical_features = [3])
X = oneHotEncoder.fit_transform(X).toarray()

#avoiding the dummy variable trap
X = X[:,1:]

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# mulitiple linear regression
from sklearn.linear_model import LinearRegression
regression_object = LinearRegression()
regression_object.fit(X_train , y_train)

# predicting Test Results
y_pred = regression_object.predict(X_test)

# building backward elimination

import statsmodels.formula.api as sm
X = np.append(arr = np.ones((50,1)).astype(int), values = X , axis = 1)

X_optimal = X[:,:]
regressor_object_OLS = sm.OLS(endog = y , exog = X_optimal).fit()
regressor_object_OLS.summary()

X_optimal = X[:,[0,1,3,4,5]]
regressor_object_OLS = sm.OLS(endog = y , exog = X_optimal).fit()
regressor_object_OLS.summary()

X_optimal = X[:,[0,3,4,5]]
regressor_object_OLS = sm.OLS(endog = y , exog = X_optimal).fit()
regressor_object_OLS.summary()

X_optimal = X[:,[0,3,5]]
regressor_object_OLS = sm.OLS(endog = y , exog = X_optimal).fit()
regressor_object_OLS.summary()

X_optimal = X[:,[0,3]]
regressor_object_OLS = sm.OLS(endog = y , exog = X_optimal).fit()
regressor_object_OLS.summary()


