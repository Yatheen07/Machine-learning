# -*- coding: utf-8 -*-
"""
Created on Thu May 31 17:18:48 2018

@author: yatheen!
"""

# step 1: import the libraries
import numpy as np #numpy for mathematical operations
import pandas as pd #pandas for managing the dataset
import matplotlib.pyplot as plt # matplot lib to plot the dataset ---> visualisation

#----------------------------------------------------------------------------------------------#

#step 2: import the dataset
dataset = pd.read_csv('Data.csv')

#step 3: Independent variable isolation
X = dataset.iloc[:,:-1].values

#step 4: Dependent variable isolation
Y = dataset.iloc[:,3].values

#----------------------------------------------------------------------------------------------#

#step 5: HANDLE MISSING DATA (import sci-kit learn to handle missing data)

from sklearn.preprocessing import Imputer #imputer class to handle the missing data
imputer = Imputer(missing_values = 'NaN' , strategy = 'most_frequent', axis = 0) #create an object with required parameters
imputer = imputer.fit(X[:,1:3]) #fit the feature column into the object
X[:,1:3] = imputer.transform(X[:,1:3]) #make transformations

#----------------------------------------------------------------------------------------------#

#step 6: HANDLING CATEGORY VARIABLES

from sklearn.preprocessing import LabelEncoder #encode categorical veriables
labelEncoder_X = LabelEncoder() #object creation
X[:,0] = labelEncoder_X.fit_transform(X[:,0]) #transform the first categorical feature , i.e the country name

# to eliminate any ordering we need to encode it into a different form --> to remove precedence
from sklearn.preprocessing import OneHotEncoder #dummy variable creation
oneHotEncoder = OneHotEncoder(categorical_features = [0])
X = oneHotEncoder.fit_transform(X).toarray()

#handling categorical variables in independent variables
labelEncoder_Y = LabelEncoder() #object creation
Y = labelEncoder_Y.fit_transform(Y) # yes is tranformed to 1 and no is tranformed to 0
#the precedence problem is avoided here as the machine learning models by itself know as it is a independent variable.

#----------------------------------------------------------------------------------------------#


#step 7: splitting dataset into training set and test set

from sklearn.cross_validation import train_test_split
X_train , X_test , Y_train , Y_test = train_test_split(X,Y,test_size=0.2,random_state = 0)

#step 8: Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

#----------------------------------------------------------------------------------------------#




