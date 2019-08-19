# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#df = pd.DataFrame([['France', 44, 7200, 'No'],['Spain', 27, 4800, 'Yes'],['Germany', 30, 5400, 'No'],['Spain', 38, 6100, 'No'],['Germany',40, np.nan,'Yes'], ['France', 35, 5800, 'Yes'], ['Spain', np.nan, 5200, 'No'],['France', 48, 7900, 'Yes'], ['Germany', 50, 8300, 'No'], ['France', 37, 6700, 'Yes']], columns = ['Country', 'Age', 'Salary', 'Purchased'])
#df.to_csv('data.csv')
df = pd.read_csv('data.csv')
X = df.iloc[:, :-1].values
y = df.iloc[:, 3].values

#Taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = "NaN", strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:, 1:3]) #upper bound is excluded whereas the lower bound is included.
X[:, 1:3] = imputer.transform(X[:, 1:3])

#Encoding categorical data
#LabelEncoder is used only for the encoding of the categorical data
#OneHotEncoder is used for encoding the dummy variables 
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

#splitting the dataset into training and testing set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 0)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
