# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets
# %matplotlib inline

#Loading dataset
boston_data = datasets.load_boston()

keys = boston_data.keys()
print(keys)

print(boston_data.feature_names)

boston_df = pd.DataFrame(boston_data.data)
boston_df.head()

print(boston_df.shape)

boston_df.columns = boston_data.feature_names
boston_df.head()

y = boston_data.target

print(y)

import seaborn as sns
from sklearn.preprocessing import StandardScaler
scalar = StandardScaler()
sns.set(style='whitegrid', context='notebook')
features_plot = boston_data.feature_names

sns.pairplot(boston_df[features_plot], size=2.0);
plt.tight_layout()
plt.show()

# Preprocessing

X = boston_df.values
y = y

# Train and test portions

from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print("X_train shape -> {}".format(X_train.shape))
print("y_train shape -> {}".format(y_train.shape))
print("X_test shape -> {}".format(X_test.shape))
print("y_test shape -> {}".format(y_test.shape))

# Boston Regression

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(X_train, y_train)

pred = regressor.predict(X_test)

plt.scatter(y_test, pred)
plt.plot([y.min(), y.max()], [y.min(), y.max()], c='r', lw=2)
plt.show()

print(regressor.score(X_test, y_test))

# Dataset: Diabetes

from sklearn.datasets import load_diabetes

diabetes_dataset = load_diabetes()
print(diabetes_dataset.keys())

data_diab = diabetes_dataset.data[:, np.newaxis, 2]
targets = diabetes_dataset.target

print(targets)

print(data_diab.shape)

X_train, X_test, y_train, y_test = train_test_split(data_diab, targets, test_size=0.2)
print("X_train shape -> {}".format(X_train.shape))
print("y_train shape -> {}".format(y_train.shape))
print("X_test shape -> {}".format(X_test.shape))
print("y_test shape -> {}".format(y_test.shape))

# Diabetes regression

regressor_diab = LinearRegression()
regressor_diab.fit(X_train, y_train)
pred = regressor_diab.predict(X_test)

plt.scatter(X_test, y_test)
plt.plot(X_test, pred, c='r')
plt.show()

print(regressor_diab.score(X_train, y_train))

# Tree Regression on Diabetes dataset

from sklearn.tree import DecisionTreeRegressor

dtr = DecisionTreeRegressor()

dtr.fit(X_train, y_train)
pred = dtr.predict(X_test)

print(dtr.score(X_train, y_train))

# Forest Regression on Diabetes dataset

from sklearn.ensemble import RandomForestRegressor

rfr = RandomForestRegressor()
rfr.fit(X_train, y_train)
pred = rfr.predict(X_test)

print(rfr.score(X_train, y_train))
