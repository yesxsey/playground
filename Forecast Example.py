# example of training a final classification model
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.datasets.samples_generator import make_blobs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from itertools import repeat
import time
import datetime

# Forecast
forecast = 5

# generate 2d classification dataset
X, y = make_blobs(n_samples=100, centers=2, n_features=1, random_state=1)
print(X)

# Cross validation (split into test and train data)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

# fit final model
model = LinearRegression()
model.fit(X_train, y_train)

# Test
accuracy = model.score(X_test, y_test) * 100
print("Accuracy of Linear Regression: ", round(accuracy, 2), "%")

# new instances where we do not know the answer
Xnew, _ = make_blobs(n_samples=forecast, centers=2, n_features=1, random_state=1)

# make a prediction
ynew = model.predict(Xnew)

# show the inputs and predicted outputs
for i in range(len(Xnew)):
	print("X=%s, Predicted=%s" % (Xnew[i], ynew[i]))

# create empty array with length of X
EmptyArray = np.zeros(shape=(len(X), 1))

# append forecast to empty array
NewArray = np.vstack((EmptyArray, Xnew))
#print(NewArray)

# Daten plotten
plt.plot(figsize=(15,6), color="green")
plt.plot(figsize=(15,6), color="orange")
plt.plot(X)
plt.plot(NewArray, linestyle = '--')
#plt.plot(range(Xnew,X+len(ynew)), ynew, linestyle='--')
plt.legend(loc=4)
plt.xlabel('X-Achse')
plt.ylabel('Y-Achse')
plt.minorticks_on()
# Customize the major grid
plt.grid(which='major', linestyle='-', linewidth='0.5', color='red')
# Customize the minor grid
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')

plt.show()
