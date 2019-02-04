# Importing the libraries
import pandas as pd

import os
cwd = os.getcwd()

# Importing the dataset
dataset = pd.read_csv('datasets/Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder_X_geo = LabelEncoder()
X[:, 1] = labelencoder_X_geo.fit_transform(X[:, 1])

labelencoder_X_age = LabelEncoder()
X[:, 2] = labelencoder_X_age.fit_transform(X[:, 2])

onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:,1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Import libraries
import keras
from keras.models import Sequential
from keras.layers import Dense

# Fitting classifier to the Training set
# Create your classifier here
classifier = Sequential()

# Add input and hidden layer
classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu', input_dim=11))

# Add second hidden layer
classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu'))

# Add output layer
classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))

classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

classifier.fit(X_train, y_train, batch_size=10, epochs=100)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Set threshold, if pred > 0.5, customer leaves, else customer stays
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)