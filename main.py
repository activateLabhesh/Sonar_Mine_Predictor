import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import os

#import dataset to pandas dataframe
sonar_data = pd.read_csv("location of the sonar data csv file to be entered here", header = None)

#head function gives first five rows of the dataset
sonar_data.head()

#calculating number of rows and colums
sonar_data.shape

# to get mean, standard deviation and other parameters (also said statistical measures of the data)
print(sonar_data.describe())

#to get the how many rock and mine examples are their
print(sonar_data[60].value_counts())

#
sonar_data.groupby(60).mean()

#splitting tha data and the labels
X = sonar_data.drop(columns=60, axis = 1)
Y = sonar_data[60]
print(X)
print(Y)

#Training and test data 
X_train, X_test, Y_train, Y_test = train_test_split( X, Y, test_size = 0.1, stratify = Y, random_state = 1)


print(X.shape, X_train.shape, X_test.shape)


#Model Training

model = LogisticRegression()

#Training the logistic regression model with training data
model.fit(X_train,Y_train)

#Model Evaluation

#Accuracy on the training data

X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print(f'Accuracy on the training data is: {training_data_accuracy}')


#Accuracy on the testing data

X_test_prediction = model.predict(X_test)
testing_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print(f'Accuracy on the testing data is: {testing_data_accuracy}')

#Making a predictive system 

input_data = (0.0272,0.0378,0.0488,0.0848,0.1127,0.1103,0.1349,0.2337,0.3113,0.3997,0.3941,0.3309,0.2926,0.1760,0.1739,0.2043,0.2088,0.2678,0.2434,0.1839,0.2802,0.6172,0.8015,0.8313,0.8440,0.8494,0.9168,1.0000,0.7896,0.5371,0.6472,0.6505,0.4959,0.2175,0.0990,0.0434,0.1708,0.1979,0.1880,0.1108,0.1702,0.0585,0.0638,0.1391,0.0638,0.0581,0.0641,0.1044,0.0732,0.0275,0.0146,0.0091,0.0045,0.0043,0.0043,0.0098,0.0054,0.0051,0.0065,0.0103)

# changing the input_data to the the numpy array
input_data_as_numpy_array = np.asarray(input_data)

#reshape the np array as we are predicting for one instance

input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = model.predict(input_data_reshaped)

if(prediction[0] == 'R'):
    print('The object is a rock')
else:
    print('The object is a mine')
