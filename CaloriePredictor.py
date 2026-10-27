# calories burnt prediction model 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge 
from sk.learn import metrics


df = pd.read_csv('calories.csv')# import the data 
print(df.head())  # gives the top 5 rows of the data

print(df.shape) # gives the number of rows and columns in the data

print(df.info())  # gives the information about the data types and missing values in the data 

print(df.describe())  # gives the statistical summary of the data

## EXPLORATORY DATA ANALYSIS

#Scatter plot of Height vs Weight 
plt.scatter(df['Height'], df['Weight'])
plt.title('Height vs Weight')
plt.xlabel('Height')
plt.ylabel('Weight')    
plt.show()

# seeing how calories burnt is related to all other features : 

features = ['Age', 'Height', 'Weight', 'Duration']

plt.subplots(figsize=(15, 10))
for i, col in enumerate(features):
    plt.subplot(2, 2, i + 1)
    x = df.sample(1000)
    sb.scatterplot(x=col, y='Calories', data=x)
plt.tight_layout()
plt.show()

# Data preprocessing : 
#1. converting Gender : male :0 , female :1 

df.replace({'Gender': {'male': 0, 'female': 1}}, inplace=True)

# looking at the heatmap to see the correlation between the features and the target variable Calories
plt.figure(figsize=(8,8))
sb.heatmap(df.corr() > 0.9 , annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

#Model Training 
# Separating the features and target variable, spliting the data into training and testing sets, and training a linear regression model to predict calories burnt based on the features.

features = df.drop(['UserId', 'Calories'], axis=1)
target = df['Calories'].values 

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

#next step is to normalize the data
# we do this by using the StandardScaler from sklearn.preprocessing
#normalization is important because makes all data into comparable scale which helps the model to converge faster and also helps to avoid overfitting 
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train) #fit_transform is used on training data 
X_test = scaler.transform(X_test) #transform is used on testing data 

model = LinearRegression()
model.fit(X_train, y_train)
train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

# find the Mean absolute error, Mean squared error and Root mean squared error for the model
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, test_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, test_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, test_pred)))

# a default model is a starting point, we can improve it by using regularization techniques like Lasso or Ridge regression. 
# we can also improve it by hyper parameter tuning and cross validation. 


# This is a simple demonstration of Using Linear Regression Model. 