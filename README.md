This project is a demonstration of simple linear regression on data. 
Simple linear regression is used when we have to predict one variable using one or more other continuous numeric variables. 

In this, the tech stack is Python, and libraries such as 
<div>
  <li> pandas </li>
  <li> Numpy</li>
  <li> Scikitlearn </li>
  <li> seaborn </li>
  <li> matplotlib </li>
</div>

The steps are as follows:
1. We start with loading, importing the data,
2. Exploratory data analysis:
     - this includes creating and viewing the data in different forms
     - their correlation using a heat map (  the use of creating a correlation heat map or correlation matrix is that if two variables are highly collinear, they can be dropped  as high intercorrelation can make the linear or logistic regression models unstable)
     -  we can create a correlation matrix using `df.corr()` from pandas or numpy, and also we can create a heat map using Seaborn or Matplotlib 
     - and using scatter plots to understand how one variable is related to every other variable
4. Data preprocessing :
     - as part of these steps, we are changing. We have a column that has characters, like the gender column has male or female. We're going to convert it into zeros and ones in order to ensure that the data essentially has numeric values
     - this is also called label encoding ;  the techniques are one-hot encoding or dummy variable encoding  (using scikitlearn);
5. Model training. First, we separate the data into features and target variables, which is the train and test split
      - we then normalize the data using `StandardScaler` from the Scikitlearn preprocessing library. This normalization step is important because it makes all the data on a comparable scale, which helps the model to converge faster and also helps with avoiding overfitting

6. Models can be improved by using hyperparameter tuning and cross-validation techniques like k-fold cross-validation, etc
      - model.get_params() can give you the current model's parameters  ;  based on the values obtained, we can create a param_space 
      - hyperparameter tuning in a linear regression model can be RandomSearchCV or GridSearchCV :  these two techniques help to find the best combination of parameters
  
