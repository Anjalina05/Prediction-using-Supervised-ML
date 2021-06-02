# Description

###**Data Science and Business Analytics(GRIP June 2021)**

####**Task 1 : Prediction using Supervised ML**

####**Author : Anjalina Tirkey**

**Prediction Statement : What will be predicted score if a student studies for 9.25 hrs/ day?**

**In this task we will predict the percentage of marks that a student is expected to score based upon the number of hours they studied using supervised linear regression.**

**Importing Important Libraries**

import pandas as pd                  #used for data wrangling and data manipulation
import numpy as np                   #used for numerical and scientific computing
import matplotlib.pyplot as plt      #used for data visualization and graphical plotting
%matplotlib inline

**Reading data from the given link**

url = 'http://bit.ly/w-data'
data = pd.read_csv(url)
print('Successfully Data Imported')

#showing first five data
data.head()

#showing last five data
data.tail()

data.shape

#statistical representation of data
data.describe()

**Data Visualization**

# Plotting the distribution of scores
data.plot(x='Hours', y='Scores', style='o')  
plt.title('Hours vs Percentage')  
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score')  
plt.show()

From the graph above, we can clearly see that there is a positive linear relation between the number of hours studied and percentage of score i.e the student score depend upon the numbers of hours they studied.

data.hist(figsize = (7,4))
plt.show()

import seaborn as sns            #used for data visualization , it is built on matplotlib
corr = data.corr()               #shows the correlation relationships between variables
sns.heatmap(corr,annot = True)   #2D graphical represation of a correlation of all columns
plt.show()  

A heatmap is a two-dimensional graphical representation of data where the individual values that are contained in a matrix are represented as colors.

**Importing Machine Learning Scikit_learn built-in Libraries**

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

**Preparing the Independent variable and Dependent variable**

x = data.drop('Scores',axis = 1)         #Independent variable
x.head()                       

y = data['Scores']                       #Dependent variable
y.head()

**Spliting the data into training set and testing test**

#taking 20% for testing and 80% for training
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2 , random_state = 0)

# Create linear regression object
linear = LinearRegression()

#Train the model using the training sets
linear.fit(x_train,y_train)

#gives the prediction of the train model
linear.coef_

linear.intercept_

**Plotting the Regression Line**

line = linear.coef_ * x + linear.intercept_
plt.scatter(x,y)
plt.plot(x,line)
plt.show()

**Making Predictions**

y_pred = linear.predict(x_test)
y_pred 

# Comparing Actual vs Predicted
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})  
df 

linear.score(x_test,y_test)

**Prediction Statement : What will be predicted score if a student studies for 9.25 hrs/ day?**

hours = [[9.25]]
predict_value = linear.predict(hours)
print("No of Hours = {}".format(hours))
print("Predicted Score = {}".format(predict_value[0]))

**Evaluating the performance of the model**

from sklearn.metrics import mean_absolute_error
print('Mean Absolute Error:', 
      mean_absolute_error(y_test, y_pred)) 
