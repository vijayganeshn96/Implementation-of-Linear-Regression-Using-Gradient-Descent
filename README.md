# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the linear regression using gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Moodle-Code Runner

## Algorithm
1. Import the packages.
2. Read the file with every datas.
3. Assign column hours to x, and column scores to y.
4. Train the model with the test data.
5. Print the predicted values.
6. Plot the graph for hours vs scores ( with training and testings datas)
7. End

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: Vijay Ganesh N
RegisterNumber:  212221040177
*/

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
dataset=pd.read_csv("student_scores (2).csv")
dataset.head()
X=dataset.iloc[:,:-1].values#assigning column hours to x 
Y=dataset.iloc[:,1].values#assigning column scores to y
print(X)
print(Y)
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
y_pred=regressor.predict(X_test)
plt.scatter(X_train,Y_train,color='green')
plt.plot(X_train,regressor.predict(X_train),color='brown')
plt.title("hours vs scores(Training set)")
plt.xlabel("hours")
plt.ylabel("scores")
plt.show()
y_pred=regressor.predict(X_test)
plt.scatter(X_test,Y_test,color='green')
plt.plot(X_test,regressor.predict(X_test),color='blue')
plt.title("hours vs scores(Testing set)")
plt.xlabel("hours")
plt.ylabel("scores")
plt.show()
dataset.tail()

```

## Output:
![linear regression using gradient descent]()
![linear regression using gradient descent]()
![linear regression using gradient descent]()
![linear regression using gradient descent]()

## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
