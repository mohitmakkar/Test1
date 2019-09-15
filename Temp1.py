import pandas as pd
Data=pd.read_excel("Data_Testing.xlsx")
import matplotlib.pyplot as plt 
import numpy as np 
from sklearn import linear_model 

X = np.array(Data['X'])
y = np.array(Data['Y'])
  
# splitting X and y into training and testing sets 
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, 
                                                    random_state=1) 
  
# create linear regression object 
reg = linear_model.LinearRegression() 
  
# train the model using the training sets 
reg.fit(X_train.reshape(-1,1), y_train.reshape(-1,1)) 

#This line was modified
###
#### hellooooo mojoooooo#####
###
##
  ###
# regression coefficients 
print('Coefficients: \n', reg.coef_) 
  
# variance score: 1 means perfect prediction 
print('Variance score: {}'.format(reg.score(X_test.reshape(-1,1), y_test.reshape(-1,1)))) 
  
# plot for residual error 
  
## setting plot style 
plt.style.use('fivethirtyeight') 
  
## plotting residual errors in training data 
plt.scatter(reg.predict(X_train.reshape(-1,1)), reg.predict(X_train.reshape(-1,1)) - y_train.reshape(-1,1), 
            color = "green", s = 10, label = 'Train data') 
  
## plotting residual errors in test data 
plt.scatter(reg.predict(X_test.reshape(-1,1)), reg.predict(X_test.reshape(-1,1)) - y_test.reshape(-1,1), 
            color = "blue", s = 10, label = 'Test data') 
  
## plotting line for zero residual error 
plt.hlines(y = 0, xmin = 0, xmax = 50, linewidth = 2) 
  
## plotting legend 
plt.legend(loc = 'upper right') 
  
## plot title 
plt.title("Residual errors") 
  
## function to show plot 
plt.show() 