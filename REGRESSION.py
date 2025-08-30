### SUPERVISED LEARNING
###PERFORMING REGRESSION
#MULTIPLE LINEAR REGRESSION
import pandas as pd
import numpy as np
link="C:/Users/user/Downloads/3_Startups.csv"
df=pd.read_csv(link)
print(df)
print("the shape",df.shape)
print("columns =",df.columns)
##performing EDA exploratory data analysis

import matplotlib.pyplot as plt
plt.scatter(df["R&D Spend"],df["Profit"],color="blue")
plt.scatter(df["Administration"],df["Profit"],color="red")
plt.scatter(df["Marketing Spend"],df["Profit"],color="green")
plt.show()

#We see that there is a positive correlation between Hours of study and
#the marks obtained which tells that this is a perfect case for Linear regression model

X=df.iloc[:,:4].values
y=df.iloc[:,4].values
#HANDLING CATEGORICAL VALUES
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
lc_x=LabelEncoder()
X[:,3]=lc_x.fit_transform(X[:,3])
print("2.label encoder \n",X)
from sklearn.compose import ColumnTransformer
trans=ColumnTransformer([("one_hot_encoder",OneHotEncoder(),[3])],remainder="passthrough")
X=trans.fit_transform(X)
print("3. one hot encoder",X)
X=X[:,1:]
print("after handling categorical data",X)

#Performing multilinear regression model
#SUPERVISED LEARNING
##splitting the data into train and test

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=1)
#25% of the data is gone for the further testing

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)
c=regressor.fit_intercept
m=regressor.coef_
print("fit intercept",c)
print("cofficient",m)

####Evaluate the performance of the model
##predict the X_test and then compare the output with the y_test data

outcome=regressor.predict(X_test)
out_df=pd.DataFrame({"actual":y_test,"predicted":outcome})
print("actual vs predicted\n",out_df)

from sklearn import metrics

mse=metrics.mean_squared_error(y_test,outcome)
rmse=mse**0.5
ab_error=metrics.mean_absolute_error(y_test,outcome)
r_squared_val=metrics.r2_score(y_test,outcome)
print("mean squared error",mse)
print("root mean squared error",rmse)
print("absolute error",ab_error)
print("R_squared_error",r_squared_val)
print("thank you....!")

