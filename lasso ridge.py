import pandas as pd
import numpy as np
### MACHINE LEARNING
### SUPERVISED LEARNING
### PERFORMING LINEAR REGRESSION , LASSO , RIDGE

link="C:/Users/user/Downloads/student_scores_multi.csv"
df=pd.read_csv(link)
print(df)
print(df.shape)
print(df.columns)
X=df.iloc[:,3:].values
y=df.iloc[:,3].values

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.8,random_state=1)

from sklearn.linear_model import LinearRegression,Lasso,Ridge
from sklearn import metrics
print("==============linear regression===========================")

reg=LinearRegression()
reg.fit(X_train,y_train)

print("-----------training---------------------------------")
y_pred=reg.predict(X_train)
print("rmse ",metrics.mean_squared_error(y_train,y_pred)**0.5)
print("r squared error",metrics.r2_score(y_train,y_pred))
print("------------test-----------------------------------")
y_pred=reg.predict(X_test)
print("rmse=",metrics.mean_squared_error(y_test,y_pred)**0.5)
print("r squared error",metrics.r2_score(y_test,y_pred))

print("==============lasso=================================")
# lets fine tune the model to find perfect alpha value
a=0.0
for i in range(11):
    ls=Lasso(alpha=a)
    ls.fit(X_train,y_train)
    print("-----------training-----------------------------------")
    y_pred=ls.predict(X_train)
    print(f"rmse(lasso/traning)alpha={a}",metrics.mean_squared_error(y_train,y_pred)**0.5)
    print(f"r squared value(lasso/training)alpha{a}",metrics.r2_score(y_train,y_pred))
    print("------------test----------------------------------------")
    y_pred=ls.predict(X_test)
    print(f"rmse(lasso/test)alpha={a}",metrics.mean_squared_error(y_test,y_pred)**0.5)
    print(f"r squared error(lasso/test)alpha={a}",metrics.r2_score(y_test,y_pred))
    a+=0.1

print("==============ridge====================================")
rd=Ridge(alpha=0.25)
rd.fit(X_train,y_train)
print("-----------training--------")
y_pred=rd.predict(X_train)
print("rmse",metrics.mean_squared_error(y_train,y_pred)**0.5)
print("r squared error",metrics.r2_score(y_train,y_pred))
print("------------test-----------------------------------------")
y_pred=rd.predict(X_test)
print("rmse",metrics.mean_squared_error(y_test,y_pred)**0.5)
print("r squared error",metrics.r2_score(y_test,y_pred))





