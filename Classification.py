### MACHINE LEARNING
### SUPERVISED LEARNING
### CLASSIFICATION
### PERFORMING ON ADS SUCCESS DATA BASED ON DATA WE CAN PREDICT WHO WILL PURCHASE
import numpy as np
import pandas as pd

link2="C:/Users/user/Downloads/5_Ads_Success (1).csv"
df=pd.read_csv(link2)
print(df)
print(df.shape)
print(df.columns)
X=df.iloc[:,1:4].values
y=df.iloc[:,4].values

#HANDLING CATEGORICAL DATA

from sklearn.preprocessing import LabelEncoder
lc_x=LabelEncoder()
X[:,0]=lc_x.fit_transform(X[:,0])
print("after handling categorical data",X)

##PERFORMING EDA EXPLORATORY DATA ANALYSIS

import matplotlib.pyplot as plt
plt.scatter(df["Age"],df["EstimatedSalary"],color="red")
plt.show()

##diving into training and test data
import numpy
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=1)
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train[:,1:3] = sc.fit_transform(X_train[:,1:3])
X_test[:,1:3] = sc.fit_transform(X_test[:,1:3])



# This is a classification example as target variable y is about
# predicting Yes (1) or No (0) - customer will purchase or not

##MODEL BUILDING LOGISTIC REGRESSION
from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression(random_state=1)
classifier.fit(X_train,y_train)
y_pred=classifier.predict(X_test)

## EVALUATE OUR CLASSIFICATION ALGORITHM / MODEL
from sklearn import metrics
print("validation / test metric")
print("accuracy",metrics.accuracy_score(y_test,y_pred))
print("confusion matrix",metrics.confusion_matrix(y_test,y_pred))

### output
#accuracy 0.7875
#confusion matrix [[43  5]
# [12 20]]

### NOW WE PERFORM VISUALIZATION ###

from matplotlib.colors import ListedColormap
import numpy  as np
X_set,y_set=X_train[:,1 :3],y_train

#new classfier with two inputs in x axis
classifier.fit(X_train[:,1:3],y_train)

# X1 on X axis X2 on y axis
X1,X2=np.meshgrid(np.arange(start=X_set[:,0].min()-1,
                            stop=X_set[:,0].max()+1,step=0.01),
                  np.arange(start=X_set[:,1].min()-1,
                            stop=X_set[:,1].max()+1,step=0.01))
plt.contourf(X1,X2,
             classifier.predict(np.array([X1.ravel(),X2.ravel()]).T).reshape(X1.shape),alpha=0.75,cmap=ListedColormap(("red","green")))

## ploting the actual values (y_train) on messgrid
for i,j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set==j,0],X_set[y_set==j,1],color=ListedColormap(("red","green"))(i),label=j)
plt.title("CLASSIFICATION TRAINING")
plt.xlabel("AGE")
plt.ylabel("SALARY")
plt.show()
