#!/usr/bin/env python
# coding: utf-8

# In[25]:


pip install xgboost


# In[62]:


import os
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from pandas import Series, DataFrame
from termcolor import colored as cl
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rc("font", size=14)
plt.rcParams['axes.grid'] = True
plt.figure(figsize=(6,3))
plt.gray()
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import metrics
from sklearn.impute import MissingIndicator , SimpleImputer
from sklearn.preprocessing import PolynomialFeatures, KBinsDiscretizer, FunctionTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, LabelBinarizer, OrdinalEncoder
import statsmodels.formula.api as smf
import statsmodels.tsa as tsa
from sklearn.linear_model import LogisticRegression, LinearRegression, ElasticNet, Lasso, Ridge
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_graphviz
from sklearn.ensemble import BaggingClassifier, BaggingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, AdaBoostClassifier, AdaBoostRegressor
from sklearn.svm import LinearSVC, LinearSVR, SVC, SVR
from xgboost import XGBClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn import tree


# In[27]:


df = pd.read_csv(r"C:\Users\Lenovo\Desktop\data science projects\credit card fraud detection\creditcard.csv")


# In[33]:


df.head()


# In[38]:


total_transactions=len(df)
normal=len(df[df.Class==0])
fraudulent=len(df[df.Class==1])
fraud_percentage=round(fraudulent/normal*100,2)
print('Total number of transaction are ' + str(total_transactions))
print('Number of Normal Transactions are ' + str(normal))
print('Number of Fraudulent Transactions are ' + str(fraudulent))
print('Percentage of fraudulent Transactions is ' + str(fraud_percentage))


# In[39]:


df.info()


# In[42]:


min(df.Amount), max(df.Amount)


# In[48]:


sc = StandardScaler()
amount = df['Amount'].values
df['Amount'] = sc.fit_transform(amount.reshape(-1,1))
df.Amount


# In[51]:


df.drop(['Time'], axis=1, inplace=True)


# In[52]:


df.shape


# In[53]:


df.drop_duplicates(inplace=True)


# In[54]:


df.shape


# In[58]:


X = df.drop(['Class'], axis=1).values
y = df['Class'].values


# In[74]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state =1)


# In[75]:


###Decision Tree
DT = DecisionTreeClassifier(max_depth = 4, criterion='entropy')
DT.fit(X_train, y_train)
dt_yhat = DT.predict(X_test)


# In[76]:


fig = plt.figure(figsize=(30,30))
_ = tree.plot_tree(DT, filled=True)


# In[77]:


print('Accuracy Score of the decision tree is '+ str(accuracy_score(y_test, dt_yhat)))


# In[78]:


print('F1 score of the decision tree model is '+ str(f1_score(y_test, dt_yhat)))


# In[79]:


confusion_matrix(y_test, dt_yhat, labels = [0,1])


# In[85]:


###KNN
n = 7
KNN = KNeighborsClassifier(n_neighbors = n)
KNN.fit(X_train, y_train)
knn_yhat = KNN.predict(X_test)


# In[86]:


print('Accuracy score of the K-Nearest Neighbors model is {}'.format(accuracy_score(y_test, knn_yhat)))


# In[87]:


print('F1 score of the K-Nearest Neighbors model is {}'.format(f1_score(y_test, knn_yhat)))


# In[88]:


###Logistic Regression
lr = LogisticRegression()
lr.fit(X_train, y_train)
lr_yhat = lr.predict(X_test)


# In[89]:


print('Accuracy score of the Logistic Regression model is {}'.format(accuracy_score(y_test, lr_yhat)))


# In[90]:


print('F1 score of the Logistic Regression model is {}'.format(f1_score(y_test, lr_yhat)))


# In[91]:


###Support Vector Machines
svm = SVC()
svm.fit(X_train, y_train)
svm_yhat = svm.predict(X_test)


# In[92]:


print('Accuracy score of the Support Vector Machines model is {}'.format(accuracy_score(y_test, svm_yhat)))


# In[93]:


print('F1 score of the Support Vector Machines model is {}'.format(f1_score(y_test, svm_yhat)))


# In[94]:


###Random Forest regression
rf = RandomForestClassifier(max_depth = 4)
rf.fit(X_train, y_train)
rf_yhat = rf.predict(X_test)


# In[95]:


print('Accuracy score of the Random Forest model is {}'.format(accuracy_score(y_test, rf_yhat)))


# In[96]:


print('F1 score of the Random Forest model is {}'.format(f1_score(y_test, rf_yhat)))


# In[97]:


###XGBoost classifier
xgb = XGBClassifier(max_depth = 4)
xgb.fit(X_train, y_train)
xgb_yhat = xgb.predict(X_test)


# In[98]:


print('Accuracy score of the XGBoost model is {}'.format(accuracy_score(y_test, xgb_yhat)))


# In[99]:


print('F1 score of the XGBoost model is {}'.format(f1_score(y_test, xgb_yhat)))

