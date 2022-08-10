pip install xgboost

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
df = pd.read_csv(r"C:\Users\Lenovo\Desktop\data science projects\credit card fraud detection\creditcard.csv")
df.head()
total_transactions=len(df)
normal=len(df[df.Class==0])
fraudulent=len(df[df.Class==1])
fraud_percentage=round(fraudulent/normal*100,2)
print('Total number of transaction are ' + str(total_transactions))
print('Number of Normal Transactions are ' + str(normal))
print('Number of Fraudulent Transactions are ' + str(fraudulent))
print('Percentage of fraudulent Transactions is ' + str(fraud_percentage))
df.info()
min(df.Amount), max(df.Amount)
sc = StandardScaler()
amount = df['Amount'].values
df['Amount'] = sc.fit_transform(amount.reshape(-1,1))
df.Amount
df.drop(['Time'], axis=1, inplace=True)
df.shape
df.drop_duplicates(inplace=True)
df.shape
X = df.drop(['Class'], axis=1).values
y = df['Class'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state =1)

###Decision Tree
DT = DecisionTreeClassifier(max_depth = 4, criterion='entropy')
DT.fit(X_train, y_train)
dt_yhat = DT.predict(X_test)

fig = plt.figure(figsize=(30,30))
_ = tree.plot_tree(DT, filled=True)

print('Accuracy Score of the decision tree is '+ str(accuracy_score(y_test, dt_yhat)))

print('F1 score of the decision tree model is '+ str(f1_score(y_test, dt_yhat)))

confusion_matrix(y_test, dt_yhat, labels = [0,1])

###KNN
n = 7
KNN = KNeighborsClassifier(n_neighbors = n)
KNN.fit(X_train, y_train)
knn_yhat = KNN.predict(X_test)

print('Accuracy score of the K-Nearest Neighbors model is {}'.format(accuracy_score(y_test, knn_yhat)))

print('F1 score of the K-Nearest Neighbors model is {}'.format(f1_score(y_test, knn_yhat)))

###Logistic Regression
lr = LogisticRegression()
lr.fit(X_train, y_train)
lr_yhat = lr.predict(X_test)

print('Accuracy score of the Logistic Regression model is {}'.format(accuracy_score(y_test, lr_yhat)))

print('F1 score of the Logistic Regression model is {}'.format(f1_score(y_test, lr_yhat)))

###Support Vector Machines
svm = SVC()
svm.fit(X_train, y_train)
svm_yhat = svm.predict(X_test)

print('Accuracy score of the Support Vector Machines model is {}'.format(accuracy_score(y_test, svm_yhat)))

print('F1 score of the Support Vector Machines model is {}'.format(f1_score(y_test, svm_yhat)))

###Random Forest regression
rf = RandomForestClassifier(max_depth = 4)
rf.fit(X_train, y_train)
rf_yhat = rf.predict(X_test)

print('Accuracy score of the Random Forest model is {}'.format(accuracy_score(y_test, rf_yhat)))
print('F1 score of the Random Forest model is {}'.format(f1_score(y_test, rf_yhat)))

###XGBoost classifier
xgb = XGBClassifier(max_depth = 4)
xgb.fit(X_train, y_train)
xgb_yhat = xgb.predict(X_test)

print('Accuracy score of the XGBoost model is {}'.format(accuracy_score(y_test, xgb_yhat)))
print('F1 score of the XGBoost model is {}'.format(f1_score(y_test, xgb_yhat)))
