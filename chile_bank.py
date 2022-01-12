# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 14:22:34 2022

@author: kange
"""

import pandas as pd
import seaborn as sns 
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.impute import SimpleImputer

df = pd.read_stata('banking.dta')


sns.histplot(x='B20', data=df, stat='probability')
df['B20'].describe()
df['B20'].value_counts(normalize = True)


rs = 16
col = ['B3', 'B4', 'B5', 'B26', 'B27', 'n_C1_1', 'n_C1_2', 'n_C1_3', 'n_C1_4','n_C1_5', 'n_C1_6',
      'n_C2_1','n_C2_2', 'n_C2_3', 'n_C2_4', 'n_C2_5', 'n_C2_6', 'n_C2_7', 'n_C2_8', 'n_C2_9', 'B10', 'F1', 'D1', 'J1','L8', 'B20']

logit = LogisticRegression(random_state = rs)
knn = KNN()
dt = DecisionTreeClassifier(random_state = rs)
rf = RandomForestClassifier(random_state = rs)

df = df.loc[df['B20'].notnull()]
df = df[col]
df['B20'] = df['B20'].map(dict(yes = 1, no = 0))
df['B3'] = df['B3'].astype('category')

y = df['B20']
X = df.drop('B20', axis = 1)
X = pd.get_dummies(X, drop_first=True)

x_train, x_test, y_train, y_test = train_test_split(X, y, stratify = y, random_state = rs)

# impute missing values
imp = SimpleImputer(strategy = 'most_frequent')
x_train_imputed = pd.DataFrame(imp.fit_transform(x_train))
x_test_imputed = pd.DataFrame(imp.transform(x_test))
x_train_imputed.columns = x_train.columns 
x_test_imputed.columns = x_test.columns

for model in [logit, dt, rf]:
    model.fit(x_train_imputed, y_train)
    pred = model.predict_proba(x_test_imputed)
    print(f"The auc of {model} is {roc_auc_score(y_test, pred[:, 1])}")

# hyperparameter tuning for knn 
param_grid = {'n_neighbors': np.arange(1, 100)}
knn_cv = GridSearchCV(knn, param_grid, cv=5)
knn_cv.fit(x_train_imputed, y_train)

# which is the best number of neighbors to use
knn_cv.best_params_
knn_cv.best_score_

# use 2 as the number of neighbors
knn_2 = KNN(n_neighbors = 2)
knn_2.fit(x_train_imputed, y_train)
knn_pred = knn_2.predict_proba(x_test_imputed)
print(f"The auc of KNN with 2 neighbors is {roc_auc_score(y_test, knn_pred[:, 1])}")


logit_model = logit.fit(x_train_imputed, y_train)
print(logit_model.coef_)
    # look at importance by a
importance = pd.DataFrame(logit_model.coef_[0], index = x_train_imputed.columns)
importance.columns = ['contribution']
importance.sort_values(by='contribution', key=abs, ascending = False)


from treeinterpreter import treeinterpreter as ti

dt_clf = dt.fit(x_train_imputed, y_train)
prediction, bias, contributions = ti.predict(dt_clf, x_test)

for c, feature in zip(contributions[90], x_test_imputed.columns):
    print(feature, c)

# 90th is n_C1_2
importance.iloc[90]
contributions[90]
contributions[90][:, 0]

married_no_bank = pd.DataFrame(contributions[90][:, 0], index = x_test_imputed.columns)
married_no_bank.columns = ['contribution']
married_no_bank.sort_values(by='contribution', key=abs, ascending = False)
