


import numpy as np
from sklearn.model_selection import LeaveOneOut
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from sklearn.metrics import r2_score


data_mat = []
label_mat = []
data = pd.read_excel(r'C:\data.xlsx')
data_mat = data.iloc[0:, 0:].values
label_mat = data.iloc[0:, :-1].values
data_mat = np.array(data_mat)
label_mat = np.array(label_mat)
X=data_mat;
y=label_mat;
loo = LeaveOneOut()
clf = SVR()
predictions = []
true=[]
for train_index, test_index in loo.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    predictions.append(y_pred)
    true.append(y_test)
print("r2:",r2_score(true,predictions))
print("rootmean_squared_error:",np.sqrt(mean_squared_error(true,predictions)))




