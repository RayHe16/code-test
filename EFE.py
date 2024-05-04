import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneOut
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.metrics import mean_absolute_error
data_mat = []
label_mat = []
data = pd.read_excel(r'C:\data.xlsx')
data_mat = data.iloc[0:, 0:].values
label_mat = data.iloc[0:, :-1].values
data_mat = np.array(data_mat)
label_mat = np.array(label_mat)
X=data_mat;
Y=label_mat;

def looscore(X,Y, model):
    loo = LeaveOneOut()
    pre_array = []
    Y_tests = []
    for train_index, test_index in loo.split(X):
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
        Y_tests.append(Y_test)
        model.fit(X_train, Y_train)
        pre_array.append(model.predict(X_test))

    score = r2_score(Y_tests, pre_array)
    return score
def loormse(X,Y, model):
    loo = LeaveOneOut()
    pre_array = []
    Y_tests = []
    for train_index, test_index in loo.split(X):
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
        Y_tests.append(Y_test)
        model.fit(X_train, Y_train)
        pre_array.append(model.predict(X_test))

    test_rootmean_squared_error = np.sqrt(mean_squared_error(Y_tests, pre_array))
    return test_rootmean_squared_error
def loomae(X,Y, model):
    loo = LeaveOneOut()
    pre_array = []
    Y_tests = []
    for train_index, test_index in loo.split(X):
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
        Y_tests.append(Y_test)
        model.fit(X_train, Y_train)
        pre_array.append(model.predict(X_test))

    
    test_rootmean_squared_error = mean_absolute_error(Y_tests, pre_array)
    return test_rootmean_squared_error
svr = SVR(kernel='rbf')
from itertools import combinations
num_features = X.shape[1]
Xcopy = np.copy(X)
while num_features >= 1:
    feat = []
    for i in range(X.shape[1]):
        feat.append(i)
    scores = []
    bestscore = 0
    rmses = []
    bestrmse = 100
    maes = []
    bestmae = 100
    for combination in combinations(feat, num_features):
        X_subset = Xcopy[:,combination]
        score = looscore(X_subset, Y, svr)
        scores.append(score)
        if score > bestscore:
            bestscore = score
            bestfeat = combination
        rmse = loormse(X_subset, Y, svr)
        rmses.append(rmse)
        if rmse < bestrmse:
            bestrmse = rmse
            bestfeat = combination
        mae = loomae(X_subset, Y, svr)
        maes.append(mae)
        if mae < bestmae:
            bestmae = mae
            bestfeat = combination
            print('fetures：',bestfeat,'R2：',bestscore)
            print('fetures：', bestfeat, 'RMSE：', bestrmse)
            num_features -=1


