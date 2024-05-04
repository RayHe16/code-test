import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV 
from sklearn.svm import SVR
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from xgboost.sklearn import XGBRegressor
data_mat = []
label_mat = []
data = pd.read_excel(r'C:\data.xlsx')
data_mat = data.iloc[0:, 0:].values
label_mat = data.iloc[0:, :-1].values
data_mat = np.array(data_mat)
label_mat = np.array(label_mat)
X=data_mat;
Y=label_mat;
c_can = np.logspace(-4, 4, 100)
gamma_can = np.logspace(-4, 4, 100)

model=SVR(kernel='rbf')
gs = GridSearchCV(model,param_grid={'C': c_can, 'gamma': gamma_can},cv=5)  
parameters = {'n_estimators':[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,20,50,60,70],
              'max_depth':[1,2,3,4,5,6,7,8],
              'learning_rate':[0.01,0.1,0.2,0.3,0.4,0.5],
              'gamma':[0.0001,0.00001,0.001,0.01, ],
              'random_state':[0]}
model=XGBRegressor()
gs = GridSearchCV(model,param_grid = parameters,cv=5)    
parameters = {'n_estimators':[1],'max_depth':[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17],
              'random_state':[0,1,2,3,4,5,6,7,8,9,10]}
model=RandomForestRegressor()
gs = GridSearchCV(model,param_grid = parameters,cv=5)   
gs.fit(X, Y)
Y_pre=gs.predict(X)
print("r2:",r2_score(Y,Y_pre))
print("rootmean_squared_error:",np.sqrt(mean_squared_error(Y,Y_pre)))



