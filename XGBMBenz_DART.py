
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score
import pandas as pd
import numpy as np
import xgboost as xgb


## import data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Apply LabelEncoder to categorical features
for c in train.columns:
    if train[c].dtype == 'object':
        lbl = LabelEncoder() 
        lbl.fit(list(train[c].values) + list(test[c].values)) 
        train[c] = lbl.transform(list(train[c].values))
        test[c] = lbl.transform(list(test[c].values))

## My response variables
y_train = train["y"]
## My baseline prediction: an average of the y-values
y_mean = np.mean(y_train)

# Prepare dict of params for XGBoost to run with
xgb_params = {
    'booster': 'dart',
    'n_trees': 500, 
    'eta': 0.005,
    'max_depth': 4,
    'subsample': 0.95,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'base_score': y_mean, # base prediction = mean(target)
    'silent': 1
}

# Form DMatrices for Xgboost training
dtrain = xgb.DMatrix(train.drop(["y"],axis=1), y_train)
dtest = xgb.DMatrix(test)

# XGBoost, cross-validation
cv_result = xgb.cv(xgb_params, 
                   dtrain, 
                   num_boost_round=10000,
                   nfold=10,
                   early_stopping_rounds=50,
                   verbose_eval=50, 
                   show_stdv=False,
                   seed=2017
                  )

num_boost_rounds = len(cv_result)
print(num_boost_rounds)

## Train model
model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=num_boost_rounds)

## Save the model
model.save_model('models/FA.model')

# Print out the r2-score
print(r2_score(dtrain.get_label(), model.predict(dtrain)))