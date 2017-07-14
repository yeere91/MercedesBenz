from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score
import pandas as pd
import numpy as np
import xgboost as xgb
import time

time_start = time.time()

## import data
train = pd.read_csv('train_probed.csv')
test = pd.read_csv('test.csv')




# Apply LabelEncoder to categorical features
for c in train.columns:
    if train[c].dtype == 'object':
        lbl = LabelEncoder() 
        lbl.fit(list(train[c].values) + list(test[c].values)) 
        train[c] = lbl.transform(list(train[c].values))
        test[c] = lbl.transform(list(test[c].values))


## drop columns with only one value
k = train.loc[:,(train.apply(pd.Series.nunique) == 1)].columns.tolist()

train_drop = train.drop(k, axis = 1)
test_drop = test.drop(k, axis = 1)

# Try dropping non-binary variables, excluding ID?

# train_drop.drop(train_drop.columns[1:9], axis = 1, inplace = True)
# test_drop.drop(test_drop.columns[1:9], axis =1 , inplace = True)


## Adding in PCA, FA, etc.
from sklearn.decomposition import PCA
n_comp = 10
r_state = 2017

pca = PCA(n_components=n_comp, random_state = 2017)
pca2_results_train = pca.fit_transform(train_drop.drop(["y"], axis=1))
pca2_results_test = pca.transform(test_drop)


from sklearn.decomposition import FactorAnalysis

FA = FactorAnalysis(n_components=n_comp, random_state = r_state)
FA_results_train = FA.fit_transform(train_drop.drop(["y"], axis=1))
FA_results_test = FA.transform(test_drop)


from sklearn.decomposition import FastICA
from sklearn.decomposition import TruncatedSVD
from sklearn.random_projection import GaussianRandomProjection
from sklearn.random_projection import SparseRandomProjection


# tSVD
tsvd = TruncatedSVD(n_components=n_comp, random_state = r_state)
tsvd_results_train = tsvd.fit_transform(train_drop.drop(["y"], axis=1))
tsvd_results_test = tsvd.transform(test_drop)


# ICA
ica = FastICA(n_components=n_comp, random_state = r_state)
ica2_results_train = ica.fit_transform(train_drop.drop(["y"], axis=1))
ica2_results_test = ica.transform(test_drop)

# GRP
grp = GaussianRandomProjection(n_components=n_comp, eps=0.1, random_state = r_state)
grp_results_train = grp.fit_transform(train_drop.drop(["y"], axis=1))
grp_results_test = grp.transform(test_drop)

# SRP
srp = SparseRandomProjection(n_components=n_comp, dense_output=True, random_state = r_state)
srp_results_train = srp.fit_transform(train_drop.drop(["y"], axis=1))
srp_results_test = srp.transform(test_drop)

# Append decomposition components to datasets
for i in range(1, n_comp+1):
    train_drop['pca_' + str(i)] = pca2_results_train[:,i-1]
    test_drop['pca_' + str(i)] = pca2_results_test[:, i-1]
    
    #train_drop['fa' + str(i)] = FA_results_train[:,i-1]
    #test_drop['fa' + str(i)] = FA_results_test[:, i-1]

    train_drop['ica_' + str(i)] = ica2_results_train[:,i-1]
    test_drop['ica_' + str(i)] = ica2_results_test[:, i-1]

    #train_drop['tsvd_' + str(i)] = tsvd_results_train[:,i-1]
    #test_drop['tsvd_' + str(i)] = tsvd_results_test[:, i-1]

    #train_drop['grp_' + str(i)] = grp_results_train[:,i-1]
    #test_drop['grp_' + str(i)] = grp_results_test[:, i-1]

    #train_drop['srp_' + str(i)] = srp_results_train[:,i-1]
    #test_drop['srp_' + str(i)] = srp_results_test[:, i-1]


## My response variables
y_train = train["y"]
## My baseline prediction: an average of the y-values
y_mean = np.mean(y_train)





# Prepare dict of params for XGBoost to run with
xgb_params = {
    ## Number of Trees
    'n_trees': 395, 
    ## Learning Rate; default = 0.3
    'eta': 0.0065,
    ## Depth of Trees
    'max_depth': 3,
    ## Bagging 50% of the training set
    'subsample': 0.50,
    #'colsample_bytree': 0.75,
    'min_child_weight': 34,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    # Base Prediction = mean(target)
    'base_score': y_mean,
    'silent': 1
}

# Form DMatrices for Xgboost training
dtrain = xgb.DMatrix(train_drop.drop(["y"],axis=1), y_train)
dtest = xgb.DMatrix(test_drop)

# XGBoost, cross-validation
cv_result = xgb.cv(xgb_params, 
                   dtrain, 
                   num_boost_round=1500,
                   #nfold=10,
                   early_stopping_rounds=25,
                   verbose_eval=50, 
                   show_stdv=False,
                   seed=2017
                  )

num_boost_rounds = len(cv_result)
print(num_boost_rounds)

## Train model
model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=num_boost_rounds)

## Save the model
model.save_model('models/Decomp2.model')


# Check r2-score
from sklearn.metrics import r2_score

# Print out the r2-score
print("R2 Score on Training Set: {}".format(r2_score(dtrain.get_label(), model.predict(dtrain))))
print("Elapsed Time: {}".format(time.time() - time_start))

# Predict on the test set and save results.
y_pred = model.predict(dtest)
output = pd.DataFrame({'id': test['ID'].astype(np.int32), 'y': y_pred})
output.to_csv('xgboost_Decomp2.csv', index=False)

