import numpy as np
from sklearn.base import BaseEstimator,TransformerMixin, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import pandas as pd
from sklearn.linear_model import ElasticNetCV, LassoLarsCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import make_pipeline, make_union
from sklearn.utils import check_array
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.random_projection import GaussianRandomProjection
from sklearn.random_projection import SparseRandomProjection
from sklearn.decomposition import PCA, FastICA
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import r2_score
import time


class StackingEstimator(BaseEstimator, TransformerMixin):
    
    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y=None, **fit_params):
        self.estimator.fit(X, y, **fit_params)
        return self
    def transform(self, X):
        X = check_array(X)
        X_transformed = np.copy(X)
        # add class probabilities as a synthetic feature
        if issubclass(self.estimator.__class__, ClassifierMixin) and hasattr(self.estimator, 'predict_proba'):
            X_transformed = np.hstack((self.estimator.predict_proba(X), X))

        # add class prodiction as a synthetic feature
        X_transformed = np.hstack((np.reshape(self.estimator.predict(X), (-1, 1)), X_transformed))

        return X_transformed


time_start = time.time()

## import data
train = pd.read_csv('train_fudged.csv')
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


#save columns list before adding the decomposition components

usable_columns = list(set(train.columns) - set(['y']))


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

id_test = test['ID'].values

#finaltrainset and finaltestset are data to be used only the stacked model (does not contain PCA, SVD... arrays) 
finaltrainset = train[usable_columns].values
finaltestset = test[usable_columns].values


'''Train the xgb model then predict the test data'''

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
# NOTE: Make sure that the class is labeled 'class' in the data file

dtrain = xgb.DMatrix(train.drop('y', axis=1), y_train)
dtest = xgb.DMatrix(test)



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



y_pred = model.predict(dtest)

'''Train the stacked models then predict the test data'''

stacked_pipeline = make_pipeline(
    StackingEstimator(estimator=LassoLarsCV(normalize=True)),
    StackingEstimator(estimator=GradientBoostingRegressor(learning_rate=0.001, loss="huber", max_depth=3, max_features=0.55, min_samples_leaf=18, min_samples_split=14, subsample=0.7)),
    LassoLarsCV()

)


stacked_pipeline.fit(finaltrainset, y_train)
results = stacked_pipeline.predict(finaltestset)

'''R2 Score on the entire Train data when averaging'''

print('R2 Score on Training Set:')
print(r2_score(y_train,stacked_pipeline.predict(finaltrainset)*0.2855 + model.predict(dtrain)*0.7145))
print('Time Elapsed: {}'.format(time.time() - time_start))
'''Average the preditionon test data  of both models then save it on a csv file'''

sub = pd.DataFrame()
sub['ID'] = id_test
sub['y'] = y_pred*0.75 + results*0.25
sub.to_csv('xgboost_Stacked2_Fudged.csv', index=False)
