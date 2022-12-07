import os
import math
import numpy as np
import scipy.stats as stats
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler, MaxAbsScaler, MinMaxScaler, Binarizer, Normalizer, PolynomialFeatures
from sklearn.feature_selection import SelectKBest, SelectPercentile, chi2, mutual_info_regression, f_regression, VarianceThreshold
from sklearn import linear_model, tree, kernel_ridge, cross_decomposition
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso, LassoCV
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import ElasticNetCV
from sklearn.datasets import make_regression
from glob import glob
import re
from sklearn.feature_selection import RFE
from sklearn.svm import LinearSVR, NuSVR, OneClassSVM
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import NuSVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.decomposition import PCA
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import ElasticNetCV
from sklearn.datasets import make_regression
import statistics
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import SelectKBest
from numpy import random
import os
import math
import numpy as np
import scipy.stats as stats
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler, MaxAbsScaler, MinMaxScaler, Binarizer, Normalizer, PolynomialFeatures
from sklearn.feature_selection import SelectKBest, SelectPercentile, chi2, mutual_info_regression, f_regression, VarianceThreshold
from sklearn import linear_model, tree, kernel_ridge, cross_decomposition
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso, LassoCV
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import ElasticNetCV
from sklearn.datasets import make_regression
from glob import glob
import re
from sklearn.feature_selection import RFE
from sklearn.svm import LinearSVR, NuSVR, OneClassSVM
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import NuSVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.decomposition import PCA
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import ElasticNetCV
from sklearn.datasets import make_regression
import statistics
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import SelectKBest
from numpy import random
from sklearn.manifold import MDS
from sklearn.kernel_ridge import KernelRidge
from sklearn import manifold





data = pd.read_csv("split0.csv", header=None, index_col=0)
catdesc = pd.read_csv("catdesc.csv", header=None, index_col=0)
subdesc = pd.read_csv("subdesc.csv", header=None, index_col=0)


# Step 1. Concatenate ALL fucking descriptors
concatalld = catdesc

concat_all_d_no_nan = concatalld.dropna()


# Step 2. Create a list of entries where we have data
all_labels = concat_all_d_no_nan.index
have_data_labels = data.index

# Step 3. Create a list of entries where we DON'T HAVE DATA
dont_have_data_labels = [x for x in all_labels if x not in data.index]



features_with_data = concat_all_d_no_nan.loc[have_data_labels] #this is X
#features_with_shit =  concatalld.loc[dont_have_data_labels] #this is going to be X predicted
features_with_predictions = concat_all_d_no_nan.loc[dont_have_data_labels]

X = features_with_data
Xp = features_with_predictions
y = data

print("Features:VT 0.0", X.shape)

selector = VarianceThreshold(0.134444)
X =selector.fit_transform(X)
Xp = selector.transform(Xp)

print("Features:VT 0.0", X.shape)


sc = StandardScaler()
X = sc.fit_transform(X)
Xp = sc.transform(Xp)


#################################################################################
#								Start Final Models                              #
#################################################################################


selector = manifold.LocallyLinearEmbedding(n_neighbors=14, n_components=5)
X = selector.fit_transform(X)
Xp =selector.transform(Xp)

model = KernelRidge(alpha=0.000167, kernel='poly', degree = 3).fit(X, y)
print("model type kernal ridge, kernel = rbf")

# model = PLSRegression(n_components =3).fit(X,y)
# print('model-type-PLS')

y_p = model.predict(X).ravel()
yp_p = model.predict(Xp).ravel()
print(y_p)
print(yp_p)
print(y, y[1].values)

##################################################################################
#								WRITE TO FILE 									 #
##################################################################################
print((sum(cross_val_score(model, X, y, cv=5))/5))
print(cross_val_score(model, X, y, cv=5))

export = {
	"y_predict" : y_p,
	"y_observed" : y[1].values,
	"error": y[1].values - y_p,	
}

export_df = pd.DataFrame(export, index=features_with_data.index)

export_pp = {
	"y_predict" : yp_p,
}

export_pp_df = pd.DataFrame(export_pp, index=features_with_predictions.index)

export_df.to_csv('model_out_test_split0.csv')
export_pp_df.to_csv('in_silico_pred_split0.csv')


