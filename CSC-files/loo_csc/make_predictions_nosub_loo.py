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


FILE_PATTERN = re.compile(r"(?P<text>[a-zA-Z_]+)(?P<number>[0-9]+)\.(?P<ext>[a-zA-Z]+)")

LARGE_DATAFRAME_CACHE = {}

def make_models(name, cat_descriptor_fn, data_fn, test_fn):
	"""
		name is a name of the model

		data_fn and test_fn MUST match file_pattern. otherwise you messed up.
	"""
	print("="*80)
	print("making models out of the following files:")
	print("model name:", name)
	print("data:", data_fn)
	print("test:", test_fn)
	print("="*80)

	data_file_match = FILE_PATTERN.match(os.path.basename(data_fn))
	test_file_match = FILE_PATTERN.match(os.path.basename(test_fn))


	if not ( data_file_match and test_file_match ):
		print("You messed up.")
		print("Data file or test file doesn't match the file pattern")
		exit(1)

	data_n = int(data_file_match['number'])
	test_n = int(test_file_match['number'])

	assert data_n == test_n, "Your train and test splits seem to have different numbers. error."

	if cat_descriptor_fn in LARGE_DATAFRAME_CACHE:
		catdesc = LARGE_DATAFRAME_CACHE[cat_descriptor_fn]
	
	else:
		with open(cat_descriptor_fn) as cat_desc_file:
			catdesc = pd.read_csv(cat_desc_file, header = None, index_col=0)
		LARGE_DATAFRAME_CACHE[cat_descriptor_fn] = catdesc

	with open(data_fn) as data_file:
		data = pd.read_csv(data_file, header=None, index_col=0)

	with open(test_fn) as test_file:
		test = pd.read_csv(test_file, header=None, index_col=0)



	concatalld = catdesc

	concat_all_d_no_nan = concatalld.dropna()


	all_labels = concat_all_d_no_nan.index
	have_data_labels = data.index
	test_set = test.index

	dont_have_data_labels = [x for x in all_labels if x not in data.index]



	features_with_data = concat_all_d_no_nan.loc[have_data_labels] #this is X
	features_with_test_data = concat_all_d_no_nan.loc[test_set] #this is X
	print(features_with_test_data)

	features_with_predictions = concat_all_d_no_nan.loc[dont_have_data_labels]

	X = features_with_data
	Xp = features_with_test_data
	
	y = data
	yd = test
	

	print("Features:VT 0.0", X.shape)

	selector = VarianceThreshold(0.01)
	X =selector.fit_transform(X)
	Xp = selector.transform(Xp)

	print("Features:VT 0.0", X.shape)


	sc = StandardScaler()
	X = sc.fit_transform(X)
	Xp = sc.transform(Xp)

	
	#################################################################################
	#								Start Final Models                              #
	#################################################################################


	selector = manifold.LocallyLinearEmbedding(n_neighbors=14, n_components=13)
	X = selector.fit_transform(X)
	Xp =selector.transform(Xp)

	model = KernelRidge(alpha=0.01, kernel='poly', degree = 4 ).fit(X, y)
	print("model type kernal ridge, kernel = poly")

	#model = PLSRegression(n_components =2).fit(X,y)
	#print('model-type-PLS')

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
		"y_observed" : yd[1].values,
		"error": yd[1].values - yp_p,	
	}

	export_pp_df = pd.DataFrame(export_pp, index=features_with_test_data.index)


	
	export_df.to_csv(f'model_out_{data_n}.csv') #need away to track the split number
	export_pp_df.to_csv(f'in_silico_pred_{data_n}.csv') #need away to track the split number

if __name__ == '__main__':
# set up an iterator over files
	cat_descriptor_files = "catdesc.csv"
	data_files = glob("split0_loo_train/*.csv")
	test_files = glob("split0_loo_test/*.csv")

	# cat_descriptor_files.sort()
	data_files.sort()
	print("data files:", data_files)
	test_files.sort()
	print("test files:", test_files)

	#DEBUG
	for f in data_files:
		print(f"Processing {f}")
		m = re.match(FILE_PATTERN, os.path.basename(f))
		print(f"\ttext: {m['text']}, number: {int(m['number'])}, ext:{m['ext']}")
	

	for i, data_file in enumerate(data_files):
		make_models(i, cat_descriptor_files, data_file, test_files[i])
