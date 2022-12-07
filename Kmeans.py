import csv
import numpy as np
import pandas as pd
#from minisom import MiniSom    
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, SelectPercentile, chi2, mutual_info_regression, f_regression, VarianceThreshold
from scipy.spatial import distance
from sklearn.manifold import MDS
from sklearn.cluster import AgglomerativeClustering
from sklearn import datasets, cluster
import csv
import numpy as np
import pandas as pd
from minisom import MiniSom    
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
import re
from sklearn.decomposition import PCA
from sklearn import manifold
from sklearn.manifold import locally_linear_embedding
from sklearn.manifold import MDS
from sklearn.cluster import FeatureAgglomeration
from sklearn.random_projection import SparseRandomProjection
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression

X_lib = {}
X_total = []
X_label = []
y_lib = {}

	
with open('ASO_AEIF_combined_desc_filter.csv','r') as csvfile:
	csv_reader = csv.reader(csvfile, delimiter=',')
	for row in csv_reader:
		X_lib[row[0]] = [float(x) for x in row[1:-1]]
		X_total.append(X_lib[row[0]])
		X_label.append(row[0])

print (X_label)
print(len(X_label))
arr_X = np.array(X_total)

sel = VarianceThreshold(threshold=0.015)
arr_X = sel.fit_transform(arr_X)

pca = PCA(n_components=20)
arr_X = pca.fit_transform(arr_X)




nf = open('kclusterrandomstatetest.csv', 'w')
kmeans = KMeans(n_clusters=21, random_state=9, max_iter = 1000, n_init=25).fit(arr_X)

reps = []


for center in kmeans.cluster_centers_:
	closest = None
	mindist = None
	clusternumber = None
	for i in range(len(arr_X)):
		dist = distance.euclidean(arr_X[i],center)
		if closest == None:
			closest = X_label[i]
			mindist = dist
		else:
			if dist < mindist:
				mindist = dist
				closest = X_label[i]
			else:
				pass
				
	reps.append(closest)
	
for line in open('ASO_AEIF_combined_desc_filter.csv','r').readlines():
	if line.split(',')[0] in reps:
		nf.write(line)
