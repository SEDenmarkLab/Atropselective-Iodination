import csv
import numpy as np
import pandas as pd
import csv
#from minisom import MiniSom    
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn import metrics
from scipy.spatial.distance import cdist
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, SelectPercentile, chi2, mutual_info_regression, f_regression, VarianceThreshold

X_lib = {}
X_total = []
X_label = []
with open('ASO_AEIF_combined_desc_filter.csv','r') as csvfile:
	csv_reader = csv.reader(csvfile, delimiter=',')
	for row in csv_reader:
		X_lib[row[0]] = [float(x) for x in row[1:-1]]
		X_total.append(X_lib[row[0]])
		X_label.append(row[0])
arr_X = np.asarray(X_total)

sel = VarianceThreshold(threshold=0.015)
arr_X = sel.fit_transform(arr_X)

pca = PCA(n_components=20)
arr_X = pca.fit_transform(arr_X)

distortions = []
K = range(1,40)
for k in K:
	kmeanModel = KMeans(n_clusters=k, random_state=0, n_jobs=50, n_init=1000).fit(arr_X)
	kmeanModel.fit(arr_X)
	distortions.append(np.sum(np.min(cdist(arr_X, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / arr_X.shape[0])
	print(k)
print(KMeans)
print(distortions)
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
#plt.xticks(0, 16, 2)
plt.ylabel('Distortion')
plt.title('Elbow Plot')
plt.savefig('Elbow_plot.png')



