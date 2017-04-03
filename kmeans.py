import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from scipy.spatial.distance import cdist, pdist
from sklearn.cluster import KMeans
CHARACTER_DATA_FILE = '../data_sets/letter_recognition/letter_recognition.csv'
char_data = pd.read_csv(CHARACTER_DATA_FILE)
char_x = char_data[[x for x in char_data.columns if x != 'lettr']]
WINE_DATA_FILE = '../data_sets/wine/winequality.csv'
wine_data = pd.read_csv(WINE_DATA_FILE)
wine_x = wine_data[[x for x in wine_data.columns if x != '"quality"']]
K = range(1, 40)
KM_c = [KMeans(n_clusters=k).fit(char_x) for k in K]
KM_w = [KMeans(n_clusters=k).fit(wine_x) for k in K]
print("Trained kmean models")
centroids_c = [km.cluster_centers_ for km in KM_c]
centroids_w = [km.cluster_centers_ for km in KM_w]
print("Found the centroids")
Dk_c = [cdist(char_x, center, 'euclidean') for center in centroids_c]
Dk_w = [cdist(wine_x, center, 'euclidean') for center in centroids_w]
print("Calculated euclidean distance")
cIdx_c = [np.argmin(D, axis=1) for D in Dk_c]
dist_c = [np.min(D, axis=1) for D in Dk_c]
avgWithinSS_c = [sum(d) / char_x.shape[0] for d in dist_c]
# Total with-in sum of square
wcss_c = [sum(d**2) for d in dist_c]
tss_c = sum(pdist(char_x)**2) / char_x.shape[0]
bss_c = tss_c - wcss_c
cIdx_w = [np.argmin(D, axis=1) for D in Dk_w]
dist_w = [np.min(D, axis=1) for D in Dk_w]
avgWithinSS_w = [sum(d) / char_x.shape[0] for d in dist_w]
# Total with-in sum of square
wcss_w = [sum(d**2) for d in dist_w]
tss_w = sum(pdist(char_x)**2) / char_x.shape[0]
bss_w = tss_w - wcss_w
print("Calculated sum of square errors")
kIdx_c = 9
kIdx_w = 4
plt.style.use('ggplot')
# elbow curve
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(K, avgWithinSS_c, '*-', label='Letter Recognition')
ax.plot(K[kIdx_c], avgWithinSS_c[kIdx_c], marker='o', markersize=12,
        markeredgewidth=2, markeredgecolor='r', markerfacecolor='None')
ax.plot(K, avgWithinSS_w, '*-', label='Wine Quality')
ax.plot(K[kIdx_w], avgWithinSS_w[kIdx_w], marker='o', markersize=12,
        markeredgewidth=2, markeredgecolor='b', markerfacecolor='None')
plt.grid(True)
plt.xlabel('Number of clusters')
plt.ylabel('Average within-cluster sum of squares')
plt.legend(loc='best')
plt.title('Elbow for KMeans clustering')
# fig.savefig('graphs/kmeans/elbow1.png')
plt.show()
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(K, bss_c / tss_c * 100, '*-', label='Letter Recognition')
ax.plot(K, bss_w / tss_w * 100, '*-', label='Wine Quality')
plt.grid(True)
plt.xlabel('Number of clusters')
plt.ylabel('Percentage of variance explained')
plt.legend(loc='best')
plt.title('Elbow for KMeans clustering')
# fig.savefig('graphs/kmeans/elbow2.png')
plt.show()