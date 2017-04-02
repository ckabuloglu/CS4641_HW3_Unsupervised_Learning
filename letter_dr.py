import csv
import numpy as np
import pandas as pd
from collections import defaultdict
from numpy.testing import assert_array_almost_equal
import matplotlib.pyplot as plt 
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from sklearn.feature_selection import VarianceThreshold
from sklearn.random_projection import GaussianRandomProjection
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

import seaborn as sns

C = ['#ff400e',
'#fc5819',
'#f87124',
'#f5892f',
'#f2a23a',
'#eeba45',
'#ebd250',
'#d1dd5e',
'#afe36d',
'#8dea7d',
'#6bf08c',
'#49f69b',
'#27fdaa',
'#17f8b8',
'#19e8c4',
'#1bd8d1',
'#1cc8dd',
'#1eb8ea',
'#20a8f6',
'#2496fb',
'#307eea',
'#3c66d9',
'#484ec7',
'#5436b6',
'#601ea4',
'#6b0693']

sns.set(color_codes=True)

df = pd.read_csv("letter.csv")

label = 'lettr'
# Train set
df_y = df[label]
df_x = df[[x for x in df.columns if label not in x]]

##########################################################################

reduced_data = PCA(n_components=2).fit_transform(df_x)
kmeans = KMeans(init='k-means++', n_clusters=26, n_init=26)
kmeans.fit(reduced_data)

df_x.dropna()

print "PCA - kmeans"
for n in range (2,17,2):
    pca = PCA(n_components=n)
    pca.fit(df_x)

    # print "Score:", pca.score(df)
    print "N:", n
    print "Variance:", pca.noise_variance_

    reduced_data = PCA(n_components=n).fit_transform(df_x)
    kmeans = KMeans(init='k-means++', n_clusters=26, n_init=26)
    kmeans.fit(reduced_data)

    correct = 0
    for i in range(26):
        d = defaultdict(int)
        for index, row in df.iterrows():
            if row[label] == i:
                lab = kmeans.predict([reduced_data[index]])
                d[lab[0]] += 1
        correct += max(d.values())
    
    print "Accuracy:", float(correct) / 20000

print "PCA - EM"
for n in range (2,17,2):
    pca = PCA(n_components=n)
    pca.fit(df_x)

    # print "Score:", pca.score(df)
    print "N:", n
    print "Variance:", pca.noise_variance_

    reduced_data = PCA(n_components=n).fit_transform(df_x)
    em = GaussianMixture(n_components=26)
    em.fit(reduced_data)

    correct = 0
    for i in range(26):
        d = defaultdict(int)
        for index, row in df.iterrows():
            if row[label] == i:
                lab = em.predict([reduced_data[index]])
                d[lab[0]] += 1
        correct += max(d.values())
    
    print "Accuracy:", float(correct) / 20000


# Graph
h = .02

# Plot the decision boundary. For that, we will assign a color to each
x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

fig = plt.figure(1)
plt.clf()

my_cmap = plt.cm.get_cmap('gist_ncar')
my_cmap.set_under('w')

plt.imshow(Z, interpolation='nearest',
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap=my_cmap,
           aspect='auto', origin='lower')

plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)

centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=169, linewidths=3, color='w', zorder=10)

plt.title('K-means clustering on the  dataset (PCA-reduced data)')

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
fig.savefig('figures/letter_km_PCA.png')
plt.close(fig)

##########################################################################

fig = plt.figure(2)
plt.clf()

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)

for index, row in df.iterrows():
    plt.plot(reduced_data[index, 0], reduced_data[index, 1], '.', color=C[int(row[label])], markersize=2)

plt.xticks(())
plt.yticks(())

plt.title('Each label mapped on the PCA-reduced 2D graph')
fig.savefig('figures/letter_km_PCA_letters.png')
plt.close(fig)

# ##########################################################################

print "ICA - kmeans"
for n in range (2,17,2):

    # print "Score:", pca.score(df)
    print "N:", n

    reduced_data = FastICA(n_components=n).fit_transform(df_x)
    kmeans = KMeans(init='k-means++', n_clusters=26, n_init=26)
    kmeans.fit(reduced_data)

    correct = 0
    for i in range(26):
        d = defaultdict(int)
        for index, row in df.iterrows():
            if row[label] == i:
                lab = kmeans.predict([reduced_data[index]])
                d[lab[0]] += 1
        correct += max(d.values())
    
    print "Accuracy:", float(correct) / 20000

print "ICA - EM"
for n in range (2,17,2):

    # print "Score:", pca.score(df)
    print "N:", n

    reduced_data = FastICA(n_components=n).fit_transform(df_x)
    em = GaussianMixture(n_components=26)
    em.fit(reduced_data)

    correct = 0
    for i in range(26):
        d = defaultdict(int)
        for index, row in df.iterrows():
            if row[label] == i:
                lab = em.predict([reduced_data[index]])
                d[lab[0]] += 1
        correct += max(d.values())
    
    print "Accuracy:", float(correct) / 20000

reduced_data = FastICA(n_components=2).fit_transform(df_x)
kmeans = KMeans(init='k-means++', n_clusters=26, n_init=26)
kmeans.fit(reduced_data)

# Plot the decision boundary. For that, we will assign a color to each
x_min, x_max = reduced_data[:, 0].min() - 0.01, reduced_data[:, 0].max() + 0.01
y_min, y_max = reduced_data[:, 1].min() - 0.01, reduced_data[:, 1].max() + 0.01

fig = plt.figure(3)
plt.clf()

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)

for index, row in df.iterrows():
    col = kmeans.predict([reduced_data[index,:]])
    plt.plot(reduced_data[index, 0], reduced_data[index, 1], '.', color=C[col[0]], markersize=3)

plt.title('K-means clustering on the  dataset (ICA-reduced data)')

plt.xticks(())
plt.yticks(())
fig.savefig('figures/letter_km_ICA.png')
plt.close(fig)

##########################################################################

fig = plt.figure(4)
plt.clf()

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)

for index, row in df.iterrows():
    plt.plot(reduced_data[index, 0], reduced_data[index, 1], '.', color=C[int(row[label])], markersize=3)

plt.xticks(())
plt.yticks(())

plt.title('Each label mapped on the ICA-reduced 2D graph')
fig.savefig('figures/letter_km_ICA_letters.png')

plt.close(fig)

##########################################################################

print "RP - kmeans"
for n in range (2,17,2):
    transformer = GaussianRandomProjection(n_components=n)
    reduced_data = transformer.fit_transform(df_x)

    print "N:", n

    kmeans = KMeans(init='k-means++', n_clusters=26, n_init=26)
    kmeans.fit(reduced_data)

    correct = 0
    for i in range(26):
        d = defaultdict(int)
        for index, row in df.iterrows():
            if row[label] == i:
                lab = kmeans.predict([reduced_data[index]])
                d[lab[0]] += 1
        correct += max(d.values())
    
    print "Accuracy:", float(correct) / 20000

print "RP - EM"
for n in range (2,17,2):
    transformer = GaussianRandomProjection(n_components=n)
    reduced_data = transformer.fit_transform(df_x)

    print "N:", n

    kmeans = GaussianMixture(n_components=26)
    kmeans.fit(reduced_data)

    correct = 0
    for i in range(26):
        d = defaultdict(int)
        for index, row in df.iterrows():
            if row[label] == i:
                lab = kmeans.predict([reduced_data[index]])
                d[lab[0]] += 1
        correct += max(d.values())
    
    print "Accuracy:", float(correct) / 20000

##########################################################################

df_x.dropna()
varss = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]

print "Feature Selection - kmeans"
for var in varss:
    sel = VarianceThreshold(threshold=var)
    reduced_data = sel.fit_transform(df_x)

    print "Var:", var

    kmeans = KMeans(init='k-means++', n_clusters=26, n_init=26)
    kmeans.fit(reduced_data)

    correct = 0
    for i in range(26):
        d = defaultdict(int)
        for index, row in df.iterrows():
            if row[label] == i:
                lab = kmeans.predict([reduced_data[index]])
                d[lab[0]] += 1
        correct += max(d.values())
    
    print "Accuracy:", float(correct) / 20000

print "Feature Selection - EM"
for var in varss:
    sel = VarianceThreshold(threshold=var)
    reduced_data = sel.fit_transform(df_x)

    print "Var:", var

    kmeans = GaussianMixture(n_components=26)
    kmeans.fit(reduced_data)

    correct = 0
    for i in range(26):
        d = defaultdict(int)
        for index, row in df.iterrows():
            if row[label] == i:
                lab = kmeans.predict([reduced_data[index]])
                d[lab[0]] += 1
        correct += max(d.values())
    
    print "Accuracy:", float(correct) / 20000

# ##########################################################################