import csv
import numpy as np
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt 
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from sklearn.feature_selection import VarianceThreshold
from sklearn.random_projection import GaussianRandomProjection
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import seaborn as sns

sns.set(color_codes=True)

df = pd.read_csv("letter.csv")

label = 'lettr'

df_y = df[label]
df_x = df[[x for x in df.columns if label not in x]]

test_accuracy = []

hiddens = tuple(0 * [16])

##########################################################################

# Split into train and test
train_x = df_x.loc[0:14999,:]
train_y = df_y.loc[0:14999]
test_x = df_x.loc[15000:,:]
test_y = df_y.loc[15000:]

clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=hiddens)
clf.fit(train_x, train_y)

print "Raw Score:", accuracy_score(test_y, clf.predict(test_x))

##########################################################################

for n in range(2,17,2):
    pca = PCA(n_components=n)
    pca.fit(df_x)

    hiddens = tuple(1 * [n])

    # print "Score:", pca.score(df)
    print "N:", n

    reduced_data = PCA(n_components=n).fit_transform(df_x)

    # Split into train and test
    train_x = reduced_data[0:15000,:]
    train_y = df_y.loc[0:14999]
    test_x = reduced_data[15000:,:]
    test_y = df_y.loc[15000:]

    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=hiddens)
    clf.fit(train_x, train_y)

    print "Score:", accuracy_score(test_y, clf.predict(test_x))
    
##########################################################################

print "ICA"
for n in range(2,17,2):

    # print "Score:", pca.score(df)
    print "N:", n

    hiddens = tuple(0 * [n])

    reduced_data = FastICA(n_components=n).fit_transform(df_x)

    # Split into train and test
    train_x = reduced_data[0:15000,:]
    train_y = df_y.loc[0:14999]
    test_x = reduced_data[15000:,:]
    test_y = df_y.loc[15000:]

    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=hiddens)
    clf.fit(train_x, train_y)

    print "Score:", accuracy_score(test_y, clf.predict(test_x))

##########################################################################

print "RPs"
for n in range(2,17,2):
    transformer = GaussianRandomProjection(n_components=n)
    reduced_data = transformer.fit_transform(df_x)

    hiddens = tuple(0 * [n])

    print "N:", n

    # Split into train and test
    train_x = reduced_data[0:15000,:]
    train_y = df_y.loc[0:14999]
    test_x = reduced_data[15000:,:]
    test_y = df_y.loc[15000:]

    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=hiddens)
    clf.fit(train_x, train_y)

    print "Score:", accuracy_score(test_y, clf.predict(test_x))

##########################################################################