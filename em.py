import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from scipy.spatial.distance import cdist, pdist
from sklearn.mixture import GaussianMixture
CHARACTER_DATA_FILE = '../data_sets/letter_recognition/letter_recognition.csv'
char_data = pd.read_csv(CHARACTER_DATA_FILE)
char_x = char_data[[x for x in char_data.columns if x != 'lettr']]
WINE_DATA_FILE = '../data_sets/wine/winequality.csv'
wine_data = pd.read_csv(WINE_DATA_FILE)
wine_x = wine_data[[x for x in wine_data.columns if x != '"quality"']]
K = range(1, 40)
GMM_c = [GaussianMixture(n_components=k).fit(char_x) for k in K]
GMM_w = [GaussianMixture(n_components=k).fit(wine_x) for k in K]
print("Trained EM models")
LL_c = [gmm.score(char_x) for gmm in GMM_c]
LL_w = [gmm.score(wine_x) for gmm in GMM_w]
print("Calculated the log likelihood for each k")
BIC_c = [gmm.bic(char_x) for gmm in GMM_c]
BIC_w = [gmm.bic(wine_x) for gmm in GMM_w]
print("Calculated the BICs for each K")
AIC_c = [gmm.aic(char_x) for gmm in GMM_c]
AIC_w = [gmm.aic(wine_x) for gmm in GMM_w]
print("Calculated the AICs for each K")
plt.style.use('ggplot')
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(K, BIC_c, '*-', label='BIC for Letter Recognition')
ax.plot(K, AIC_c, '*-', label='AIC for Letter Recognition')
ax.plot(K, BIC_w, '*-', label='BIC for Wine Quality')
ax.plot(K, AIC_w, '*-', label='AIC for Wine Quality')
plt.grid(True)
plt.xlabel('Number of clusters')
plt.ylabel('Inference scores')
plt.legend(loc='best')
plt.title('Bayesian and Akaike Information Criterion Curve')
fig.savefig('graphs/em/bic_aic.png')
plt.show()
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(K, LL_c, '*-', label='Letter Recognition')
ax.plot(K, LL_w, '*-', label='Wine Quality')
plt.grid(True)
plt.xlabel('Number of clusters')
plt.ylabel('Log Likelihood')
plt.title('Log Likelihood Curve')
plt.legend(loc='best')
fig.savefig('graphs/em/log_likelihood.png')
plt.show()