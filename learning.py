
# coding: utf-8

# In[53]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl


# In[54]:


vr_data = pd.read_csv("/home/thomas/nameisthiscsvname.csv")


# In[55]:


vr_data.describe()


# In[56]:


vr_data[40:50]


# In[57]:


vr_data.hist(bins = 15, figsize = (8,5))


# In[58]:


vr_data["object-type"].value_counts(normalize = True)


# In[59]:


from pandas.plotting import scatter_matrix
scatter_matrix(vr_data, figsize=(8, 8), c="#f1b7b0", hist_kwds={'color':['#f1b7b0']});


# In[60]:


vr_data[vr_data["object-type"] == "SpoonSoup"]["x"]


# In[61]:


fig, ax = plt.subplots(figsize=(18, 10))
for i in range(0,len(vr_data["object-type"].value_counts())):
    color = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red'][i]
    index_name = vr_data["object-type"].value_counts().index[i]
    data_for_specific_data_type = vr_data[vr_data["object-type"] == index_name]
    x = data_for_specific_data_type["x"]
    y = data_for_specific_data_type["y"]
    scale = 200.0
    ax.scatter(x, y, c=color, s=scale, label=index_name, alpha=0.5, edgecolors='none')

ax.legend()
ax.set_title("Coordinate points from objects used for breakfast")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.grid(True)

plt.show()


# In[62]:


from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle


# In[63]:


numberOfRows = 2000
offset = pd.DataFrame(index=np.arange(0, numberOfRows), columns=("kitchen_name", "human_name", "context", 
                                                                 "object-type", "from-location","to-location",
                                                                 "x", "y", "arm","object-orient-from-cam"))
#i = 0
#for x in np.arange(-1.5, -1, 0.5): # -0.5, 0.65, 0.005
#    for y in np.arange(0, 2, 0.5): # 0.85, 1.25, 0.005
#        offset.loc[i] = ["kitchen_description", "Thomas", "TABLE-SETTING",
#                         "offset", "offset", "offset", x, y, "LEFT", "vertical"]
#        i += 1
#for x in np.arange(-0.6, 0, 0.5): # -0.5, 0.65, 0.005
#    for y in np.arange(0, 2, 0.5): # 0.85, 1.25, 0.005
#        offset.loc[i] = ["kitchen_description", "Thomas", "TABLE-SETTING",
#                         "offset", "offset", "offset", x, y, "LEFT", "vertical"]
#        i += 1
#for x in np.arange(-1, -0.6, 0.5): # -0.5, 0.65, 0.005
#    for y in np.arange(1.5, 2, 0.5): # 0.85, 1.25, 0.005
#        offset.loc[i] = ["kitchen_description", "Thomas", "TABLE-SETTING",
#                         "offset", "offset", "offset", x, y, "LEFT", "vertical"]
#        i += 1
#for x in np.arange(-1, -0.6, 0.5): # -0.5, 0.65, 0.005
#    for y in np.arange(0, 0.5, 0.5): # 0.85, 1.25, 0.005
#        offset.loc[i] = ["kitchen_description", "Thomas", "TABLE-SETTING",
#                         "offset", "offset", "offset", x, y, "LEFT", "vertical"]
#        i += 1
#offset = offset.dropna()
#for j in range(0, len(offset)):
#    elem = offset.loc[j]
#    vr_data.loc[len(vr_data) + j] = [elem["kitchen_name"], elem["human_name"],
#                                     elem["context"], elem["object-type"],
#                                     elem["from-location"], elem["to-location"],
#                                     elem["x"], elem["y"], elem["arm"],elem["object-orient-from-cam"]]
vr_data = shuffle(vr_data)
X = vr_data[["x", "y"]]
Y = vr_data["object-type"]
clf = GaussianNB()

X = np.append(X, [[-0.7, 1.15]], axis=0)
Y = np.append(Y, ["BowlLarge"], axis=0)
X = np.append(X, [[-0.72, 1.16]], axis=0)
Y = np.append(Y, ["BowlLarge"], axis=0)
X = np.append(X, [[-0.73, 1.17]], axis=0)
Y = np.append(Y, ["BowlLarge"], axis=0)
X = np.append(X, [[-0.75, 1.15]], axis=0)
Y = np.append(Y, ["BowlLarge"], axis=0)
X = np.append(X, [[-0.73, 1.2]], axis=0)
Y = np.append(Y, ["BowlLarge"], axis=0)
X = np.append(X, [[-0.75, 1.2]], axis=0)
Y = np.append(Y, ["BowlLarge"], axis=0)
clf.fit(X, Y)


# In[64]:

from sklearn import mixture
import itertools
from scipy import linalg
data = {"BowlLarge" : [],
        "JaNougatBits" : [],
        "SpoonSoup" : [],
        "BaerenMarkeFrischeAlpenmilch38" : [],
        "offset" : []}
#for x in np.arange(-1, -0.4, 0.009): # -0.5, 0.65, 0.005
#    for y in np.arange(0.5, 1.5, 0.009): # 0.85, 1.25, 0.005
#        colors = {"BowlLarge" : 'tab:blue',
#                  "JaNougatBits" : "tab:orange",
#                  "SpoonSoup" : 'tab:green',
#                  "BaerenMarkeFrischeAlpenmilch38" : 'tab:red',
#                  "offset" : 'tab:yellow'}
#        predicted = clf.predict([[x, y]])[0]
#        probs = clf.predict_proba([[x, y]])
#        max_arg = probs.argmax()
#        prob = probs[0][max_arg]
#        color = colors[predicted]
#        data[predicted].append([x,y,prob])


# In[65]:


#i = 0
#bigger_than_prob = 0.01
#fig, ax = plt.subplots(figsize=(18, 10))
#for key, value in data.items():
#    color = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:pink'][i]
#    index_name = key
#    x = []
#    y = []
#    for v in value:
#        if v[2] > bigger_than_prob:
#           x.append(v[0])
#           y.append(v[1])
#    scale = 600.0
#    ax.scatter(x, y, c=color, s=scale, label=index_name, alpha=0.1, edgecolors='none')
#    vr_data_per_type = vr_data[vr_data['object-type'] == index_name]
#    ax.scatter(vr_data_per_type["x"],
#               vr_data_per_type["y"],
#               c=color, s=100, label="real " + index_name, alpha=0.8, edgecolors='tab:grey')
#    i += 1

#ax.legend()
#ax.set_title("Naive Bayes Distribution with probability of " + str(bigger_than_prob) + " from objects used for breakfast")
#ax.set_xlabel("x")
#ax.set_ylabel("y")
#ax.grid(True)

#plt.show()
color_iter = itertools.cycle(['navy', 'c', 'cornflowerblue', 'gold',
                              'darkorange'])
def plot_results(X, Y_, means, covariances, index, title):
    splot = plt.subplot(2, 1, 1 + index)
    for i, (mean, covar, color) in enumerate(zip(
            means, covariances, color_iter)):
        v, w = linalg.eigh(covar)
        v = 2. * np.sqrt(2.) * np.sqrt(v) # 2. * np.sqrt(2.) == sqrt(-2log(0.0001)), so 99,99% coverage
        # see too:
        # https://mars.wiwi.hu-berlin.de/mediawiki/mmstat3/index.php/Maximum-Likelihood-Methode
        # https://www.xarg.org/2018/04/how-to-plot-a-covariance-error-ellipse/
        u = w[0] / linalg.norm(w[0])
        # as he DP will not use every component it has access to
        # unless it needs it, we shouldn't plot the redundant
        # components.
        if not np.any(Y_ == i):
            continue
        plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], .8, color=color)

        # Plot an ellipse to show the Gaussian component
        angle = np.arctan(u[1] / u[0])
        angle = 180. * angle / np.pi  # convert to degrees
        ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=color)
        ell.set_clip_box(splot.bbox)
        ell.set_alpha(0.5)
        splot.add_artist(ell)
    splot.legend()
    splot.set_xlabel("x")
    splot.set_ylabel("y")
    plt.xlim(-1.25, -0.6)
    plt.ylim(0.8, 1.3)
    splot.grid(True)
    plt.xticks(())
    plt.yticks(())
    plt.title(title)


train_X = X[:]
train_Y = Y[:]
gmm = mixture.GaussianMixture(n_components=4).fit(train_X, train_Y)
plot_results(X, gmm.predict(X[:]), gmm.means_, gmm.covariances_, 0, 'Gaussian Mixture')

print("done1")
dpgmm = mixture.BayesianGaussianMixture(n_components=5,
                                        covariance_type='full').fit(train_X, train_Y)
plot_results(X, dpgmm.predict(X), dpgmm.means_, dpgmm.covariances_, 1,
             'Bayesian Gaussian Mixture with a Dirichlet process prior')
print("done2")
plt.show()
print("done3")

# In[66]:


from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics.pairwise import pairwise_distances_argmin
from sklearn.preprocessing import StandardScaler
vr_data = pd.read_csv("/home/thomas/nameisthiscsvname.csv")
X = vr_data[["x", "y"]]
X = StandardScaler().fit_transform(X)
#Y = vr_data["object-type"].values
#k_means = KMeans(n_clusters=4, n_init=10, max_iter=100)
#k_means.fit_transform(X)
#y_pred = k_means.predict(X)
#fig, ax = plt.subplots(figsize=(18, 10))
#plt.scatter(X[:,0], X[:,1], c=y_pred)
#ax.set_title("KMeans with 4 clusters")
#ax.set_xlabel("x")
#ax.set_ylabel("y")

#def fun(x):
#    if x == 'SpoonSoup':
#        return 3
#    elif x == 'JaNougatBits':
#        return 1
#    elif x == 'BaerenMarkeFrischeAlpenmilch38':
#        return 2
#    else:
#        return 0

#result = list(map(fun, Y))
#print(result)
#print(accuracy_score(result, y_pred))
#k_means_cluster_centers = np.sort(k_means.cluster_centers_, axis=0)
#plt.scatter(list(map(lambda x: x[0], k_means_cluster_centers)),
#            list(map(lambda x: x[1], k_means_cluster_centers)), c='tab:grey', s=300)
#k_means_cluster_centers


# In[67]:


#from scipy.cluster.vq import vq, kmeans, kmeans2, whiten
#X = vr_data[["x", "y"]]
#X = np.append(X, [[-4.5, 1.3]], axis=0)
#X = np.append(X, [[-3.5, 1.2]], axis=0)

#whitened = whiten(X) # kann auch X sein
#codebook, distortion = kmeans(whitened, 4, iter=5)
#plt.scatter(whitened[:, 0], whitened[:, 1])
#plt.scatter(codebook[:, 0], codebook[:, 1], c='r')
#plt.show()
#codebook
#distortion


# In[68]:


#whitened = whiten(X) # kann auch nur X sein
#codebook, distortion = kmeans2(whitened, 4, iter=100) # <- sehr fehleranfällig :/
#plt.scatter(whitened[:, 0], whitened[:, 1])
#plt.scatter(codebook[:, 0], codebook[:, 1], c='r')
#plt.show()
#whitened


# In[69]:


from itertools import cycle, islice
from sklearn.neighbors import kneighbors_graph
knn_graph = kneighbors_graph(X, 2, include_self=False)
ward = AgglomerativeClustering( # average, complete with d_t = 0.1 and k_nn 2; single d_t = 0.055 and k_nn 4
        affinity="cityblock", n_clusters=None, linkage='complete', connectivity=knn_graph, distance_threshold=0.1, compute_full_tree=True)
ward = AgglomerativeClustering(
        n_clusters=4, linkage='average', connectivity=knn_graph)
average = AgglomerativeClustering(
        n_clusters=4, linkage='average', connectivity=knn_graph)
ward.fit(X)
ward.labels_.astype(np.int)

plt.scatter(X[:, 0], X[:, 1], s=100, c=ward.labels_)
plt.show()

print(ward.labels_)
subtrees = []
