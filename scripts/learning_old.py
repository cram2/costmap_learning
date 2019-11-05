
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[4]:


vr_data = pd.read_csv("/home/thomas/nameisthiscsvname.csv")


# In[5]:


vr_data.describe()


# In[6]:

vr_data[40:50]


# In[7]:


vr_data.hist(bins = 15, figsize = (8,5))


# In[8]:


vr_data["object-type"].value_counts(normalize = True)


# In[9]:


from pandas.plotting import scatter_matrix
scatter_matrix(vr_data, figsize=(8, 8), c="#f1b7b0", hist_kwds={'color':['#f1b7b0']});


# In[10]:


vr_data[vr_data["object-type"] == "SpoonSoup"]["x"]


# In[11]:


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


# In[86]:


from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score


# In[87]:


X = vr_data[["x", "y"]]
Y = vr_data["object-type"]
clf = GaussianNB()
clf.fit(X, Y)


# In[88]:


clf.predict([[-0.75, 0.95]])
clf.predict_proba([[-0.75, 1.25]])[0][3]


# In[89]:


a = [1,2]
c = [3,4]
for i, j in a, c:
    print(i) 
    print(j)


# In[109]:


data = {"BowlLarge" : [],
        "JaNougatBits" : [],
        "SpoonSoup" : [],
        "BaerenMarkeFrischeAlpenmilch38" : []}
for x in np.arange(-3, 3, 0.075): # -0.5, 0.65, 0.005
    for y in np.arange(0, 3, 0.075): # 0.85, 1.25, 0.005
        colors = {"BowlLarge" : 'tab:blue',
                  "JaNougatBits" : "tab:orange",
                  "SpoonSoup" : 'tab:green',
                  "BaerenMarkeFrischeAlpenmilch38" : 'tab:red'}
        predicted = clf.predict([[x, y]])[0]
        probs = clf.predict_proba([[x, y]])
        max_arg = probs.argmax()
        prob = probs[0][max_arg]
        color = colors[predicted]
        data[predicted].append([x,y,prob])


# In[125]:


i = 0
bigger_than_prob = 0.99
fig, ax = plt.subplots(figsize=(18, 10))
for key, value in data.items():
    color = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red'][i]
    index_name = key
    x = []
    y = []
    for v in value:
        if v[2] > bigger_than_prob:
            x.append(v[0])
            y.append(v[1])
    scale = 600.0
    ax.scatter(x, y, c=color, s=scale, label=index_name, alpha=0.8, edgecolors='none')
    i += 1

ax.legend()
ax.set_title("Naive Bayes Distribution with probability of " + str(bigger_than_prob) + " from objects used for breakfast")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.grid(True)

plt.show()
y_pred = GaussianNB().fit(X,Y).predict(X)
accuracy_score(Y.tolist(), y_pred)


# In[101]:


from sklearn.cluster import KMeans
y_pred = KMeans(n_clusters=4, random_state=234).fit_predict(X)
fig, ax = plt.subplots(figsize=(18, 10))
plt.scatter(X["x"], X["y"], c=y_pred)
ax.set_title("KMeans with 4 clusters")
ax.set_xlabel("x")
ax.set_ylabel("y")
plt.show()

def fun(x):
    if x == 'SpoonSoup': 
        return 2
    elif x == 'JaNougatBits':
        return 3
    elif x == 'BaerenMarkeFrischeAlpenmilch38':
        return 1
    else:
        return 0

accuracy_score(map(fun, Y), y_pred)


# In[64]:




