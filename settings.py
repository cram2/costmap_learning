import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import cross_val_score
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from sklearn import metrics
from matplotlib.patches import Ellipse

from costmaps import Costmap


class Settings:

    def __init__(self, table_id, table_data):
        self.table_id = table_id
        self.contexts = {}
        for context in np.unique(table_data["context"]):
            self.add_context_per_object_type(str(context), table_data.loc[table_data["context"] == str(context)])
            # self.add_context(str(context), table_data.loc[table_data["context"] == str(context)])

    def add_context_per_object_type(self, context_name, context_data):
        for object in np.unique(context_data["object-type"]):
            self.contexts[context_name] = Costmap(str(object), self.table_id, context_name,
                                                  context_data.loc[context_data["object-type"] == str(object)])

    def add_context(self, context_name, context_data):
        n_components = self.get_opt_n_components_1(context_data)
        self.contexts[context_name] = GaussianMixture(n_components=9).fit(context_data[["x", "y"]])
        self.plot_gmm(self.contexts[context_name], context_data[["x", "y"]])
        self.vis_prob_map(context_name, context_data)

    def SelBest(self, arr: list, X: int) -> list:
        '''
        returns the set of X configurations with shorter distance
        '''
        dx = np.argsort(arr)[:X]
        return arr[dx]

    def draw_ellipse(self, position, covariance, ax=None, **kwargs):
        """Draw an ellipse with a given position and covariance"""
        ax = ax or plt.gca()
        # Convert covariance to principal axes
        if covariance.shape == (2, 2):
            U, s, Vt = np.linalg.svd(covariance)
            angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
            width, height = 2 * np.sqrt(s)
        else:
            angle = 0
            width, height = 2 * np.sqrt(covariance)

        # Draw the Ellipse
        for nsig in range(1, 4):
            ax.add_patch(Ellipse(position, nsig * width, nsig * height,
                                 angle, **kwargs))

    def plot_gmm(self, gmm, X, label=True, ax=None):
        ax = ax or plt.gca()
        labels = gmm.fit(X).predict(X)
        if label:
            ax.scatter(X["x"], X["y"], c=labels, s=40, cmap='viridis', zorder=2)
        else:
            ax.scatter(X["x"], X["y"], s=40, zorder=2)

        w_factor = 0.2 / gmm.weights_.max()
        for pos, covar, w in zip(gmm.means_, gmm.covariances_, gmm.weights_):
            self.draw_ellipse(pos, covar, alpha=w * w_factor)
        plt.title("GMM with %d components" % len(gmm.means_), fontsize=(20))
        plt.xlabel("U.A.")
        plt.ylabel("U.A.")
        plt.show()

    def get_opt_n_components_1(self, data):
        n_clusters = np.arange(2, 25)
        sils = []
        sils_err = []
        iterations = 20
        for n in n_clusters:
            tmp_sil = []
            for _ in range(iterations):
                gmm = GaussianMixture(n, n_init=2).fit(data[["x", "y"]])
                labels = gmm.predict(data[["x", "y"]])
                sil = metrics.silhouette_score(data[["x", "y"]], labels, metric='euclidean')
                tmp_sil.append(sil)
            val = np.mean(self.SelBest(np.array(tmp_sil), int(iterations / 5)))
            err = np.std(tmp_sil)
            sils.append(val)
            sils_err.append(err)
        plt.errorbar(n_clusters, sils, yerr=sils_err)
        plt.title("Silhouette Scores", fontsize=20)
        plt.xticks(n_clusters)
        plt.xlabel("N. of clusters")
        plt.ylabel("Score")
        plt.show()

    def get_opt_n_components(self, context_data, max_n_components=25, k_cross=3):
        threshold = 0.02
        last_best = 0.00
        last_best_i = 0
        num_after_best = 5
        i_after_best = num_after_best
        object_names = np.unique(context_data["object-type"])
        X = context_data[["x", "y"]]
        for i in range(0, len(object_names)):
            context_data.loc[context_data["object-type"] == object_names[i]] = i
        Y = context_data["object-type"]
        for i in range(len(object_names), max_n_components + 1):
            clf = GaussianMixture(n_components=i)
            score = cross_val_score(clf, X[:77], Y[:77], cv=k_cross, scoring='accuracy').mean()
            if score > last_best + threshold:
                last_best = score
                last_best_i = i
                i_after_best = 5
            if i_after_best == 0:
                return i
        return last_best_i

    def vis_prob_map(self, context_name, context_data):
        clf = self.contexts[context_name]
        cmaps = [
            'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
            'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
            'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        for i in range(0, clf.n_components):
            mu = self.contexts[context_name].means_[i]
            cov = self.contexts[context_name].covariances_[i]
            x, y = np.mgrid[0.8:1.2:500j, 0.7:1.5:500j]
            # x, y = np.mgrid[mu[0] - 120 * cov[0][0]:mu[0] + 120 * cov[0][0]:50j,
            #                mu[1] - 120 * cov[1][1]:mu[1] + 120 * cov[1][1]:50j]
            xy = np.column_stack([x.flat, y.flat])
            z = multivariate_normal.pdf(xy, mean=mu, cov=cov)
            z = z.reshape(x.shape)
            ax.plot_surface(x, y, z, cmap=cmaps[clf.predict([[mu[0], mu[1]]])[0]])
        plt.show()
