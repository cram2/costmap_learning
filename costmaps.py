import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Ellipse
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
from scipy import linalg
import pandas as pd
from sklearn.exceptions import NotFittedError

from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

from matrix import OutputMatrix


class Costmap:

    def __init__(self, object_id, table_id, context, data, random_state=42,
                 minimum_sample_size=10, gmm_clf=None):
        try:
            self.context = str(context)
            self.object_id = str(object_id)
            self.table_id = str(table_id)
        except ValueError:
            print("object_id, table_id and context should be possible to be strings")
            return
        if data.shape < (minimum_sample_size, 0) and not gmm_clf:
            print("Sample size for object type ", object_id, " is to small.")
            return
        self.raw_data = data  # save raw_data so costmaps can be updated or replaced
        self.object_storage = []  # lists of tuples [('storage_name', number)]
        # self.add_object_storage(data)  # saves storage information in self.object_storage
        if not gmm_clf:
            optimal_n_components = self.get_component_amount(data, random_state=random_state)
            self.clf = GaussianMixture(n_components=optimal_n_components, random_state=random_state,
                                       init_params="kmeans").fit(data[["x", "y"]])
        else:
            self.clf = gmm_clf
        self.related_costmaps = {}
        # self.plot_gmm(self.clf, data[["x", "y"]])
        # clf = BayesianGaussianMixture(n_components=5, random_state=random_state).fit(data[["x", "y"]])
        # self.plot_gmm(clf, data[["x", "y"]])

    def add_object_storage(self, data):
        """Saves where the object in data was storaged in the kitchen"""
        object_types = np.unique(data["object-type"])
        if len(object_types) == 1 and object_types == self.object_id:
            for storage in np.unique(data["from-location"]):
                # Get how often self.object_id was taken from storage
                new_value = len(data[data["from-location"] == storage])
                # If storage was already added in self.object_storage this will be updated here
                updated = False
                for i, n in enumerate(self.object_storage):
                    if str(storage) == n[0]:
                        updated = True
                        old_value = self.object_storage[i][1]
                        self.object_storage[i] = (str(storage), old_value + new_value)
                        break
                # If storage is not in self.object_storage it will be appended
                if not updated:
                    self.object_storage.append((str(storage), new_value))
            # Sort for getting later the storages ordered
            self.object_storage.sort(key=lambda t: t[1])
        else:
            print("The given data contains more than one object type or wrong object type data set was given.")

    def add_related_costmaps(self, costmaps, random_state=42, relation_seperation="<->"):
        for costmap in costmaps:
            if costmap.object_id != self.object_id:
                i, j = self.colliding(costmap)
                if not (i is None or j is None):
                    # Copy x, y data from this and other costmap
                    raw_data_cpy = self.raw_data[["x", "y"]].copy()
                    #print(costmap.object_id)
                    other_raw_data_cpy = costmap.raw_data[["x", "y"]].copy()
                    # Use only x, y data which are in the component i in self and j in given costmap
                    raw_data_cpy = raw_data_cpy[self.clf.predict(raw_data_cpy.to_numpy()) == i]
                    other_raw_data_cpy = other_raw_data_cpy[costmap.clf.predict(other_raw_data_cpy.to_numpy()) == j]
                    # Merge the filtered x and y data
                    merged_data = pd.DataFrame().append(raw_data_cpy).append(other_raw_data_cpy)
                    # Copy the means and cov from the component i in self and j in given costmap
                    means_i, means_j = self.clf.means_[i], costmap.clf.means_[j]
                    cov_i, cov_j = self.clf.covariances_[i], costmap.clf.covariances_[j]
                    # create a new GMM representing the relation of self and given costmap
                    gmm = GaussianMixture(n_components=2, means_init=[means_i, means_j],
                                          precisions_init=[np.linalg.inv(cov_i), np.linalg.inv(cov_j)],
                                          random_state=random_state)
                    # and save it in self inside a costmap
                    self.related_costmaps[costmap.object_id] = Costmap(
                        self.object_id + relation_seperation + costmap.object_id,
                        self.table_id, self.context, merged_data,
                        gmm_clf=gmm)
                    print("created relation ", self.object_id + relation_seperation + costmap.object_id)

    def colliding(self, costmap):
        if costmap:
            for r in range(0, self.clf.n_components):
                for s in range(0, self.clf.n_components):
                    if True:  # todo do smart spooky math things
                        return r, s

    def get_boundries(self, n_samples=100, component_i=0):
        """:return the smallest x0 and y0 value in the GMM and the width and height of given component in GMM"""
        gmm = self.clf
        X = []
        try:
            X, _ = gmm.sample(n_samples=100)
        except NotFittedError:
            vr_data = self.raw_data[["x", "y"]].to_numpy()
            gmm = gmm.fit(vr_data)
            X, _ = gmm.sample(n_samples=100)
        covars = gmm.covariances_
        covar = covars[component_i]
        means = gmm.means_
        mean = means[component_i]
        v, _ = linalg.eigh(covar)
        v = 20. * np.sqrt(2.) * np.sqrt(v) # factor is normally 2 for 99%
        x0 = mean[0] - (v[0] / 2)
        y0 = mean[1] - (v[1] / 2)
        return OutputMatrix(x0, y0, v[0], v[1])

    def get_values(self, x, y):
        r = 0
        for i in range(0, len(self.clf.means_)):
            r += self.get_value(x, y, i)
        return r

    def get_value(self, x, y, component_i=0):
        return self.clf.predict_proba([[x, y]])[0][component_i]

    def merge_related_into_matrix(self, resolution=0.02):
        relations = self.related_costmaps
        output_matrix = None
        res = []
        if relations:
            for relation_name, relation in relations.items():
                if relation.clf.n_components == 2:
                    output_matrix = relation.costmap_to_output_matrices(resolution=resolution)[0]
                    res.append(output_matrix)
                else:
                    raise ValueError("oy, this ", relation.object_id, " aint no relation costmap mate.")
            #print("going to merge these relations")
            #print(res)
            return OutputMatrix.merge_matrices(res, resolution=resolution)

    def merge(self, other, self_component=0, o_component=0, resolution=0.02):
        output_matrix = self.costmap_to_output_matrices(resolution)[self_component]
        other_output_matrix = other.costmap_to_output_matrixs(resolution)[o_component]
        return OutputMatrix.merge_matrices([output_matrix, other_output_matrix], resolution=resolution)

    def get_component_amount(self, data, min_n_clusters=2, max_n_clusters=10,
                             verbose=False, visualize=False, random_state=42):
        """Returns the optimal amount of components to separate the data with GMMs"""
        if max_n_clusters <= min_n_clusters:
            raise Exception("max_n_clusters has to be bigger than min_n_clusters")

        X = data[["x", "y"]]
        silhouette_avgs = []

        if visualize:
            clfs = []
        if verbose:
            print("Following scores for object-type ",
                  str(self.object_type), " on table ", str(self.table_id), ":")

        for n_clusters in range(min_n_clusters, max_n_clusters):
            clf = KMeans(n_clusters=n_clusters, random_state=random_state).fit(X)
            if visualize:
                clfs.append(clf)

            silhouette_avg = silhouette_score(X, clf.labels_)
            silhouette_avgs.append(silhouette_avg)
            if verbose:
                print("For ", n_clusters,
                      " clusters the average silhouette score is ", silhouette_avg)
        if visualize:
            self.vis_clusters_with_silhouette(clfs, X, max_n_clusters, min_n_clusters)

        optimal_n_clusters = min_n_clusters + np.argmax(silhouette_avgs)
        if verbose:
            print("Optimal n_clusters amount is", optimal_n_clusters)
        return optimal_n_clusters

    def vis_clusters_with_silhouette(self, clfs, X, max_n_clusters, min_n_clusters):
        fig, axes = plt.subplots(max_n_clusters - min_n_clusters, 2)
        fig.set_size_inches(14, 28)
        for i in range(0, len(clfs)):
            ax1 = axes[i][0]
            ax2 = axes[i][1]
            # first subplot for n_clusters
            ax1.set_xlim([-0.1, 1])
            # second subplot for n_clusters with blank spaces
            ax1.set_ylim([0, len(X) + (i + min_n_clusters + 1) * 10])

            clf = clfs[i]
            silhouette_avg = silhouette_score(X, clf.labels_)
            sample_silhouette_values = silhouette_samples(X, clf.labels_)
            y_lower = 10
            for j in range(i + min_n_clusters):
                # Aggregate the silhouette scores for samples belonging to
                # cluster j, and sort them
                ith_cluster_silhouette_values = \
                    sample_silhouette_values[clf.labels_ == j]

                ith_cluster_silhouette_values.sort()

                size_cluster_i = ith_cluster_silhouette_values.shape[0]
                y_upper = y_lower + size_cluster_i

                color = cm.nipy_spectral(float(j) / (i + 1))
                ax1.fill_betweenx(np.arange(y_lower, y_upper),
                                  0, ith_cluster_silhouette_values,
                                  facecolor=color, edgecolor=color, alpha=0.7)

                # Label the silhouette plots with their cluster numbers at the middle
                ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(j))

                # Compute the new y_lower for next plot
                y_lower = y_upper + 10  # 10 for the 0 samples

            ax1.set_xlabel("Silhouette coefficient values")
            ax1.set_ylabel("Cluster labels")

            # The vertical line for average silhouette score of all the values
            ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

            ax1.set_yticks([])  # Clear the yaxis labels / ticks
            ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

            # 2nd Plot showing the actual clusters formed
            colors = cm.nipy_spectral(clf.labels_.astype(float) / (i + 1))
            ax2.scatter(X["x"], X["y"], marker='.', s=250, lw=0, alpha=0.7,
                        c=colors, edgecolor='k')

            # Labeling the clusters
            centers = clf.cluster_centers_
            # Draw white circles at cluster centers
            ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                        c="white", alpha=1, s=300, edgecolor='k')

            for i, c in enumerate(centers):
                ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                            s=50, edgecolor='k')

            ax2.set_xlabel("X")
            ax2.set_ylabel("Y")
        fig.suptitle("Clusters for object type " + str(self.object_id) +
                     "with their Silhouette Scores", fontsize=14)
        plt.show()

    def plot_gmm(self, label=True, ax=None, Test=False):
        ax = ax or plt.gca()
        gmm = self.clf
        X = self.raw_data[["x", "y"]]
        labels = gmm.fit(X).predict(X)
        if label:
            ax.scatter(X["x"], X["y"], c=labels, s=40, cmap='viridis', zorder=2)
        else:
            ax.scatter(X["x"], X["y"], s=40, zorder=2)

        w_factor = 0.2 / gmm.weights_.max()
        for pos, covar, w in zip(gmm.means_, gmm.covariances_, gmm.weights_):
            self.draw_ellipse(pos, covar, alpha=w * w_factor)
        plt.title("GMM with %d components" % len(gmm.means_), fontsize=(20))
        plt.xlabel("X")
        plt.ylabel("Y")
        if not Test:
            plt.show()

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

    def costmap_to_output_matrices(self, resolution=0.02, n_components=[]):
        output_matrices = []
        for i in range(0, self.clf.n_components):
            empty_output_matrix = self.get_boundries(i)
            x_steps = abs(int(empty_output_matrix.width / resolution))
            y_steps = abs(int(empty_output_matrix.height / resolution))
            res = np.zeros((x_steps, y_steps))
            for x in range(0, x_steps):
                for y in range(0, y_steps):
                    res[x][y] = self.get_value(empty_output_matrix.x + x * resolution,
                                               empty_output_matrix.y + y * resolution,
                                               i)
            output_matrix = empty_output_matrix.copy()
            output_matrix.insert(res)
            output_matrices.append(output_matrix)
        return output_matrices
