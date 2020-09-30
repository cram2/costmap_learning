import math

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Ellipse
import matplotlib.lines as lines
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
from math import pi
from random import randrange
from scipy import linalg, ndimage
from scipy.stats import multivariate_normal
import pandas as pd
from sklearn.exceptions import NotFittedError

from sklearn.mixture import GaussianMixture
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score, accuracy_score
from sklearn.model_selection import train_test_split

from rospy import logerr, get_param
from costmap_learning.srv import GetCostmapResponse
from matrix import OutputMatrix

from pathlib import Path
from diskcache import Cache


CACHE = Cache(str(Path.home()) + "/.cache/costmap_learning") # TODO: DOES NOT SUPPORT DIFFERENT HUMANS, CONTEXTS OR KITCHENS YET
# TODO: To fix above maybe add reference object to kitchen and add ids of above in one big id together with the current db-id

minimum_sample_size = int(get_param('minimum_sample_size'))

class Costmap:

    def __init__(self, object_id, table_id, context, data, x_name, y_name, orient_name,
                 random_state=42, optimal_n_components=None, clf=None):
        try:
            self.resolution = 0.01
            self.context = str(context)
            self.object_id = str(object_id)
            self.table_id = str(table_id)
            self.x_name = x_name
            self.y_name = y_name
            self.orient_name = orient_name
        except ValueError:
            logerr("(costmap) object_id, table_id and context should be possible to be strings")
            return
        if data.shape < (minimum_sample_size, 0) and not optimal_n_components:
            logerr("(costmap) sample size for object type %s is too small.", object_id)
            return
        self.raw_data = data  # save raw_data so costmaps can be updated or replaced
        if not optimal_n_components:
            optimal_n_components = self.get_component_amount(data, random_state=random_state)
        # X_train, X_test = train_test_split(data[[self.x_name, self.y_name]], test_size=.1)
        self.clf = clf.fit(data[[self.x_name, self.y_name]]) \
            if clf \
            else GaussianMixture(n_components=optimal_n_components, random_state=random_state,
                                 init_params="kmeans").fit(data[[self.x_name, self.y_name]])
        self.angles_clfs = []
        self.angles_clfs = self.init_angles_clfs(random_state=random_state)
        # kernel = 1.0 * RBF([1.0]) # check if RBF([1.0]) or RBF([1.0, 1.0]) with
        # https://scikit-learn.org/stable/auto_examples/gaussian_process/plot_gpc.html#sphx-glr-auto-examples-gaussian-process-plot-gpc-py
        # y = self.clf.predict(X_train) # TODO REPLACE y WITH REAL y!
        # y_pred = self.clf.predict(X_test)
        # self.output_clf = GaussianProcessClassifier(kernel=kernel).fit(X_train, y)
        self.cache = CACHE
        self.output_matrices = self.costmap_to_output_matrices()
        # self.plot_gmm(self.clf, data[[self.x_name, self.y_name]])
        # clf = BayesianGaussianMixture(n_components=5, random_state=random_state).fit(data[[self.x_name, self.y_name]])
        # self.plot_gmm(clf, data[[self.x_name, self.y_name]])

    def init_angles_clfs(self, random_state=42):
        angles_by_components = self.sort_angles_to_components()
        #print("angles by components of object" + self.object_id)
        #print(angles_by_components)
        ret = [None for i in range(self.clf.n_components)]
        for i in range(0, self.clf.n_components):
            # If the there is only one angle for a component of a GMM, ...
            if len(angles_by_components[i]) == 1:
                pseudo_data = [angles_by_components[i], angles_by_components[i]] # the point will be appended again, ...
                ret[i] = GaussianMixture(n_components=1, random_state=random_state, # so the GMM can be initialized.
                                         init_params="kmeans").fit(np.array(pseudo_data).reshape(-1, 1))
            else:
                ret[i] = GaussianMixture(n_components=1, random_state=random_state,
                                         init_params="kmeans").fit(np.array(angles_by_components[i]).reshape(-1, 1))
                var = ret[i].covariances_[0]
                # If the variance is bigger than pi we only take a look at the range(0, pi) or range(0, -pi)
                if var > pi:
                    mean  = ret[i].means_[0]
                    mean_smaller_zero = mean < 0
                    offset = -2 * pi if mean_smaller_zero else 2 * pi
                    arranged_angles = []
                    for angle in angles_by_components[i]:
                        if mean_smaller_zero and angle > 0 or not mean_smaller_zero and angle < 0:
                            arranged_angles.append(angle + offset)
                        else:
                            arranged_angles.append(angle)
                    ret[i] = GaussianMixture(n_components=1, random_state=random_state,
                                             init_params="kmeans").fit(np.array(np.array(arranged_angles).reshape(-1, 1)))
        return ret

    def sort_angles_to_components(self):
        angles = self.raw_data[self.orient_name]
        coords = self.raw_data[[self.x_name, self.y_name]]
        component_labels = self.clf.predict(coords)
        ret = [[] for i in range(self.clf.n_components)]
        for angle, component_label in zip(angles, component_labels):
            ret[component_label].append(angle)
        return ret


    def get_boundries(self, n_samples=100, component_i=0, std=7.0):
        """:return the smallest x0 and y0 value in the GMM and the width and height of given component in GMM"""
        gmm = self.clf
        X = []
        try:
            X, _ = gmm.sample(n_samples=n_samples)
        except NotFittedError:
            vr_data = self.raw_data[[self.x_name, self.y_name]].to_numpy()
            gmm = gmm.fit(vr_data)
            X, _ = gmm.sample(n_samples=n_samples)
        covars = gmm.covariances_
        covar = covars[component_i]
        w = np.sqrt(covar[0, 0]) * std
        h = np.sqrt(covar[1, 1]) * std
        U, s, Vt = np.linalg.svd(covar)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        w, h = (2 * np.sqrt(s)) * std
        w = max(w, h)
        h = max(w, h)
        means = gmm.means_
        mean = means[component_i]
        x0 = mean[0] - (w / 2)
        y0 = mean[1] - (h / 2)
        return OutputMatrix(x0, y0, w, h), angle

    def get_values(self, x, y):
        r = 0
        for i in range(0, len(self.clf.means_)):
            r += self.get_value(x, y, i)
        return r

    def get_cond_value(self, x, y):
        p_a = self.get_value(x, y, 0)
        p_b = self.get_value(x, y, 1)
        return (p_a * p_b) / p_b # = P(a|b) = P(a, b) / P(b)
    
    def get_value(self, x, y, component_i=0):
        # p_in_component = self.output_clf.predict_proba([[x, y]])[0][component_i]
        means = self.clf.means_[component_i]
        covs = self.clf.covariances_[component_i]
        return multivariate_normal.pdf([[x, y]], mean=means, cov=covs)
        # U, s, Vt = np.linalg.svd(self.clf.covariances_[component_i])
        # theta = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        # print(theta)
        # a = round((math.cos(theta)**2)/(2*s_x) +
        #           (math.sin(theta)**2)/(2*s_y**2), 10)
        # b = round((-1 * (math.sin(2*theta)/(4*s_x**2))) +
        #           (math.sin(2*theta)/(4*s_y**2)), 10)
        # c = round((math.sin(theta)**2/(2*s_x**2)) +
        #           (math.cos(theta)**2/(2*s_y**2)), 10)
        # print("x_0 %3.3f, y_0 %3.3f, s_x %3.3f, s_y %3.3d" % (x_mu,
        #                                                       y_mu,
        #                                                       s_x, s_y))
        # try: 
        #     return 1 * np.exp( -1 * (float(a*(x-x_mu)**2) + 
        #                              float(2*b*(x-x_mu)*(y-y_mu)) + 
        #                              float(c*round((round((y-y_mu),10)**2), 10))))
        # except OverflowError:
        #     return float('inf')
        # return 2 * math.exp(-1 * ((((x - x_0) ** 2) / ( 2 * s_x)) + (((y - y_0) ** 2) / ( 2 * s_y))))
        # return p_in_component * self.clf.predict_proba([[x, y]])[0][component_i]

    def costmap_to_ros_getcostmapresponse(self, relations=None, cs=None):
        """Returns the given costmaps in a ROS GetCostmapResponse message.

        If there are relations towards self, meaning relations is not empty, these will
        be evaluated by returning only the components of the Costmap self which are in
        relation with them. If relations is empty and no valid relation was found, the
        components of the Costmap self will be returned accordingly to the given components
        in cs.

        :param relations: contains costmaps which are in relation towards self.object_id
        :type relations: list
        :key cs: Optional list of Integers containing indices of components of a GMM
        :type cs: list
        """
        # Actually ros_getcostmap_response should contain more costmaps, but CRAM cannot handle it.
        # moreover, the srv GetCostmap should then have lists of width, height, res and Point instead.
        if relations:
            component_i = 1
            output_matrices = []
            angles = []
            for i in range(0, len(relations)):
                relation_name = relations[i].object_id
                object_id_label = int(relation_name[len(relation_name) - 1])
                output_matrices.append(self.output_matrices[object_id_label])
            for relation in relations:
                #tmp = relation.output_matrices[component_i] # <- the 2 gmm relation way
                #output_matrices.append(tmp) # <- the 2 gmm relation way and uncomment above for loop
                mean = relation.angles_clfs[component_i].means_[0]
                cov = relation.angles_clfs[component_i].covariances_[0]
                angles.extend([mean, cov])
            ros_costmap_response = OutputMatrix.get_ros_costmap_response(output_matrices)
            ros_costmap_response.angles = angles
            return ros_costmap_response
        else:
            filtered_output_matrices = []
            if cs:
                filtered_output_matrices = np.array(self.output_matrices)[cs]
            else:
                filtered_output_matrices = np.array(self.output_matrices)
            ros_costmap_response = OutputMatrix.get_ros_costmap_response(filtered_output_matrices)
            angles = []
            indices = range(0, len(self.output_matrices)) if not cs else cs
            for i in indices:
                mean = self.angles_clfs[i].means_[0]
                cov = self.angles_clfs[i].covariances_[0]
                angles.extend([mean, cov])
            ros_costmap_response.angles = angles
            return ros_costmap_response

    def merge(self, other, self_component=0, o_component=0):
        output_matrix = self.costmap_to_output_matrices()[self_component]
        other_output_matrix = other.costmap_to_output_matrices()[o_component]
        return OutputMatrix.merge_matrices([output_matrix, other_output_matrix])

    def get_component_amount(self, data, min_n_clusters=2, max_n_clusters=10,
                             verbose=False, visualize=False, random_state=42):
        """Returns the optimal amount of components to separate the data with GMMs"""
        if max_n_clusters <= min_n_clusters:
            raise Exception("max_n_clusters has to be bigger than min_n_clusters")

        X = data[[self.x_name, self.y_name]]
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
            ax2.scatter(X[self.x_name], X[self.y_name], marker='.', s=250, lw=0, alpha=0.7,
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

    def plot_gmm(self, label=True, ax=None, plot_in_other=False, edgecolor=(0, 0, 0)):
        ax = ax or plt.gca()
        gmm = self.clf
        X = self.raw_data[[self.x_name, self.y_name]]
        labels = gmm.fit(X).predict(X)
        if label:
            ax.scatter(X[self.x_name], X[self.y_name], c=labels, s=40, cmap='viridis', zorder=2)
        else:
            ax.scatter(X[self.x_name], X[self.y_name], s=40, zorder=2)

        w_factor = 0.8 / gmm.weights_.max()
        for pos, covar, w in zip(gmm.means_, gmm.covariances_, gmm.weights_):
            self.draw_ellipse(pos, covar, alpha=w * w_factor,
                              facecolor="none", edgecolor=edgecolor)
        plt.title("GMM of %s with %d components" % (self.object_id, len(gmm.means_)), fontsize=(20))
        plt.xlabel("X")
        plt.ylabel("Y")
        if not plot_in_other:
            plt.show()

    def draw_ellipse(self, position, covariance, ax=None, **kwargs):
        """Draw an ellipse with a given position and covariance"""
        ax = ax or plt.gca()
        # Convert covariance to principal axes
        if covariance.shape == (2, 2):
            U, s, Vt = np.linalg.svd(covariance)
            angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
            width, height = 2 * np.sqrt(s)
            v, w = linalg.eigh(covariance)
            v = 2. * np.sqrt(2.) * np.sqrt(v)
            u = w[0] / linalg.norm(w[0])
            angle = np.arctan(u[1] / u[0])
            angle = 180. * angle / np.pi  # convert to degrees
            # width = np.sqrt(covariance[1, 1])
            # height = np.sqrt(covariance[0, 0])
        else:
            angle = 0
            width, height = 2 * np.sqrt(covariance)

        # Draw the Ellipse
        for nsig in range(1, 4):
            ax.add_patch(Ellipse(position, nsig * v[0], nsig * v[1],
                                 180. + angle, **kwargs))

    def costmap_to_output_matrices(self, with_closest=True, max_width_or_height=10000):

        if "from" in self.x_name:
            storage_p = True
            storage_suffix = "_storage"
        elif "to" in self.x_name:
            storage_p = False
        else:
            raise Exception("Unknown costmap type, neither storage nor destination.")

        with Cache(self.cache.directory) as reference:
            cached_output_matrices = []
            if storage_p:
                cached_output_matrix = reference.get(self.object_id + storage_suffix)
                if cached_output_matrix:
                    print("Loaded storage cache for " + self.object_id)
                    return [cached_output_matrix]
            else:
                for i in range(0, self.clf.n_components):
                    cached_output_matrix = reference.get(self.object_id + str(i))
                    if cached_output_matrix is not None:
                        cached_output_matrices.append(cached_output_matrix)
                if cached_output_matrices and len(cached_output_matrices) == self.clf.n_components:
                    print("Loaded destination cache for " + self.object_id)
                    return cached_output_matrices

        placement=" storage" if storage_p else " destination"
        print("No cache found for output matrix of " + self.object_id +  placement + " placements. Creating one.")

        output_matrices = []
        for i in range(0, self.clf.n_components):
            empty_output_matrix, angle = self.get_boundries(component_i=i)
            width = empty_output_matrix.width
            height = empty_output_matrix.height
            x_0 = empty_output_matrix.x
            y_0 = empty_output_matrix.y
            columns = abs(int(width / self.resolution))
            rows = abs(int(height / self.resolution))
            if rows > max_width_or_height or columns > max_width_or_height:
                logerr("Failed to created output_matrix for %s, since output matrix with width %d and height %d is to "
                       "big. Please check the coordinates of %s or lower the resolution.",
                       self.object_id, columns, rows, self.object_id)
                exit()
            try:
                res = np.zeros((rows, columns))
            except MemoryError:
                logerr("Failed to created output_matrix for %s. Please check the coordinates of %s.",
                       self.object_id, self.object_id)
                exit()
            #arr_row = np.arange(x_0, x_0 + width, self.resolution).reshape(1, columns)
            #arr_column = np.flip(np.arange(y_0, y_0 + height, self.resolution)).reshape(rows, 1)
            #arr_rows = np.copy(res)
            #arr_cols = np.copy(res)
            #for r in rows:
            #    arr_rows[r] = np.copy(arr_row)
            #for c in columns:
            #    arr_cols[:,c] = np.copy(arr_column)
            #res = list(map(get_value, zip(arr_rows, arr_cols)))
            for r in range(0, rows):
               for c in range(0, columns):
                   res[r][c] = self.get_value(empty_output_matrix.x + r * self.resolution,
                                              empty_output_matrix.y + c * self.resolution,
                                              i)
            #print("---")
            #print(self.object_id)
            #print(angle)
            if angle < 0:
                angle += 180
            else:
                angle = 180 - angle
            #print(angle)
            res = np.copy(ndimage.rotate(res, angle, reshape=False))
            res /= res.max()
            output_matrix = empty_output_matrix.copy()
            output_matrix.insert(res)
            with Cache(self.cache.directory) as reference:
                if storage_p:
                    reference.set(self.object_id + storage_suffix, output_matrix)
                else:
                    reference.set(self.object_id + str(i), output_matrix)
            output_matrices.append(output_matrix)
        return output_matrices

    def costmap_to_output_matrix(self):
        ms = self.costmap_to_output_matrices()
        m = OutputMatrix.summarize(ms)
        return m
