import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Ellipse
import matplotlib.lines as lines
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
from math import pi
from scipy import linalg, ndimage
from scipy.stats import multivariate_normal

from sklearn.exceptions import NotFittedError
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

from rospy import logdebug, logerr, get_param
from costmap_learning.srv import GetCostmapResponse
from matrix import OutputMatrix

minimum_sample_size = int(get_param('minimum_sample_size'))

class CostmapRelation:
    """
    This class saves the relation between two costmap objects by saving next to the costmaps
    the components, which are in relation. Moreover, a relation_name needs to be given to identify
    the relation. Currently the name scheme is the following with separator being the string "<->".:
        relation_name = costmap.object_id + separator + other_costmap.object_id
    For more information see the function add_related_costmaps in item.py.
    """

    def __init__(self, relation_name, costmap, component, other_costmap, other_component):
        """This constuctor creates a CostmapRelation object given the parameters.

        :param relation_name: name of the relation
        :type relation_name: string
        :type costmap: Costmap
        :type component: int
        :type other_costmap: Costmap
        :type other_component: int"""
        self.relation_name = relation_name
        self.costmap = costmap
        self.component = component
        self.other_costmap = other_costmap
        self.other_component = other_component

class Costmap:
    """
    This class saves poses and locations of the given object object_id for the table table_id
    and the given context.
    """

    def __init__(self, object_id, table_id, context, data, x_name, y_name, orient_name,
                 random_state=42, optimal_n_components=None, clf=None):
        """
        This constructor creates a Costmap object for each object_id, table_id and context.
        The positions are represented by GaussianMixture objects with n components, whether
        the orientations are represented by GaussianMixture objects with 1 component.
        The positions are only using x and y coordinates and the orientation uses only the z rotation.

        :param object_id: object id
        :param table_id: table id
        :param context: context e.g. 'TABLE-SETTING'
        :param data: data for object_id, table_id and context saving pose information
        :param x_name: x coordinate feature name for positions
        :param y_name: y coordinate feature name for positions
        :param orient_name: z coordinate feature name for orientations
        :param random_state: seed for RNG
        :param optimal_n_components: number of components for the position GMM
        :param clf: position GMM
        """
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
        # save raw_data so costmaps can be updated or replaced
        self.raw_data = data
        if not optimal_n_components:
            optimal_n_components = self.get_component_amount(data, random_state=random_state)
        self.clf = clf.fit(data[[self.x_name, self.y_name]]) \
            if clf \
            else GaussianMixture(n_components=optimal_n_components, random_state=random_state,
                                 init_params="kmeans").fit(data[[self.x_name, self.y_name]])
        self.angles_clfs = []
        self.angles_clfs = self.init_angles_clfs(random_state=random_state)

    def init_angles_clfs(self, random_state=42):
        """
        This function initializes orientation GMM objects for each
        component in the position GMM.

        :param random_state: seed for RNG
        :return: orientation GMMs, which can be indexed by the positions components
        :rtype: list[GaussianMixture]
        """
        angles_by_components = self.sort_angles_to_components()
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
        """
        This function sorts the orientation samples to the components of the position GMM.

        :return: angles for each position GMM component
        :rtype: list[list[float]]
        """
        angles = self.raw_data[self.orient_name]
        coords = self.raw_data[[self.x_name, self.y_name]]
        component_labels = self.clf.predict(coords)
        ret = [[] for i in range(self.clf.n_components)]
        for angle, component_label in zip(angles, component_labels):
            ret[component_label].append(angle)
        return ret


    def get_boundary(self, component_i, n_samples=100, std=7.0):
        """ This function returns the boundary of one component from the position GMM.

        :param component_i: Return the boundary for this component of the position GMM.
        :param n_samples: Get n_samples many samples before calculating the boundary
        :param std: standard deviation
        :return the smallest x0 and y0 value in the GMM and the width and height of given component in GMM"""
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

    def get_component_amount(self, data, min_n_clusters=2, max_n_clusters=10,
                             visualize=False, random_state=42):
        """ This function returns the optimal amount of components to separate
        the poses with GMMs by using the silhouette score.

        :param data: pose data for object type, context and table_id
        :param min_n_clusters: minimum amount of clusters
        :param max_n_clusters: maximal amount of clusters
        :param visualize: bool to visualizes the different silhouette scores
        :param random_state: seed for RNG
        :returns: number of components
        :rtype: int
        """
        if max_n_clusters <= min_n_clusters:
            raise Exception("max_n_clusters has to be bigger than min_n_clusters")

        X = data[[self.x_name, self.y_name]]
        silhouette_avgs = []

        if visualize:
            clfs = []
        logdebug("Following scores for object-type %s on table %s:",
                  str(self.object_id), str(self.table_id))

        for n_clusters in range(min_n_clusters, max_n_clusters):
            clf = KMeans(n_clusters=n_clusters, random_state=random_state).fit(X)
            if visualize:
                clfs.append(clf)

            silhouette_avg = silhouette_score(X, clf.labels_)
            silhouette_avgs.append(silhouette_avg)
            logdebug("For %d clusters the average silhouette score is %d.",
                     n_clusters, silhouette_avg)
        if visualize:
            self.vis_clusters_with_silhouette(clfs, X, max_n_clusters, min_n_clusters)

        optimal_n_clusters = min_n_clusters + np.argmax(silhouette_avgs)
        logdebug("The optimal amount of clusters is %d.", optimal_n_clusters)
        return optimal_n_clusters

    def get_clf_params(self, components=None):
        """
        This function returns all parameters of the position and orientation GMM, if
        components None. Otherwise, only the parameters of the clusters given in
        components are returned.

        :param components: list of components ids
        :type components: list[int]
        :return: mean, covariance and weights of the position GMM and means, covariance of the orientation GMM
        """
        position_means = np.array([])
        position_covs = np.array([])
        weights = np.array([])
        angle_means = np.array([])
        angle_covs = np.array([])
        cs = components if components else range(0, self.clf.n_components)
        for component_i in cs:
            position_means = np.append(position_means, self.clf.means_[component_i].flatten())
            position_covs = np.append(position_covs, self.clf.covariances_[component_i].flatten())
            weights = np.append(weights, self.clf.weights_[component_i])
            angle_means = np.append(angle_means, self.angles_clfs[component_i].means_[0])
            angle_covs = np.append(angle_covs, self.angles_clfs[component_i].covariances_[0])
        return position_means, position_covs, weights, angle_means, angle_covs

    def costmap_to_ros_getcostmapresponse(self, relations=None, cs=None):
        """Returns the given costmaps in a ROS GetCostmapResponse message.

        If there are relations towards self, meaning relations is not empty, these will
        be evaluated by returning only the components of self.clf, which are given in
        relations. If relations is empty and no valid relation was found, the components
        of self.clf will be returned accordingly to the given component ids in cs.

        :param relations: contains costmaps which are in relation towards self.object_id
        :type relations: list[CostmapRelation]
        :key cs: Optional list of Integers containing indices of components of a GMM
        :type cs: list[int]
        """

        ros_costmap_response = GetCostmapResponse()

        # Get inidices of components which should be exported
        if relations:
            indices = list(map(lambda r: r.other_component, relations))
        else:
            indices = range(0, len(self.clf.weights_)) if not cs else cs

        # Get Params of Position and Angle GMM
        means, covs, weights, angle_means, angle_covs = self.get_clf_params(components=indices)
        angles = np.array(list(zip(angle_means, angle_covs))).flatten()

        # Get approximated bottom left point, width and height of each
        # position component from the position GMM
        for i in indices:
            output_matrix, _ = self.get_boundary(i)
            ros_costmap_response = output_matrix.set_ros_costmap_response(ros_costmap_response)

        logdebug("(costmap) Returning Means:")
        logdebug(means)
        logdebug("(costmap) Returning Covariance:")
        logdebug(covs)
        logdebug("(costmap) Returning the angles:")
        logdebug(angles)

        ros_costmap_response.angles = angles
        ros_costmap_response.means = means
        ros_costmap_response.covs = covs
        ros_costmap_response.weights = weights
        return ros_costmap_response

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

    #deprecated
    def get_value(self, x, y, component_i=0):
        means = self.clf.means_[component_i]
        covs = self.clf.covariances_[component_i]
        return multivariate_normal.pdf([[x, y]], mean=means, cov=covs)

    #deprecated
    def costmap_to_output_matrices(self, max_width_or_height=10000):
        output_matrices = []

        for i in range(0, self.clf.n_components):
            empty_output_matrix, angle = self.get_boundary(i)
            width = empty_output_matrix.width
            height = empty_output_matrix.height
            x_0 = empty_output_matrix.x
            y_0 = empty_output_matrix.y
            columns = abs(int(width / self.resolution))
            rows = abs(int(height / self.resolution))

            if rows > max_width_or_height or columns > max_width_or_height:
                logerr("Failed to created output_matrix for %s, since output matrix with width %d and height %d is to "
                       "big. Please check the coordinates of %s or lower the resolution.", self.object_id, columns,
                       rows, self.object_id)
                exit()
            try:
                res = np.zeros((rows, columns))
            except MemoryError:
                logerr("Failed to created output_matrix for %s. Please check the coordinates of %s.", self.object_id,
                       self.object_id)
                exit()
            for r in range(0, rows):
                for c in range(0, columns):
                    res[r][c] = self.get_value(empty_output_matrix.x + r * self.resolution,
                                               empty_output_matrix.y + c * self.resolution,
                                               i)
            if angle < 0:
                angle += 180
            else:
                angle = 180 - angle
            res = np.copy(ndimage.rotate(res, angle, reshape=False))
            res /= res.max()
            output_matrix = empty_output_matrix.copy()
            output_matrix.insert(res)
            output_matrices.append(output_matrix)
        return output_matrices

    #deprecated
    def costmap_to_output_matrix(self):
        ms = self.costmap_to_output_matrices()
        m = OutputMatrix.summarize(ms)
        return m