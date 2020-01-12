import math

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Ellipse
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
from scipy import linalg, ndimage
from scipy.stats import multivariate_normal
import pandas as pd
from rospy import loginfo
from sklearn.exceptions import NotFittedError

from sklearn.mixture import GaussianMixture
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score, accuracy_score
from sklearn.model_selection import train_test_split

from matrix import OutputMatrix
from costmap import Costmap
from random import randrange


class VRItem:

    def __init__(self, object_id, table_id, context, data, random_state=42,
                 minimum_sample_size=10):
        self.storage_costmap = Costmap(object_id, table_id, context, data, random_state=random_state,
                                    x_name="from-x", y_name="from-y", orient_name="from-orient",
                                    minimum_sample_size=minimum_sample_size,
                                    optimal_n_components=1)
        self.dest_costmap = Costmap(object_id, table_id, context, data, random_state=random_state,
                                    x_name="to-x", y_name="to-y", orient_name="to-orient",
                                    minimum_sample_size=minimum_sample_size)
        try:
            self.context = str(context)
            self.object_id = str(object_id)
            self.table_id = str(table_id)
        except ValueError:
            print("object_id, table_id and context should be possible to be strings")
            return
        if data.shape < (minimum_sample_size, 0):
            print("Sample size for object type ", object_id, " is too small. Ignored.")
            return
        self.raw_data = data  # save raw_data so costmaps can be updated or replaced
        self.object_storage = []  # lists of tuples [('storage_name', number)]
        self.add_object_storage(data)  # saves storage information in self.object_storage
        self.related_costmaps = {}
        print("Created Item " + self.object_id + ".")

    def get_output_matrix(self, x_base_object_position, y_base_object_position):
        # If one of the given float64 has its max values
        if x_base_object_position == np.finfo(np.float32).max or \
                y_base_object_position == np.finfo(np.float32).max:
            return self.dest_costmap.output_matrices[randrange(self.dest_costmap.clf.n_components)]
        else :
            base_object_name = "BowlLarge"
            related_to_base_object = []
            sample = [[x_base_object_position, y_base_object_position]]
            for related_costmap in self.related_costmaps.values():
                if "<->" + base_object_name in related_costmap.object_id:
                    related_to_base_object.append([related_costmap, related_costmap.clf.predict_proba(sample)[0][1]])
            prob_relation_costmap = sorted(related_to_base_object, key=lambda c_and_p: c_and_p[1], reverse=True)[0][0]
            return prob_relation_costmap.output_matrices[0]

    def get_object_storage(self):
        if self.object_storage:
            return self.object_storage[0][0]
        else:
            raise AttributeError("The object storage was empty or was not initialized.")

    def add_object_storage(self, data):
        """Saves where the object in data was storaged in the kitchen"""
        object_types = np.unique(data["object-type"])
        if "<->" in self.object_id or len(object_types) == 1 and object_types == self.object_id:
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
                if True:
                    indices = self.nearest_components(costmap)
                    for pair in indices:
                        i = pair[0]
                        j = pair[1]
                        if not (i is None or j is None):
                            # Copy x, y data from this and other costmap
                            raw_data_xy = self.raw_data[[self.dest_costmap.x_name, self.dest_costmap.y_name]].copy()
                            other_raw_data_xy = costmap.raw_data[[costmap.x_name, costmap.y_name]].copy()
                            # Use only x, y data which are in the component i in self and j in given costmap
                            raw_data_cpy = self.raw_data[self.dest_costmap.clf.predict(raw_data_xy.to_numpy()) == i].copy()
                            other_raw_data_cpy = costmap.raw_data[costmap.clf.predict(other_raw_data_xy.to_numpy()) == j].copy()
                            # Merge the filtered x and y data
                            merged_data = pd.DataFrame().append(raw_data_cpy).append(other_raw_data_cpy)
                            # Copy the means and cov from the component i in self and j in given costmap
                            means_i, means_j = self.dest_costmap.clf.means_[i], costmap.clf.means_[j]
                            cov_i, cov_j = self.dest_costmap.clf.covariances_[i], costmap.clf.covariances_[j]
                            # create a new GMM representing the relation of self and given costmap
                            gmm = GaussianMixture(n_components=2, means_init=[means_i, means_j],
                                                  precisions_init=[np.linalg.inv(cov_i), np.linalg.inv(cov_j)],
                                                  random_state=random_state)
                            relation_name = self.object_id + str(i) + relation_seperation + costmap.object_id + str(j)
                            # and save it in self inside a costmap
                            self.related_costmaps[relation_name] = Costmap(
                                self.object_id + str(i) + relation_seperation + costmap.object_id + str(j),
                                self.table_id, self.context, merged_data,
                                random_state=random_state, optimal_n_components=2,
                                x_name=self.dest_costmap.x_name, y_name=self.dest_costmap.y_name,
                                orient_name=self.dest_costmap.orient_name)
                            print("Created relation ", self.object_id + str(i) + relation_seperation
                                  + costmap.object_id + str(j) + ".")

    def nearest_components(self, costmap):
        res = []
        if costmap:
            s_means = self.dest_costmap.clf.means_
            o_means = costmap.clf.means_
            indices = []
            for s_i in range(0, len(s_means)):
                for o_i in range(0, len(o_means)):
                    distance = np.linalg.norm(s_means[s_i] - o_means[o_i])
                    new = np.array([s_i, o_i, distance])
                    indices.append(new)
            indices = np.array(indices)
            for s_i in range(0, len(s_means)):
                tmp = indices[indices[:, 0] == s_i]
                tmp = sorted(tmp, key=lambda l: l[2])
                res.append([int(s_i), int(tmp[0][1])])
            return res

    def merge_related_into_matrix(self):
        relations = self.related_costmaps
        output_matrix = None
        res = []
        if relations:
            for relation_name, relation in relations.items():
                if relation.clf.n_components == 2:
                    #relation.costmap_to_output_matrices()[0].plot(relation_name)
                    output_matrix = relation.costmap_to_output_matrices()[0]
                    res.append(output_matrix)
                else:
                    raise ValueError("oy, this ", relation.object_id, " aint no relation costmap mate.")
            return OutputMatrix.summarize(res)
