import numpy as np

from rospy import logdebug, loginfo, logwarn, logerr, get_param

from matrix import OutputMatrix
from costmap import Costmap, CostmapRelation

from_location_feature = get_param('from_location_feature')
to_location_feature = get_param('to_location_feature')
object_type_feature = get_param('object_feature')
from_x_name_feature = get_param('from_x_name_feature')
from_y_name_feature = get_param('from_y_name_feature')
from_orient_feature = get_param('from_orient_feature')
to_x_name_feature = get_param('to_x_name_feature')
to_y_name_feature = get_param('to_y_name_feature')
to_orient_feature = get_param('to_orient_feature')

class VRItem:
    """
    The VRItem class represent one object type in the given data set and saves its
    destination and storage poses. The poses are saved in Costmap object, whether the
    location is saved in a string object.
    Lastly, for each context and table a new VRItem object is created.
    """

    def __init__(self, object_id, table_id, context, data, random_state=42):
        """
        The constructor initializes the VRItem object by creating Costmap objects for the discrete
        storage and destination poses for the given object_id, table_id and context. Moreover,
        related_costmaps is not initialized, since it must be invoked from the outside (see
        add_related_costmaps).

        :param object_id: Object type
        :param table_id: Id of a table
        :param context: Context e.g. 'TABLE-SETTING'
        :param data: Values saving the locations of the object_id for the given table_id and context
        :param random_state: Seed value for RNG
        """
        self.storage_costmap = Costmap(object_id, table_id, context, data, from_x_name_feature,
                                       from_y_name_feature, from_orient_feature, random_state=random_state,
                                       optimal_n_components=1)
        self.dest_costmap = Costmap(object_id, table_id, context, data, to_x_name_feature,
                                    to_y_name_feature, to_orient_feature, random_state=random_state)
        if not self.storage_costmap or not self.dest_costmap:
            logerr("(item) Object with type %s could not be created.", object_id)
            return
        try:
            self.context = str(context)
            self.object_id = str(object_id)
            self.table_id = str(table_id)
        except ValueError:
            logerr("(item) object_id, table_id and context should be possible to be strings")
            return
        self.raw_data = data  # save raw_data so costmaps can be updated or replaced
        self.object_storage = []  # lists of tuples [('storage_name', number)]
        self.object_destination = []  # lists of tuples [('placed_name', number)]
        self.add_object_storage(data)  # saves storage information in self.object_storage
        self.add_object_destination(data)  # saves storage information in self.object_storage
        self.related_costmaps = {} # initialized by function add_related_costmaps
        logdebug("(item) Created Item %s.", self.object_id)

    def get_angles(self, x, y):
        """Returns the mean and covariance of the GMM distribution, which represents
        most likely the angle at the given coordinate.

        :param x: x coordinate
        :param y: y coordinate
        :returns: mean and covariance in a Tuple
        :rtype: tuple[Mean, Covariance]
        """
        component = self.dest_costmap.clf.predict([[x, y]])
        clf = self.dest_costmap.angles_clfs[component]
        if clf:
            return clf.means_[0], clf.covariances_[0]

    def get_relations_from_base_object(self, x_object_positions, y_object_positions,
                                       placed_object_types, costmaps_to_placed_object_types,
                                       object_id_item):
        """ Returns the relational costmaps for given parameters in a GetCostmapResponse object.

        First the relational costmaps in self.related_costmaps will be spotted which are on the given coordinates
        x_object_positions and y_object_positions. If the costmap is in relation with the given object_id_item,
        it will be saved temporarily. After checking all coordinates with the given type self.object_id,
        the other object types in placed_object_types and their coordinates in x_object_positions and y_object_positions
        are inspected. To prevent returning costmaps in which objects are already placed on, the function removes
        all relations between the already placed objects and self.
        If there are still costmaps left, these will be wrapped in a GetCostmapResponse ROS message.

        :param x_object_positions: contains double values of the x-coordinates from the objects in placed_object_types
        :type x_object_positions: list
        :param y_object_positions: contains double values of the y-coordinates from the objects in placed_object_types
        :type y_object_positions: list
        :param placed_object_types: represents the already placed objects with their object type which is encoded as str
        :type placed_object_types: list
        :param costmaps_to_placed_object_types: contains VRItem objects for corresponding types given in placed_object_types
        :type costmaps_to_placed_object_types: list
        :returns the relational costmaps wrapped in a GetCostmap-Response
        :rtype: GetCostmapResponse
        """

        base_object_name = self.object_id
        object_id_costmap = object_id_item.dest_costmap
        relation_object_name = object_id_costmap.object_id

        # If object type for wanted costmap is not itself a base object we pass it,
        # since reflexive relationals are not supported
        if not base_object_name == relation_object_name:
            relation_costmaps = []

            # x, y with the name of base_object_name of the VRItem object self
            x_base_object_positions = [x for type, x in zip(placed_object_types, x_object_positions) if
                                       type == base_object_name]
            y_base_object_positions = [y for type, y in zip(placed_object_types, y_object_positions) if
                                       type == base_object_name]

            # Calculate most probable relation costmaps for given base (!) object positions
            for x, y in zip(x_base_object_positions, y_base_object_positions):
                sample = [[x, y]]
                related_to_base_object = []
                component = self.dest_costmap.clf.predict(sample)[0]
                for related_costmap in self.related_costmaps.values():
                    if component == related_costmap.component and \
                            relation_object_name in related_costmap.other_costmap.object_id:
                        related_to_base_object.append(
                            [related_costmap, related_costmap.costmap.clf.predict_proba(sample)[0][0]])
                prob_relation_costmap = sorted(related_to_base_object, key=lambda c_and_p: c_and_p[1], reverse=True)[0][0]
                relation_costmaps.append(prob_relation_costmap)

            ret_costmaps = relation_costmaps[:]

            # Coordinates and types of objects which are not of the type base_object_name
            x_object_positions_and_type = [(x, type) for type, x in zip(placed_object_types, x_object_positions) if
                                           not type == base_object_name]
            y_object_positions_and_type = [(y, type) for type, y in zip(placed_object_types, y_object_positions) if
                                           not type == base_object_name]

            # Remove relation if object of type in placed_object_types on x,y is probably placed
            for x_and_type, y_and_type in zip(x_object_positions_and_type, y_object_positions_and_type):
                sample = [[x_and_type[0], y_and_type[0]]]
                type = x_and_type[1] if x_and_type[1] == y_and_type[1] else None
                if not type:
                    raise Exception("Types are not equal")
                for related_costmap in ret_costmaps:
                    if str(type) in related_costmap.other_costmap.object_id:
                        clf = next(i.dest_costmap.clf for i in costmaps_to_placed_object_types
                                   if i.object_id == relation_object_name)
                        relation_label = clf.predict(sample)[0]
                        if relation_label == related_costmap.other_component and \
                                related_costmap in ret_costmaps:
                            ret_costmaps.remove(related_costmap)
            if ret_costmaps:
                return object_id_costmap.costmap_to_ros_getcostmapresponse(relations=ret_costmaps)

    def get_costmap_for_object_type(self, x_object_positions, y_object_positions):
        """ Returns costmap dest_costmap, which is cut accordingly to the given parameters.

        This function simply returns the wrapped costmap by previously removing components in which objects
        with the coordinates from x_base_object_positions and y_base_object_positions probably are.

        :param x_object_positions: contains double values of the x-coordinates from the objects in placed_object_types
        :type x_object_positions: list
        :param y_object_positions: contains double values of the y-coordinates from the objects in placed_object_types
        :type y_object_positions: list
        :returns: cut Costmap wrapped in a GetCostmapResponse
        :rtype: GetCostmapResponse object
        """
        if not x_object_positions and not y_object_positions:
            return self.dest_costmap.costmap_to_ros_getcostmapresponse()
        else:
            samples = list(map(list, zip(x_object_positions, y_object_positions)))
            components = [i for i in range(0, self.dest_costmap.clf.n_components)]
            removed_labels = []  # if e.g. more objects of the same type are in one component
            for label in self.dest_costmap.clf.predict(samples):
                if label not in removed_labels:
                    components.remove(label)
                    removed_labels.append(label)

            if components:
                return self.dest_costmap.costmap_to_ros_getcostmapresponse(cs=components)
            else:
                raise Exception("No costmaps left.")

    def get_object_storage(self):
        """Returns the object storage by simply returning the storage location
        which was used the most for the object self.object_id.

        :returns: Most used storage location of the object self.object_id
        :rtype: string"""
        if self.object_storage:
            return self.object_storage[0][0]
        else:
            raise AttributeError("The object storage was empty or was not initialized.")

    def add_object_destination(self, data):
        """Saves where the object in data was placed in the kitchen.

        :param data: Saves the destination information of one object type
        :rtype: None"""
        self.add_object_location(data, to_location_feature, True)

    def add_object_storage(self, data):
        """Saves where the object in data was stored in the kitchen.

        :param data: Saves the storage information of one object type
        :rtype: None"""
        self.add_object_location(data, from_location_feature, False)

    def add_object_location(self, data, location_feature, destination_p):
        """Saves where the object in data was stored or placed in the kitchen.

        :param data: Saves the storage or destination information of one object type
        :param destination_p: Tells if the storage or destination location should be saved
        :type destination_p: bool
        :param location_feature: Feature name of the location
        :type location_feature: string
        :rtype: None"""
        object_types = np.unique(data[object_type_feature])
        if len(object_types) == 1 and object_types == self.object_id:
            # Get the current object locations
            if destination_p:
                object_location = self.object_destination
            else:
                object_location = self.object_storage
            for storage in np.unique(data[location_feature]):
                # Get how often self.object_id was taken from storage
                new_value = len(data[data[location_feature] == storage])
                # If locations were already added in object_location this will be updated here
                updated = False
                for i, n in enumerate(object_location):
                    if str(storage) == n[0]:
                        updated = True
                        old_value = object_location[i][1]
                        object_location[i] = (str(storage), old_value + new_value)
                        break
                # If locations were not updated in object_location it will be appended
                if not updated:
                    object_location.append((str(storage), new_value))
            # Sort for getting later the storages ordered
            object_location.sort(key=lambda t: t[1])
            if destination_p:
                self.object_destination = object_location
            else:
                self.object_storage = object_location
        else:
            logerr("(item) The given data contains more than one object type or wrong object type data set was given.")


    def add_related_costmaps(self, costmaps, relation_seperation="<->"):
        """ Calculates and saves a CostmapRelation object for each given costmap in costmaps

        This function takes each given costmap in costmaps and calculates the nearest components between
        the destination costmap of self and the given costmap. This information is then saved in a CostmapRelation
        object in self.related_costmaps.

        :param costmaps: contains double values of the x-coordinates from the objects in placed_object_types
        :type costmaps: list
        :param relation_seperation: contains double values of the y-coordinates from the objects in placed_object_types
        :type relation_seperation: string
        :rtype: None
        """
        for costmap in costmaps:
            if costmap.object_id != self.object_id:
                indices = self.nearest_components(costmap)
                for pair in indices:
                    i = pair[0]
                    j = pair[1]
                    if not (i is None or j is None):
                        relation_name = self.object_id + str(i) + relation_seperation + costmap.object_id + str(j)
                        self.related_costmaps[relation_name] = CostmapRelation(relation_name, self.dest_costmap, i, costmap, j)
                        logdebug("(item) Created relation " + relation_name + ".")

    def nearest_components(self, costmap):
        """ Calculates the nearest components between the destination costmap of self
         and the given costmap based on its means.

        :param costmap: the other costmap object
        :type costmap: Costmap
        :returns: components ids of the nearest means
        :rtype: list[list[int]]"""
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
        """Converts the CostmapRelation objects in self.related_costmaps in one OutputMatrix object

        :returns: OutputMatrix saving discrete values of the GMM distributions in self.related_costmaps
        :rtype: OutputMatrix object"""
        relations = self.related_costmaps
        res = []
        if relations:
            for relation_name, relation in relations.items():
                if relation.clf.n_components == 2:
                    output_matrix = relation.costmap_to_output_matrices()[0]
                    res.append(output_matrix)
                else:
                    raise ValueError("The relation ", relation.object_id, " is no CostmapRelation object.")
            return OutputMatrix.summarize(res)
