import numpy as np
from rospy import loginfo, get_param

from human import Human

kitchen_feature = get_param('kitchen_feature')
human_feature = get_param('human_feature')

class Kitchen:
    """
    This class represents a kitchen and saves Human objects.
    """

    def __init__(self, kitchen_id, data):
        """
        This constructor creates a kitchen object, which saves the kitchen id
        and the data used for this kitchen.

        :param kitchen_id: name of the kitchen
        :type kitchen_id: string
        :param data: data for this kitchen
        :type data: DataFrame
        """
        self.kitchen_id = kitchen_id
        self.raw_data = data
        self.humans = []
        self.fit_data(kitchen_feature, human_feature)

    def get_all_objects(self, table_name, context):
        """Returns all objects saved in this kitchen given the table
        and context for every human.

        :param table_name: name of the table
        :type table_name: string
        :param context: name of context
        :type context: string
        :returns: all objects for every human given the parameters
        :rtype: list[list[VRItems]]
        """
        return [human.get_all_objects(table_name, context)
                for human in self.humans]

    def fit_data(self, kitchen_feature, human_feature):
        """Initializes the human attribute using data saved in self.raw_data
        and the given features.

        :param kitchen_feature: name of the kitchen feature
        :type kitchen_feature: string
        :param human_feature: name of the human feature
        :type human_feature: string"""
        vr_data = self.raw_data
        for kitchen_name in np.unique(vr_data[kitchen_feature]):
            data_by_kitchen = vr_data.loc[vr_data[kitchen_feature] == str(kitchen_name)]
            for human_name in np.unique(data_by_kitchen[human_feature]):
                new_human = Human(str(human_name), data_by_kitchen)
                if not (new_human in self.humans):
                    self.humans.append(new_human)

    def get_object_location(self, object_id):
        """Gets storage location of the given object for the whole kitchen.

        :param object_id: id of the object
        :type object_id: string
        :rtype: string"""
        locations = []
        for human in self.humans:
            locations.append(human.get_object_storage(str(object_id)))
        locations.sort(key=lambda t: t[1])
        return locations[0][0]

    def get_object_destination(self, object_id, context, human_name, table_id):
        """Gets the destination location of given object specific to the context, human_name
        and table_id."""
        for human in self.humans:
            if human.name == human_name:
                return human.get_object(table_id, context, object_id).object_destination[0][0]

    def get_storage_costmap(self, context_name, object_id):
        """This function returns the storage costmap for the given object_id and context context_name.

        :rtype: GetCostmapResponse"""
        if self.humans:
            human = self.humans[0]
            costmap = human.get_storage_costmap(context_name, object_id)
            loginfo("(GetCostmap) Returning storage costmaps")
            return costmap

    def get_destination_costmap(self, table_id, context_name, human_name, object_id,
                                x_object_positions, y_object_positions, placed_object_types):
        """This function returns the destination costmap for the given parameters.

        :rtype: GetCostmapResponse"""
        for human in self.humans:
            if human.name == human_name:
                costmap = human.get_destination_costmap(table_id, context_name, object_id,
                                                        x_object_positions, y_object_positions, placed_object_types)
                loginfo("(GetCostmap) Returning destination costmaps")
                return costmap