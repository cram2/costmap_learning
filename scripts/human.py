import numpy as np
from random import randrange
from rospy import get_param

from settings import Settings

class Human:
    """
    This class represents a human by saving its name and Setting objects for every context.
    """

    def __init__(self, name, data_w_tables):
        """This constructor creates a Human object by saving its name and creating
        Setting objects which are saved for every table the human interacted with.

        :param name: name id
        :type name: string
        :param data_w_tables: data for the human
        :type data_w_tables: DataFrame"""
        self.name = name
        self.settings_by_table = {}
        table_feature = get_param('table_feature')
        for table_name in np.unique(data_w_tables[table_feature]):
            table_data = data_w_tables.loc[data_w_tables[table_feature] == str(table_name)]
            self.settings_by_table[str(table_name)] = Settings(str(table_name), table_data)

    def __eq__(self, other):
        return self.name == other.name

    def get_object_storage(self, object_type):
        tmp = []
        for name_and_setting in self.settings_by_table.items():
            for context_name_and_costmaps in name_and_setting[1].contexts.items():
                for costmap in context_name_and_costmaps[1]:
                    if costmap.object_id == str(object_type):
                        tmp.append(costmap.object_storage[0])
        tmp.sort(key=lambda t: t[1], reverse=True)
        return tmp[0]

    def get_all_objects(self, table_name, context_name):
        """Returns all VRItem objects saved."""
        setting = self.settings_by_table[table_name]
        if setting:
            vritems = setting.contexts[context_name]
            return vritems

    def get_object(self, table_name, context_name, object_id):
        """ Returns the VRItem object corresponding to the given parameters.

            :param table_name: encoded name of the table
            :type table_name: str
            :param context_name: encoded name of the context
            :type context_name: str
            :param object_id: encoded type of the object used as identifier for VRItem objects
            :type object_id: str
            :returns: VRItem object for parameters
            :rtype: VRItem
        """
        vritems = self.get_all_objects(table_name, context_name)
        for vritem in vritems:
            if vritem.object_id == object_id:
                return vritem

    def get_storage_costmap(self, context_name, object_id):
        if self.settings_by_table:
            table_name_and_setting = list(self.settings_by_table.values())[0]
            context = table_name_and_setting.contexts[context_name]
            for vritem in context:
                if vritem.object_id == object_id:
                    return vritem.storage_costmap.costmap_to_ros_getcostmapresponse()


    def get_destination_costmap(self, table_name, context_name, object_id, x_object_positions, y_object_positions, placed_object_types):
        """ Returns the Costmap for the given arguments wrapped in a GetCostmapResponse object.

        First the function checks, if the placed objects are all of the given type object_id. If this is true, a cut
        Costmap of the given object type object_id will be returned. But if it is false, we probably need to return
        a Costmap which has to consider the poses of different objects too.
        At the beginning we choose a base_object_type which is only declared because of the given implementation in item.py.
        In general any object can be a base_object_type. After choosing a base_object_type, we get the corresponding VRItem object.
        Moreover, we get the VRItem objects of the placed_object_types too. With that we can now try to get relational_costmaps.
        If we get no relational_costmaps, we try another base_object_type. If every base_object_type returns no relational
        costmaps, we return a cut costmap of the object type object_id.

        :param table_name: encoded name of the table
        :type table_name: str
        :param conext_name: encoded name of the context/setting
        :type context_name: str
        :param object_id: encoded type of the wanted costmap
        :type object_id: str
        :param x_object_positions: contains double values of the x-coordinates from the objects in placed_object_types
        :type x_object_positions: list
        :param y_object_positions: contains double values of the y-coordinates from the objects in placed_object_types
        :type y_object_positions: list
        :param placed_object_types: represents the already placed objects with their object type which is encoded as str
        :type placed_object_types: list
        :returns: the costmap for the given parameters
        :rtype: GetCostmapResponse
        """
        tmp = self.get_object(table_name, context_name, object_id)
        if tmp:
            # If all placed objects have the same type as the one to place (= object_id), just return its cut costmap
            # with information of already placed objects in x_object_positions and y_object_positions
            if all(obj_type == object_id for obj_type in placed_object_types):
                costmaps = tmp.get_costmap_for_object_type(x_object_positions, y_object_positions)
                return costmaps
            else:
                # Else there are different object types, which does not have to mean needing of a "relational costmap"

                # Choose base object for future comparisons between costmaps
                if "BowlLarge" in placed_object_types:
                    base_object_type = "BowlLarge"
                elif "SpoonSoup" in placed_object_types:
                    base_object_type = "SpoonSoup"
                else:
                    base_object_type = placed_object_types[randrange(0, len(placed_object_types))]

                # Get the VRItem of base_object_type
                base_object = self.get_object(table_name, context_name, base_object_type)

                # Get VRItem's of already placed objects
                costmaps_to_placed_object_types = []
                for object_type in placed_object_types:
                    object = self.get_object(table_name, context_name, object_type)
                    if object not in costmaps_to_placed_object_types:
                        costmaps_to_placed_object_types.append(object)

                # Get VRItems for object that should be placed
                object_id_item = self.get_object(table_name, context_name, object_id)

                # Get relational costmaps wrapped in the GetCostmapResponse from the base object w.r.t. the placed objects
                relation_costmaps = base_object.get_relations_from_base_object(x_object_positions, y_object_positions,
                                                                               placed_object_types,
                                                                               costmaps_to_placed_object_types,
                                                                               object_id_item)
                # If there are no free places for object_id_item the base_object will be changed
                if not relation_costmaps:
                    object_types = list(set(placed_object_types[:]))
                    object_types.remove(object_id) # reflexive relational costmaps do not exist
                    while object_types and not relation_costmaps:
                        object_type = object_types[randrange(0, len(object_types))]
                        object_types.remove(object_type)
                        new_base_object = self.get_object(table_name, context_name, object_type)
                        relation_costmaps = new_base_object.get_relations_from_base_object(x_object_positions,
                                                                                           y_object_positions,
                                                                                           placed_object_types,
                                                                                           costmaps_to_placed_object_types,
                                                                                           object_id_item)
                    # If Costmaps were found return them, else....
                    if relation_costmaps:
                        return relation_costmaps
                    # ... return the cut costmaps of object type object_id
                    else:
                        xs = [x for type, x in zip(placed_object_types, x_object_positions) if
                              type == object_id]
                        ys = [y for type, y in zip(placed_object_types, y_object_positions) if
                              type == object_id]
                        return object_id_item.get_costmap_for_object_type(xs, ys)
                else:
                    # Return relational costmaps if base_object from the beginning was right
                    return relation_costmaps