import numpy as np
from random import randrange

from settings import Settings
from costmap import Costmap


class Human:

    def __init__(self, name, data_w_tables):
        self.name = name
        self.settings_by_table = {}
        for table_name in np.unique(data_w_tables["table_name"]):
            self.add_table(str(table_name), data_w_tables.loc[data_w_tables["table_name"] == str(table_name)])

    #    def __init__(self, name, kitchen):
    #        self.name = name
    #        self.settings_by_table = {}
    #        for table_i in range(0, len(kitchen)):
    #            self.settings_by_table[kitchen.table_ids[table_i]] = costmaps[table_i]

    def add_table(self, table_name, table_data):
        self.settings_by_table[table_name] = Settings(table_name, table_data)

    def get_object_storage(self, object_type):
        tmp = []
        for name_and_setting in self.settings_by_table.items():
            print(name_and_setting)
            for context_name_and_costmaps in name_and_setting[1].contexts.items():
                print(context_name_and_costmaps)
                for costmap in context_name_and_costmaps[1]:
                    print(object_type)
                    print(costmap.object_id)
                    if costmap.object_id == str(object_type):
                        tmp.append(costmap.object_storage[0])
        tmp.sort(key=lambda t: t[1], reverse=True)
        print(tmp)
        return tmp[0]

    def get_object(self, table_name, context_name, object_id):
        """:return VRItem object of given table_name, context_name, object_id
            :argument table_name name of the table
            :argument context_name name of the context
            :argument object_id the type of the object used as identifier for VRItem's
        """
        table_name_and_setting = self.settings_by_table[table_name]
        if table_name_and_setting:
            context = table_name_and_setting.contexts[context_name]
            for costmap in context:
                if costmap.object_id == object_id:
                    return costmap

    def get_costmap(self, table_name, context_name, object_id, x_object_positions, y_object_positions, placed_object_types):
        """:return GetCostmap-Response
        
        This function gets the table_name, context_name and object_id which are together a identifier for the desired 
        GetCostmap-Response. Object_id encodes the type of wanted costmap. With the information of already placed object
        positions encoded in x_object_positions and y_object_positions and the corresponding types in placed_object_types,
        the costmap will be calculated as following:
        First it will be checked if the placed object are all of the same type and if the object_id has the same type. 
        If this is true a cut Costmap of the object type will be returned. If this is false we probably need to return 
        a Costmap which has to consider the poses of different objects. At the beginning we choose a base_object_type
        which is only declared because of technical calculation reasons. Typically any object can be a base_object_type.
        After choosing a base_object_type we get the corresponding VRItem object. Moreover, we get the VRItem objects of 
        the placed_object_types too. With that we can now try to get relational_costmaps. If does not work for the first 
        base_object_type, we try another. If no base_object_type returns relational costmaps, we return a cut costmap 
        of the object type object_id.
        """
        tmp = self.get_object(table_name, context_name, object_id)
        if tmp:
            # If all placed objects have the same type as the one to place (= object_id), just return its cut costmap
            # with information of already placed objects in x_object_positions and y_object_positions
            if all(obj_type == object_id for obj_type in placed_object_types):
                costmaps = tmp.get_costmap_for_object_type(x_object_positions, y_object_positions)
                return costmaps
            else:
                # Else there are different object types, which does not mean needing of a "relational costmap"

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

                # Get relational costmaps wrapped in the GetCostmap-Response from the base object w.r.t. the placed objects
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
                    # If a Costmaps were found return them, else....
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


    #    def add_kitchen(self, kitchen):
    #        for table in kitchen.settings_by_table.values():
    #            self.add_table(table)

    def __eq__(self, other):
        return self.name == other.name
