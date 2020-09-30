import numpy as np
from rospy import get_param

from item import VRItem


class Settings:

    def __init__(self, table_id, table_data):
        self.table_id = table_id  # str als id for tables
        self.contexts = {}  # list of objects for each dict elem
        context_feature = get_param('context_feature')
        for context in np.unique(table_data[context_feature]):
            context_data = table_data.loc[table_data[context_feature] == str(context)]
            self.add_context_per_object_type(str(context), context_data)

    def add_context_per_object_type(self, context_name, context_data):
        # Init list of objects for specific context in self.contexts
        if context_name not in self.contexts.keys():
            self.contexts[context_name] = []

        object_feature = get_param('object_feature')
        current_objects = self.contexts[context_name]
        new_object_names = np.unique(context_data[object_feature])
        for new_object_name in new_object_names:
            old_object_existed_p = False
            # If Object for object type already existed, add old data to new data set
            i = 0
            for j, obj in enumerate(current_objects):
                if str(new_object_name) == obj.object_id:
                    old_object_existed_p = True
                    i = j
                    context_data.append(current_objects[j].raw_data)
                    break
            # Create new Object for given object
            data_for_object = context_data.loc[context_data[object_feature] == str(new_object_name)]
            new_object = VRItem(str(new_object_name), self.table_id, context_name, data_for_object)
            if new_object:
                if old_object_existed_p:
                    current_objects[i] = new_object
                else:
                    self.contexts[context_name].append(new_object)

