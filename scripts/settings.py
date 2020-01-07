import numpy as np

from object import Object


class Settings:

    def __init__(self, table_id, table_data, context_feature="context"):
        self.table_id = table_id  # str als id for tables
        self.contexts = {}  # list of objects for each dict elem
        for context in np.unique(table_data[context_feature]):
            context_data = table_data.loc[table_data[context_feature] == str(context)]
            self.add_context_per_object_type(str(context), context_data)

    def add_context_per_object_type(self, context_name, context_data):
        # Init list of objects for specific context in self.contexts
        if context_name not in self.contexts:
            self.contexts[context_name] = []

        objects = self.contexts[context_name]
        new_objects = np.unique(context_data["object-type"])
        for object in new_objects:
            old_object_existed_p = False
            # If Object for object type already existed, add old data to new data set
            i = 0
            for j, n in enumerate(objects):
                if str(object) == n.object_id:
                    old_object_existed_p = True
                    i = j
                    context_data.append(objects[j].raw_data)
                    break
            # Create new Object for given object
            data_for_object = context_data.loc[context_data["object-type"] == str(object)]
            new_object = Object(str(object), self.table_id, context_name, data_for_object)
            if new_object:
                if old_object_existed_p:
                    objects[i] = new_object
                else:
                    self.contexts[context_name].append(new_object)
            else:
                print("Object with type ", str(object), " could not be created.")

