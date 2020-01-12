import numpy as np

from settings import Settings


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
        table_name_and_setting = self.settings_by_table[table_name]
        if table_name_and_setting:
            context = table_name_and_setting.contexts[context_name]
            for costmap in context:
                if costmap.object_id == object_id:
                    return costmap

    def get_object_matrix(self, table_name, context_name, object_id,
                          x_base_object_position, y_base_object_position):
        tmp = self.get_object(table_name, context_name, object_id)
        if tmp:
            return tmp.get_output_matrix( x_base_object_position, y_base_object_position)

    #    def add_kitchen(self, kitchen):
    #        for table in kitchen.settings_by_table.values():
    #            self.add_table(table)

    def __eq__(self, other):
        return self.name == other.name
