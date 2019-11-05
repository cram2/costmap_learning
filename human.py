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

    def get_object_matrix(self, table_name, context_name, object_id):
        setting = self.settings_by_table[table_name]
        if setting:
            context = setting.contexts[context_name]
            for costmap in context:
                if costmap.object_id == object_id:
                    return costmap.output_matrix

    #    def add_kitchen(self, kitchen):
    #        for table in kitchen.settings_by_table.values():
    #            self.add_table(table)

    def __eq__(self, other):
        return self.name == other.name
