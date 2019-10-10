class Kitchen:

    def __init__(self, kitchen_id, table_ids):
        self.kitchen_id = kitchen_id
        for table_i in range(0, len(table_ids)):
            self.settings_by_table[table_ids[table_i]] = []

    def calc_settings(self, table, humans, data):
        pass