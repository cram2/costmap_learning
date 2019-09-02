import kitchen.py
import costmaps.py


class Human:

    def __init__(self, name, costmaps, table):
        self.name = name
        self.costmaps = {table: costmaps}

    def __init__(self, name, costmaps, kitchen):
        self.name = name
        self.costmaps = {}
        for table_i in range(0, len(kitchen)):
            self.costmaps[kitchen.table_ids[table_i]] = costmaps[table_i]
