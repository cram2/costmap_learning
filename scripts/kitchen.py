import pandas as pd
import numpy as np

from human import Human

class Kitchen:

    def __init__(self, kitchen_id):
        self.kitchen_id = kitchen_id
        self.humans = []

    def fit_data(self, full_path="/home/thomas/nameisthiscsvname_with_full_setup_2_short.csv",
                 kitchen_feature="kitchen_name", human_feature="human_name"):
        vr_data = pd.read_csv(full_path, na_values="NIL").dropna()
        for kitchen_name in np.unique(vr_data[kitchen_feature]):
            data_by_kitchen = vr_data.loc[vr_data[kitchen_feature] == str(kitchen_name)]
            for human_name in np.unique(data_by_kitchen[human_feature]):
                if not (human_name in map(lambda h: h.name, self.humans)):
                    self.humans.append(Human(str(human_name), data_by_kitchen))

    def get_object_location(self, object_id):
        """Gets storage location of the given object for the whole kitchen"""
        locations = []
        for human in self.humans:
            locations.append(human.get_object_storage(str(object_id)))
        locations.sort(key=lambda t: t[1])
        print(locations[0][0])
        return locations[0][0]

    def get_object_destination(self, object_id, context, human_name, table_id):
        """Gets the destination location of given object specific to the context, human_name
        and table_id"""
        for human in self.humans:
            if human.name == human_name:
                print("IslandArea")
                return "IslandArea"
                #return human.get_object(table_id, context,
                #object_id).object_storage[0][0] <- bull, since only
                #destinations are saved

    def get_costmap(self, table_id, context_name, human_name, object_id,
                    x_object_positions, y_object_positions, placed_object_types):
        for human in self.humans:
            if human.name == human_name:
                costmap = human.get_costmap(table_id, context_name, object_id,
                                            x_object_positions, y_object_positions, placed_object_types)
                print("Returning costmaps")
                return costmap


