# coding: utf-8
import numpy as np
import pandas as pd

from human import Human

# db:
humans = []
kitchen = {}
# input file
vr_data = pd.read_csv("/home/thomas/nameisthiscsvname_with_full_setup_2_short.csv", na_values="NIL").dropna()

for kitchen_name in np.unique(vr_data["kitchen_name"]):
    data_by_kitchen = vr_data.loc[vr_data["kitchen_name"] == str(kitchen_name)]
    for human_name in np.unique(data_by_kitchen["human_name"]):
        if not (human_name in map(lambda h: h.name, humans)):
            humans.append(Human(str(human_name), data_by_kitchen))

costmaps = humans[0].settings_by_table["rectangular_table"].contexts["BREAKFAST"]
i = 0
for costmap in costmaps:
    cpy = costmaps[:]
    del cpy[i]
    costmap.costmap_to_output_matrices(resolution=0.05)[0].plot(costmap.object_id + " without relation")
    costmap.add_related_costmaps(cpy)
    #for relation_name, relation in costmap.related_costmaps.items():
    #    relation.costmap_to_output_matrices(resolution=0.05)[0].plot(relation.object_id)
    costmap.merge_related_into_matrix(resolution=0.05).plot(costmap.object_id)
    #for k, cr in costmap.related_costmaps.items():
        #print(k)
        #print(cr)
        #cr.plot_gmm()
    i += 1