# coding: utf-8
import numpy as np
import pandas as pd

from human import Human

# fancy db:
humans = []
kitchen = {}


def fit_data(full_path="/home/thomas/nameisthiscsvname_with_full_setup_2_short.csv",
             kitchen_feature="kitchen_name", human_feature="human_name"):
    vr_data = pd.read_csv(full_path, na_values="NIL").dropna()
    for kitchen_name in np.unique(vr_data[kitchen_feature]):
        data_by_kitchen = vr_data.loc[vr_data[kitchen_feature] == str(kitchen_name)]
        for human_name in np.unique(data_by_kitchen[human_feature]):
            if not (human_name in map(lambda h: h.name, humans)):
                humans.append(Human(str(human_name), data_by_kitchen))


def lookup(kitchen_name, table_id, human_name, context_name, object_id):
    kitchen = kitchen[kitchen_name]
    if kitchen and table_id in kitchen.table_ids:
        for human in humans:
            if human.name == human_name:
                human.get_object_matrix(table_id, context_name, object_id)


def vis_learned_data():
    costmaps = humans[0].settings_by_table["rectangular_table"].contexts["BREAKFAST"]
    i = 0
    resolution = 0.02
    for costmap in costmaps:
        cpy = costmaps[:]
        del cpy[i]
        costmap.plot_gmm()
        for i in range(0, costmap.clf.n_components):
            costmap.plot_gmm(plot_in_other=True)
            costmap.output_matrix.plot(costmap.object_id)
            # costmap.costmap_to_output_matrices(resolution=resolution)[i].plot(
            #    costmap.object_id + " component " + str(i))
        # costmap.add_related_costmaps(cpy)
        # for relation_name, relation in costmap.related_costmaps.items():
        #    relation.plot_gmm(Test=True)
        #    relation.costmap_to_output_matrices(resolution=resolution)[0].plot(relation.object_id)
        # costmap.merge_related_into_matrix(resolution=resolution).plot(costmap.object_id)
        # for k, cr in costmap.related_costmaps.items():
        # print(k)
        # print(cr)
        # cr.plot_gmm()
        i += 1


def init_dataset():
    fit_data()
    vis_learned_data()


init_dataset()