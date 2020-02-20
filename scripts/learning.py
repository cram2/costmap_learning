# coding: utf-8
import numpy as np
import pandas as pd
from rospy import logwarn

# from human import Human
from kitchen import Kitchen
from costmap import *
from item import *

# fancy db:
kitchens = {}
cram_obj_t_to_vr_obj_t_dict = {
    "BOWL": ["BowlLarge"],
    "SPOON": ["SpoonSoup", "SpoonDessert"],
    "KNIFE": ["KnifeTable"],
    "PLATE": ["PlateClassic28"],
    "CUP": ["Cup", "GlassRound", "GlassTall"],
    "BREAKFAST-CEREAL": ["JaNougatBits",
                         "KoellnMuesliKnusperHonigNuss",
                         "KoellnMuesliCranberry"],
    "MILK": ["BaerenMarkeFrischeAlpenmilch38"],
    "BOTTLE": ["HohesCOrange"]
}  ## TODO: muss mit einer reasoning-fun in die reasoning.py

def fit_data(full_path="/home/thomas/nameisthiscsvname_short.csv",
             kitchen_feature="kitchen_name", human_feature="human_name"):
    vr_data = pd.read_csv(full_path, na_values="NIL").dropna()
    for kitchen_name in np.unique(vr_data[kitchen_feature]):
        kitchens[kitchen_name] = Kitchen(kitchen_name)
        kitchens[kitchen_name].fit_data(full_path=full_path,
                                        kitchen_feature=kitchen_feature,
                                        human_feature=human_feature)


def get_symbolic_location(req):
    storage_p = req.storage
    try:
        object_id = cram_obj_t_to_vr_obj_t_dict[req.object_type][0]
    except KeyError:
        logwarn("%s is no known object_type", req.object_type)
        return
    kitchen_name = str(req.kitchen)
    context = str(req.context)
    human_name = str(req.name)
    table_id = str(req.table_id)
    if storage_p:
        # If only the storage of the object is necessary
        return kitchens[kitchen_name].get_object_location(
            object_id)  # TODO: Maybe loc depends from human_name or other params too?
    else:
        # Else: Get the location of the object for the context, human_name, kitchen
        return kitchens[kitchen_name].get_object_destination(object_id, context, human_name, table_id)


def get_costmap(req):
    print("Lookup for:")
    try:
        object_id = cram_obj_t_to_vr_obj_t_dict[req.object_type][0]
    except KeyError:
        logwarn("%s is no known object_type", req.object_type)
        return
    print(object_id)
    x_object_positions = req.placed_x_object_positions
    y_object_positions = req.placed_y_object_positions
    placed_object_types = req.placed_object_types
    for i in range(0, len(placed_object_types)):
        placed_object_types[i] = cram_obj_t_to_vr_obj_t_dict[placed_object_types[i]][0]
    if (len(x_object_positions) != len(y_object_positions)):
        raise Exception("x_base_object_positions and y_base_object_positions does not have the same length.")
        return
    context_name = req.context
    print(context_name)
    human_name = req.name
    print(human_name)
    kitchen_name = req.kitchen
    print(kitchen_name)
    table_id = req.table_id
    print(table_id)
    if True:  # k and table_id in k.table_ids:
        return kitchens[kitchen_name].get_costmap(table_id, context_name, human_name, object_id,
                                                  x_object_positions, y_object_positions, placed_object_types)


def generate_relations_between_items(visualize=False):
    for kitchen in kitchens.values():
        costmaps = kitchen.humans[0].settings_by_table["rectangular_table"].contexts["TABLE-SETTING"]
        i = 0
        for costmap in costmaps:
            cpy = list(map(lambda object: object.dest_costmap, costmaps[:]))
            del cpy[i]
            # costmap.dest_costmap.plot_gmm()
            # for i in range(0, costmap.dest_costmap.clf.n_components):
            #    costmap.costmap_to_output_matrices()[i].plot(costmap.dest_costmap.object_id + " component " + str(i))
            if visualize:
                costmap.dest_costmap.plot_gmm(plot_in_other=True)
                costmap.dest_costmap.costmap_to_output_matrix().plot("Destination of " + costmap.object_id)
                costmap.storage_costmap.plot_gmm(plot_in_other=True)
                costmap.storage_costmap.output_matrices[0].plot("Storage of " + costmap.object_id)
            costmap.add_related_costmaps(cpy)
            # Merge relation costmaps and the standard destination costmap
            # costmap.dest_costmap.plot_gmm(plot_in_other=True)
            # costmap.merge_related_into_matrix().plot(costmap.object_id + " with related")
            # plot relation
            if visualize:
                for relation_name, relation in costmap.related_costmaps.items():
                    relation.plot_gmm(plot_in_other=True)
                    relation.costmap_to_output_matrix().plot(relation.object_id)
            i += 1


def init_dataset():
    fit_data()
    generate_relations_between_items()
