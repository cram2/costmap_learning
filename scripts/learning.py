# coding: utf-8
import numpy as np
import pandas as pd
from rospy import roswarn

# from human import Human
from kitchen import Kitchen

# fancy db:
kitchens = {}
cram_obj_t_to_vr_obj_t_dict = {
    "BOWL": ["BowlLarge"],
    "SPOON": ["SpoonSoup", "SpoonDessert"],
    "KNIFE": ["KnifeTable"],
    "PLATE": ["PlateClassic28"],
    "CUP": ["Cup", "GlassRound", "GlassTall"],
    "BREAKFAST-CEREAL": ["KoellnMuesliKnusperHonigNuss",
                         "JaNougatBits",
                         "KoellnMuesliCranberry"],
    "MILK": ["BaerenMarkeFrischeAlpenmilch38"],
    "BOTTLE": ["HohesCOrange"]
    } ## TODO: muss mit einer reasoning-fun in die reasoning.py


def fit_data(full_path="/home/thomas/nameisthiscsvname_with_full_setup_2_short.csv",
             kitchen_feature="kitchen_name", human_feature="human_name"):
    vr_data = pd.read_csv(full_path, na_values="NIL").dropna()
    for kitchen_name in np.unique(vr_data[kitchen_feature]):
        kitchens[kitchen_name] = Kitchen(kitchen_name)
        kitchens[kitchen_name].fit_data(full_path=full_path,
                                        kitchen_feature=kitchen_feature,
                                        human_feature=human_feature)

def get_symbolic_location(req):
    storage_p = str(req.storage)
    try:
        object_id = cram_obj_t_to_vr_obj_t_dict[req.object_type][0]
    except KeyError:
        roswarn("%s is no known object_type", req.object_type)
        return
    kitchen_name = str(req.kitchen)
    context = str(req.context)
    human_name = str(req.name)
    table_id = str(req.table_id)
    if storage_p:
        # If only the storage of the object is necessary
        return kitchens[kitchen_name].get_object_location(object_id)
    else:
        # Else: Get the location of the object for the context, human_name, kitchen
        return kitchens[kitchen_name].get_object_destination(object_id, context, human_name, table_id)


def get_costmap(req):
    print("Lookup for:")
    try:
        object_id = cram_obj_t_to_vr_obj_t_dict[req.object_type][0]
    except KeyError:
        roswarn("%s is no known object_type", req.object_type)
        return
    print(object_id)
    context_name = req.context
    print(context_name)
    human_name = req.name
    print(human_name)
    kitchen_name = req.kitchen
    print(kitchen_name)
    table_id = req.table_id
    print(table_id)
    if True:# k and table_id in k.table_ids:
        return kitchens[kitchen_name].get_costmap(table_id, context_name, human_name, object_id)


def vis_learned_data(with_relation=False):
    for kitchen in kitchens.values():
        costmaps = kitchen.humans[0].settings_by_table["rectangular_table"].contexts["BREAKFAST"]
        i = 0
        for costmap in costmaps:
            cpy = costmaps[:]
            del cpy[i]
            # costmap.plot_gmm()
            # for i in range(0, costmap.clf.n_components):
            # costmap.costmap_to_output_matrices()[i].plot(
            #    costmap.object_id + " component " + str(i))
            # for relation_name, relation in costmap.related_costmaps.items():
            #    relation.plot_gmm(Test=True)
            #    relation.costmap_to_output_matrices()[0].plot(relation.object_id)
            if with_relation:  # with_relation:
                costmap.add_related_costmaps(cpy)
                costmap.plot_gmm(plot_in_other=True)
                costmap.merge_related_into_matrix().plot(costmap.object_id + " with related")
            costmap.plot_gmm(plot_in_other=True)
            costmap.output_matrix.plot(costmap.object_id)
            # for k, cr in costmap.related_costmaps.items():
            # print(k)
            # print(cr)
            # cr.plot_gmm()
            i += 1

def init_dataset():
    fit_data()
    #vis_learned_data()
