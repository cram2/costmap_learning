# coding: utf-8
import numpy as np
import pandas as pd
import matplotlib as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy.stats import norm

from rospy import logdebug, loginfo, logwarn, get_param
from rospkg import RosPack

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

def fit_data(data_path, na_values):
    kitchen_feature = get_param('kitchen_feature')
    human_feature = get_param('human_feature')
    vr_data = pd.read_csv(data_path, na_values=na_values).dropna()
    for kitchen_name in np.unique(vr_data[kitchen_feature]):
        kitchens[kitchen_name] = Kitchen(kitchen_name)
        kitchens[kitchen_name].fit_data(vr_data, kitchen_feature, human_feature)

def get_symbolic_location(req):

    storage_p = req.storage
    try:
        object_id = cram_obj_t_to_vr_obj_t_dict[req.object_type][0]
    except KeyError:
        logwarn("%s is no known object_type", req.object_type)
        return

    human_name = str(req.name)
    kitchen_name = str(req.kitchen)
    table_id = str(req.table_id)
    context = str(req.context)

    # Set kitchen_name and human_name automatically if only
    # one kitchen and one human is saved.
    if (len(list(kitchens.keys())) == 1):
        kitchen = list(kitchens.values())[0]
        kitchen_name = kitchen.kitchen_id
        if (len(kitchen.humans) == 1):
            human_name = kitchen.humans[0].name

    loginfo('(GetCostmapLocation) Symbolic location lookup for %s', object_id)
    if storage_p:
        # If only the storage of the object is necessary
        loginfo("(GetCostmapLocation) Storage location:")
        return kitchens[kitchen_name].get_object_location(object_id)  # TODO: Maybe loc depends from human_name or other params too?
    else:
        # Else: Get the location of the object for the context, human_name, kitchen
        loginfo("(GetCostmapLocation) Destination location:")
        return kitchens[kitchen_name].get_object_destination(object_id, context, human_name, table_id)


def get_costmap(req):
    try:
        object_id = cram_obj_t_to_vr_obj_t_dict[req.object_type][0]
    except KeyError:
        logwarn("(GetCostmap) %s is no known object_type", req.object_type)
        return
    x_object_positions = req.placed_x_object_positions
    y_object_positions = req.placed_y_object_positions
    placed_object_types = req.placed_object_types
    for i in range(0, len(placed_object_types)):
        placed_object_types[i] = cram_obj_t_to_vr_obj_t_dict[placed_object_types[i]][0]
    if (len(x_object_positions) != len(y_object_positions)):
        raise Exception("x_base_object_positions and y_base_object_positions does not have the same length.")
        return

    context_name = req.context
    human_name = req.name
    kitchen_name = req.kitchen
    table_id = req.table_id

    # Set kitchen_name and human_name automatically if only
    # one kitchen and one human is saved.
    if (len(list(kitchens.keys())) == 1):
        kitchen = list(kitchens.values())[0]
        kitchen_name = kitchen.kitchen_id
        if (len(kitchen.humans) == 1):
            human_name = kitchen.humans[0].name

    loginfo("(GetCostmap) Costmap lookup for %s", object_id)
    logdebug("(GetCostmap) with context_name %s", context_name)
    logdebug("(GetCostmap) with human_name %s", human_name)
    logdebug("(GetCostmap) with kitchen_name %s", kitchen_name)
    logdebug("(GetCostmap) with table_id %s", table_id)
    if False:  # k and table_id in k.table_ids:
        return kitchens[kitchen_name].get_storage_costmap(context_name, object_id)
    else:
        return kitchens[kitchen_name].get_destination_costmap(table_id, context_name, human_name, object_id,
                                                              x_object_positions, y_object_positions, placed_object_types)

def relation_acc(vritem, relation, relation_name, relation_n):
    if relation_n == 0:
        relation_name_label = int(relation_name.replace(vritem.object_id, "")[0])
    elif relation_n == 1:
        relation_name_label = int(relation_name[-1])
    samples, y = vritem.dest_costmap.clf.sample(n_samples=100)
    filtered_samples = []
    for j in range(0, len(samples)):
        if y[j] == relation_name_label:
            filtered_samples.append(samples[j])
    predicted_relation = relation.clf.predict(filtered_samples)
    j = 0
    z = 0
    for label in predicted_relation:
        if label == 0:
            j += 1
        else:
            z += 1
    acc_1 = j / len(predicted_relation)
    acc_2 = z / len(predicted_relation)
    return acc_1 if acc_1 > acc_2 else acc_2


def generate_relations_between_items(visualize_costmap=False, visualize_related_costmap=False,
                                     validate_related_costmap=False, visualize_orientations=False):
    for kitchen in kitchens.values():
        vritems = kitchen.humans[0].settings_by_table["rectangular_table"].contexts["TABLE-SETTING"]
        i = 0
        for vritem in vritems:
            # Calculating related costmaps
            dest_costmaps = list(map(lambda object: object.dest_costmap, vritems[:]))
            del dest_costmaps[i]
            vritem.add_related_costmaps(dest_costmaps)

            # costmap.dest_costmap.plot_gmm()
            # for i in range(0, costmap.dest_costmap.clf.n_components):
            #    costmap.costmap_to_output_matrices()[i].plot(costmap.dest_costmap.object_id + " component " + str(i))
            if visualize_costmap:
                vritem.dest_costmap.plot_gmm(plot_in_other=True)
                vritem.dest_costmap.costmap_to_output_matrix().plot("Destination of " + vritem.object_id)
                vritem.storage_costmap.plot_gmm(plot_in_other=True)
                vritem.storage_costmap.output_matrices[0].plot("Storage of " + vritem.object_id)
            if visualize_orientations:
                clfs = vritem.dest_costmap.angles_clfs
                if len(clfs) % 2 == 0:
                    columns = 2
                else:
                    columns = 1
                rows = int(len(clfs) / columns)
                fig, axes = plt.subplots(rows, columns, sharex=False, sharey=False, figsize=(6, 8))
                fig.suptitle("Orientation of " + vritem.dest_costmap.object_id)

                i = 0
                original_orients = vritem.dest_costmap.sort_angles_to_components()
                for r in range(rows):
                    for c in range(columns):
                        if i == len(clfs):
                            break
                        original_orient = np.array(original_orients[i])
                        samples, shape = clfs[i].sample(n_samples=100)
                        current_ax = axes.flatten()[r * columns + c]
                        current_ax.hist(original_orient.flatten(),
                                        bins=15, density=True, histtype='bar', color=["red"], alpha=1)
                        sns.distplot(samples.flatten(), bins=15,
                                     ax=axes.flatten()[r * columns + c],
                                     hist_kws = {"density": True, "align": "left"},
                                     norm_hist=False, hist=False,
                                     kde=False, fit=norm)
                        # sns.distplot(original_orient.flatten(),
                        #              ax=axes.flatten()[r * columns + c],
                        #              color="blue",
                        #              hist_kws = {"density": False, "align": "right"},
                        #              norm_hist=True, hist=True,
                        #              kde=False)
                        i += 1
                fig.add_subplot(111, frameon=False)
                plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
                plt.xlabel("degree [rad]")
                plt.ylabel("density")
                red_patch = mpatches.Patch(color='red', label='Original orientations')
                black_patch = mpatches.Patch(color='black', label='Learned distribution')
                plt.legend(handles=[red_patch, black_patch])
                plt.show()
            # Merge relation costmaps and the standard destination costmap
            # costmap.desgit_costmap.plot_gmm(plot_in_other=True)
            # costmap.merge_related_into_matrix().plot(costmap.object_id + " with related")
            # plot relation
            if visualize_related_costmap:
                for relation_name, relation in vritem.related_costmaps.items():
                    acc = 0
                    plot_text = relation.object_id
                    if validate_related_costmap:
                        acc = relation_acc(vritem, relation, relation_name, 0)
                        plot_text += "\n with acc. of " + str(acc) + " for " + vritem.object_id
                        other_relation_name = relation_name.replace(vritem.object_id, "")[4:-1]
                        other_vr_item = None
                        for vri in kitchen.humans[0].settings_by_table["rectangular_table"].contexts["TABLE-SETTING"]:
                            if vri.object_id == other_relation_name:
                                other_vr_item = vri
                        acc = relation_acc(other_vr_item, relation, relation_name, 1)
                        plot_text += "\n with acc. of " + str(acc) + " for " + other_relation_name
                    relation.plot_gmm(plot_in_other=True)
                    relation.costmap_to_output_matrix().plot(plot_text,name=relation_name)

            i += 1


def init_dataset():
    rospack = RosPack()
    ros_package_name = get_param('package_name')
    ros_package_path = rospack.get_path(ros_package_name)
    csv_file_name = get_param('data_csv_file_name')
    na_values = get_param('na_values')
    fit_data(ros_package_path + '/resource/' + csv_file_name, na_values)
    loginfo("(learning) Initialized data set.")
    generate_relations_between_items()
