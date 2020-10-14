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
    """This function creates for every kitchen mentioned in the CSV a Kitchen object.

    :param data_path: absoulte path of the data file
    :type data_path: string
    :param na_values: element representing na values
    :type na_values: string
    :rtype: None
    """
    kitchen_feature = get_param('kitchen_feature')
    vr_data = pd.read_csv(data_path, na_values=na_values).dropna()
    for kitchen_name in np.unique(vr_data[kitchen_feature]):
        kitchens[kitchen_name] = Kitchen(kitchen_name, vr_data)


def get_symbolic_location(req):
    """This function answers the request GetSymbolicLocationRequest by getting
    the object storage or destination location.

    :param req: request
    :returns: symbolic location
    :rtype: string
    """

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

    # Set kitchen_name, human_name and context automatically if only
    # one kitchen, human and context is saved.
    if (len(list(kitchens.keys())) == 1):
        kitchen = list(kitchens.values())[0]
        kitchen_name = kitchen.kitchen_id
        if (len(kitchen.humans) == 1):
            human_name = kitchen.humans[0].name
        contexts = np.unique(kitchen.raw_data[get_param('context_feature')])
        if len(contexts) == 1:
            context = contexts[0]

    loginfo('(GetCostmapLocation) Symbolic location lookup for %s', object_id)
    if storage_p:
        # If only the storage of the object is necessary
        location = kitchens[kitchen_name].get_object_location(object_id)
        loginfo("(GetCostmapLocation) Storage location: %s", location)
    else:
        # Else: Get the location of the object for the context, human_name, kitchen
        location = kitchens[kitchen_name].get_object_destination(object_id, context, human_name, table_id)
        loginfo("(GetCostmapLocation) Destination location: %s", location)
    return location


def get_costmap(req):
    """This function answers the request GetCostmapRequest by returning the
     parameters of the learned GMMs modeling different positions and orientations.
     Moreover, the boundaries of the position GMMs are returned.

    :param req: GetCostmapRequest request
    :returns: GMMs parameters and boundaries of the position GMMs
    :rtype: GetCostmapResponse
    """
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

    # Set kitchen_name, human_name and context automatically if only
    # one kitchen, human and context is saved.
    if (len(list(kitchens.keys())) == 1):
        kitchen = list(kitchens.values())[0]
        kitchen_name = kitchen.kitchen_id
        if (len(kitchen.humans) == 1):
            human_name = kitchen.humans[0].name
        contexts = np.unique(kitchen.raw_data[get_param('context_feature')])
        if len(contexts) == 1:
            context_name = contexts[0]

    loginfo("(GetCostmap) Costmap lookup for %s", object_id)
    logdebug("(GetCostmap) with context_name %s", context_name)
    logdebug("(GetCostmap) with human_name %s", human_name)
    logdebug("(GetCostmap) with kitchen_name %s", kitchen_name)
    logdebug("(GetCostmap) with table_id %s", table_id)

    if req.location == kitchens[kitchen_name].get_object_location(object_id):  # k and table_id in k.table_ids:
        return kitchens[kitchen_name].get_storage_costmap(context_name, object_id)
    else:
        return kitchens[kitchen_name].get_destination_costmap(table_id, context_name, human_name, object_id,
                                                              x_object_positions, y_object_positions,
                                                              placed_object_types)


def relation_acc(vritem, relation, relation_name, relation_n):
    """This function calculates the accuracy of the given relation."""
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


def generate_relations_between_items():
    """This function adds the related costmaps to the already created VRItem objects.
    Moreover, it allows to visualizes the learned position and orientation GMMs, as well
    validating the related costmap and visualizes these too."""

    visualize_costmap = get_param('visualize_costmap')
    visualize_related_costmap = get_param('visualize_related_costmap')
    validate_related_costmap = get_param('validate_related_costmap')
    visualize_orientations = get_param('visualize_orientations')

    for kitchen in kitchens.values():

        tables = np.unique(kitchen.raw_data[get_param('table_feature')])
        contexts = np.unique(kitchen.raw_data[get_param('context_feature')])

        for table in tables:
            for context in contexts:

                vritems_per_human = kitchen.get_all_objects(table, context)

                for vritems in vritems_per_human:
                    for vritem in vritems:

                        i = 0
                        # Calculating related costmaps
                        dest_costmaps = list(map(lambda object: object.dest_costmap, vritems[:]))
                        del dest_costmaps[i]
                        vritem.add_related_costmaps(dest_costmaps)

                        if visualize_costmap:
                            vritem.dest_costmap.plot_gmm(plot_in_other=True)
                            vritem.dest_costmap.costmap_to_output_matrix().plot("Destination of " + vritem.object_id)
                            vritem.storage_costmap.plot_gmm(plot_in_other=True)
                            vritem.storage_costmap.costmap_to_output_matrix().plot("Storage of " + vritem.object_id)

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
                                                hist_kws={"density": True, "align": "left"},
                                                norm_hist=False, hist=False,
                                                kde=False, fit=norm)
                                    i += 1
                            fig.add_subplot(111, frameon=False)
                            plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
                            plt.xlabel("degree [rad]")
                            plt.ylabel("density")
                            red_patch = mpatches.Patch(color='red', label='Original orientations')
                            black_patch = mpatches.Patch(color='black', label='Learned distribution')
                            plt.legend(handles=[red_patch, black_patch])
                            plt.show()

                        if visualize_related_costmap:
                            for relation_name, relation in vritem.related_costmaps.items():
                                acc = 0
                                plot_text = relation.relation_name
                                if validate_related_costmap:
                                    acc = relation_acc(vritem, relation.costmap, relation_name, 0)
                                    plot_text += "\n with acc. of " + str(acc) + " for " + vritem.object_id
                                    other_relation_name = relation_name.replace(vritem.object_id, "")[4:-1]
                                    other_vr_item = None
                                    for vri in vritems:
                                        if vri.object_id == other_relation_name:
                                            other_vr_item = vri
                                    acc = relation_acc(other_vr_item, relation.other_costmap, relation_name, 1)
                                    plot_text += "\n with acc. of " + str(acc) + " for " + other_relation_name
                                relation.costmap.costmap_to_output_matrices()[relation.component].plot(plot_in_other=True)
                                relation.other_costmap.costmap_to_output_matrices()[relation.other_component].plot(text=plot_text)

                        i += 1


def init_dataset():
    """This function initializes the GMM models for the given data saved in the resource directory."""
    rospack = RosPack()
    ros_package_name = get_param('package_name')
    ros_package_path = rospack.get_path(ros_package_name)
    csv_file_name = get_param('data_csv_file_name')
    na_values = get_param('na_values')
    fit_data(ros_package_path + '/resource/' + csv_file_name, na_values)
    loginfo("(learning) Initialized data set.")
    generate_relations_between_items()
