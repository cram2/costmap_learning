# coding: utf-8
import numpy as np
import pandas as pd

from human import Human

# db:
humans = []
kitchen = {}

# input file
vr_data = pd.read_csv("/home/thomas/new_nameisthiscsvname_short.csv", na_values="NIL").dropna()

for kitchen_name in np.unique(vr_data["kitchen_name"]):
    data_by_kitchen = vr_data.loc[vr_data["kitchen_name"] == str(kitchen_name)]
    for human_name in np.unique(data_by_kitchen["human_name"]):
        if not (human_name in map(lambda h: h.name, humans)):
            humans.append(Human(str(human_name), data_by_kitchen))