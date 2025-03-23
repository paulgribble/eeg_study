import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os
from tqdm import tqdm


def get_rows(participant_range, group):
    row_list = []
    for i in tqdm(participant_range, unit=" participant "):
        fname = "processed_data/"+group+"_group/P"+str(i)+"_signals.csv"
        signals = pd.read_csv(fname, sep=",")
        cp3_pre_p2p = np.max(signals.cp3_pre)-np.min(signals.cp3_pre)
        cp3_post_p2p = np.max(signals.cp3_post)-np.min(signals.cp3_post)
        snap_pre_p2p = np.max(signals.snap_pre)-np.min(signals.snap_pre)
        snap_post_p2p = np.max(signals.snap_post)-np.min(signals.snap_post)
        fname = "processed_data/"+group+"_group/P"+str(i)+"_behaviour.csv"
        behaviour_dict = dict(pd.read_csv(fname, sep=",").mean())
        signals_dict = { "participant"  : "P"+str(i),
                        "group"        : group,
                        "cp3_pre_p2p"  : cp3_pre_p2p,
                        "cp3_post_p2p" : cp3_post_p2p,
                        "snap_pre_p2p" : snap_pre_p2p,
                        "snap_post_p2p": snap_post_p2p}
        signals_dict.update(behaviour_dict)
        row_list.append(signals_dict)
    return row_list


rows_control  = get_rows(range(0,15),  "control")
rows_learning = get_rows(range(15,30), "learning")

summary_table = pd.DataFrame(rows_control + rows_learning)

if not os.path.exists("tables"):
    os.makedirs("tables")

summary_table.to_csv("tables/summary_table.csv", sep=",")
