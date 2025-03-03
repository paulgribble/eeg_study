# process_data.py
#
# depends on raw_data/ directory with control_group/ and learning_group/ subdirectories

import os
import math
import numpy as np
from tqdm import tqdm
import shutil

# read in a .bin file into a numpy array
#
def read_binary_eeg(fname):
    with open(fname, 'rb') as f:
        data = np.fromfile(f, dtype='<f8') # little endian float64 on my apple silicon mac
    return data

# slice recording around stimulation onsets and return a numpy array
# n x t where n is number of stimulations and t is time steps
#
def clip_SEP(stim, sep, sample_rate=5000, pre_clip=0.010, post_clip=0.090):
	stim_up = np.where(np.diff(stim)==1)[0]
	n_stims = len(stim_up)
	pre_clip_n  = int(pre_clip * sample_rate)  # samples
	post_clip_n = int(post_clip * sample_rate) # samples
	sep_clipped = np.zeros((n_stims, pre_clip_n + post_clip_n))
	for i in range(n_stims):
		i1 = stim_up[i] - pre_clip_n
		i2 = stim_up[i] + post_clip_n
		sep_clipped[i,:] = sep[i1:i2]
	return sep_clipped, n_stims

# load in .bin files, clip around each stim, and save in n x t .csv files
# where n is number of stimulations and t is time steps
#
def process_group(raw_dir, processed_dir, participant_list):
    if not os.path.exists(processed_dir):
        print(f"creating directory {processed_dir}")
        os.makedirs(processed_dir)
    for i_participant in tqdm(participant_list, unit="participant"):
        fname_prefix         = raw_dir + "P" + str(i_participant) + "_"
        stim                 = read_binary_eeg(fname_prefix + "stim_pre.bin")
        cp3_pre              = read_binary_eeg(fname_prefix + "cp3_pre.bin")
        snap_pre             = read_binary_eeg(fname_prefix + "snap_pre.bin")
        cp3_post             = read_binary_eeg(fname_prefix + "cp3_post.bin")
        snap_post            = read_binary_eeg(fname_prefix + "snap_post.bin")
        cp3_pre_c, n_stims   = clip_SEP(stim, cp3_pre)
        cp3_post_c, n_stims  = clip_SEP(stim, cp3_post)
        snap_pre_c, n_stims  = clip_SEP(stim, snap_pre)
        snap_post_c, n_stims = clip_SEP(stim, snap_post)
        np.savetxt(processed_dir + "P" + str(i_participant) + "_cp3_pre.csv"  , cp3_pre_c  , delimiter=",")
        np.savetxt(processed_dir + "P" + str(i_participant) + "_cp3_post.csv" , cp3_post_c , delimiter=",")
        np.savetxt(processed_dir + "P" + str(i_participant) + "_snap_pre.csv" , snap_pre_c , delimiter=",")
        np.savetxt(processed_dir + "P" + str(i_participant) + "_snap_post.csv", snap_post_c, delimiter=",")
        # copy over .csv behavioural files as is
        shutil.copy(raw_dir       + "P" + str(i_participant) + "_sequence_times_pre.csv", \
                    processed_dir + "P" + str(i_participant) + "_sequence_times_pre.csv")
        shutil.copy(raw_dir       + "P" + str(i_participant) + "_sequence_times_post.csv", \
                    processed_dir + "P" + str(i_participant) + "_sequence_times_post.csv")

# process control_group and learning_group
#
process_group("raw_data/control_group/" , "processed_data/control_group/" , range(0 ,15))
process_group("raw_data/learning_group/", "processed_data/learning_group/", range(15,30))


