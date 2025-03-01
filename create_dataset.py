# creates a synthetic EEG dataset
#
# two groups: experimental and control
# 15 participants in each group
# 1 trial "pre" and 1 trial "post" for each participant
# each trial is 120 seconds of data for 3 channels, 4800 Hz sampling rate:
# CP3 electrode (standard EEG montage position above somatosensory cortex)
# SNAP electrode (median nerve at the elbow), 
# and a digital signal (boxcars) indicating median nerve stimulation times
# median nerve stimulation is 1 ms duration at 3 Hz
# the expected somatosensory evoked potential at CP3 is negative peak at 20 ms and 8 ms period negative sinudoid
# the expected SNAP (sensory nerve action potential) is 5 ms negative peak and 3 ms period negative sinusoid

import numpy as np
import matplotlib.pyplot as plt
import math
import os
from tqdm import tqdm

def create_stim(dur=120, sr=5000, pulse_width=0.001, pulse_freq=3):
    n = dur * sr
    stim = np.zeros((n,))
    i = 250 # starting point
    i_width = math.ceil(pulse_width * sr) + 1
    freq_step = math.ceil((1/pulse_freq) * sr)
    while ((i+i_width) < n):
        stim[i+1:i+i_width]=1
        i = i + freq_step
    return stim

def create_SEP(stim, sr=5000, sep_peak=0.020, sep_period=0.010, sep_amplitude = 1.0, noise_sd=0):
    n = np.shape(stim)[0]
    i_stim = np.where(np.diff(stim)==1)[0]
    i = 0   # starting stim
    i_peak = math.ceil(sep_peak * sr)
    i_period = math.ceil(sep_period * sr)
    SEP = np.zeros((n,))
    t = np.linspace(0,sep_period,math.ceil(sep_period*sr))
    sep_single = -np.sin(2*math.pi*t*(1/sep_period)) * np.hanning(math.ceil(sep_period*sr)) * sep_amplitude
    i_offset = i_stim[i] + i_peak + int(i_period/2)
    while (((i_offset + i_period) < n) and (i < len(i_stim)-1)):
        SEP[i_offset : i_offset + i_period] = sep_single
        i = i + 1
        i_offset = i_stim[i] + i_peak + int(i_period/2)
    SEP += np.random.randn(len(SEP)) * noise_sd
    return SEP

def write_binary_eeg(data, fname):
    with open(fname, "wb") as f:
        data.tofile(f) # little endian float64 on my apple silicon mac


n_participants = 15
sample_rate    = 5000  # Hz
raw_dir        = "raw_data"

np.random.seed(9040)

def create_group(participant_list=range(15), folder_name="control_group", sep_change=0, snap_change=0):
    if not os.path.exists(folder_name):
        print(f"creating directory {folder_name}")
        os.makedirs(folder_name)
    print(f"Generating {n_participants} participants in folder {folder_name} ...")
    for participant_num in tqdm(participant_list, unit="participant"):
        sep_change  = sep_change  + np.random.randn() * 0.0   # pre to post change
        snap_change = snap_change + np.random.randn() * 0.0   # pre to post change
        sep_pre     = 1.0 + np.random.randn() * 0.0
        snap_pre    = 4.0 + np.random.randn() * 0.0
        sep_post    = sep_pre  + sep_change 
        snap_post   = snap_pre + snap_change
        sep_noise   = 1.5   # signal noise
        snap_noise  = 1.5   # signal noise
        trial_duration = 120 + np.random.randint(-5,5)
        stim = create_stim(dur=trial_duration, sr=sample_rate, pulse_width=0.001, pulse_freq=3)
        cp3_pre   = create_SEP(stim, sep_peak=0.020, sep_period=0.012, sep_amplitude=sep_pre  , noise_sd=sep_noise)
        cp3_post  = create_SEP(stim, sep_peak=0.020, sep_period=0.012, sep_amplitude=sep_post , noise_sd=sep_noise)
        snap_pre  = create_SEP(stim, sep_peak=0.005, sep_period=0.003, sep_amplitude=snap_pre , noise_sd=snap_noise)
        snap_post = create_SEP(stim, sep_peak=0.005, sep_period=0.003, sep_amplitude=snap_post, noise_sd=snap_noise)
        write_binary_eeg(cp3_pre  , folder_name+"/"+"P"+str(participant_num)+"_cp3_pre"+".bin")
        write_binary_eeg(cp3_post , folder_name+"/"+"P"+str(participant_num)+"_cp3_post"+".bin")
        write_binary_eeg(snap_pre , folder_name+"/"+"P"+str(participant_num)+"_snap_pre"+".bin")
        write_binary_eeg(snap_post, folder_name+"/"+"P"+str(participant_num)+"_snap_post"+".bin")
        write_binary_eeg(stim     , folder_name+"/"+"P"+str(participant_num)+"_stim_pre"+".bin")
        write_binary_eeg(stim     , folder_name+"/"+"P"+str(participant_num)+"_stim_post"+".bin")


# create control group
create_group(range(0 ,15), "raw_data/control_group", sep_change=0, snap_change=0)

# create learning group
create_group(range(15,30), "raw_data/learning_group", sep_change=0.07, snap_change=0)


