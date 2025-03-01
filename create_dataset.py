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

def create_SEP(stim, sr=5000, sep_peak=0.020, sep_period=0.008, sep_amplitude = 1.0, jitter_onset=0, noise_sd=0):
    n = np.shape(stim)[0]
    i_stim = np.where(np.diff(stim)==1)[0]
    i = 0 # starting stim
    i_peak = math.ceil(sep_peak * sr)
    i_period = math.ceil(sep_period * sr)
    SEP = np.zeros((n,))
    t = np.linspace(0,sep_period,math.ceil(sep_period*sr))
    sep_single = -np.sin(2*math.pi*t*(1/sep_period)) * np.hanning(math.ceil(sep_period*sr)) * sep_amplitude
    i_offset = i_stim[i] + i_peak - math.ceil(i_period/4)
    if jitter_onset !=0:
        i_offset += np.random.randint(-jitter_onset,jitter_onset)
    while (((i_offset + i_period) < n) and (i < len(i_stim)-1)):
        SEP[i_offset : i_offset + i_period] = sep_single
        i = i + 1
        i_offset = i_stim[i] + i_peak - math.ceil(i_period/4)
    SEP += np.random.randn(len(SEP)) * noise_sd
    return SEP

def write_binary_eeg(data, fname):
    with open(fname, "wb") as f:
        data.tofile(f) # little endian float64 on my apple silicon mac


n_participants = 15
sample_rate    = 5000  # Hz
raw_dir        = "raw_data"

np.random.seed(5)

if not os.path.exists(raw_dir):
    print(f"creating directory {raw_dir}")
    os.makedirs(raw_dir)
if not os.path.exists(raw_dir+"/control_group"):
    print(f"creating directory {raw_dir+"/control_group"}")
    os.makedirs(raw_dir+"/control_group")
if not os.path.exists(raw_dir+"/learning_group"):
    print(f"creating directory {raw_dir+"/learning_group"}")
    os.makedirs(raw_dir+"/learning_group")

# control group
# no difference pre vs post
#
print(f"Generating {n_participants} participants in the control group...")
for i_participant in tqdm(range(n_participants), unit="participant"):
    sep_change  = 0   # pre to post change
    snap_change = 0   # pre to post change
    sep_ampl    = 1.0 + np.random.randn() * 0   # pre amplitude + across-participant variability
    snap_ampl   = 4.0 + np.random.randn() * 0   # pre amplitude + across-participant variability
    sep_noise   = 0.3 # within-participant variability
    snap_noise  = 0.3 # within-participant variability
    trial_duration = 120 + np.random.randint(-5,5)
    participant_num = i_participant
    stim = create_stim(dur=trial_duration, sr=sample_rate, pulse_width=0.001, pulse_freq=3)
    cp3_pre   = create_SEP(stim, sep_peak=0.020, sep_period=0.008, sep_amplitude=sep_ampl, jitter_onset=2, noise_sd=sep_noise)
    write_binary_eeg(cp3_pre, raw_dir+"/control_group/"+"P"+str(participant_num)+"_cp3_pre"+".bin")
    cp3_post  = create_SEP(stim, sep_peak=0.020, sep_period=0.008, sep_amplitude=sep_ampl + sep_change, jitter_onset=2, noise_sd=sep_noise)
    write_binary_eeg(cp3_post, raw_dir+"/control_group/"+"P"+str(participant_num)+"_cp3_post"+".bin")
    snap_pre  = create_SEP(stim, sep_peak=0.005, sep_period=0.003, sep_amplitude=snap_ampl, jitter_onset=0, noise_sd=snap_noise)
    write_binary_eeg(snap_pre, raw_dir+"/control_group/"+"P"+str(participant_num)+"_snap_pre"+".bin")
    snap_post = create_SEP(stim, sep_peak=0.005, sep_period=0.003, sep_amplitude=snap_ampl + snap_change, jitter_onset=0, noise_sd=snap_noise)
    write_binary_eeg(snap_post, raw_dir+"/control_group/"+"P"+str(participant_num)+"_snap_post"+".bin")
    write_binary_eeg(stim, raw_dir+"/control_group/"+"P"+str(participant_num)+"_stim_pre"+".bin")
    write_binary_eeg(stim, raw_dir+"/control_group/"+"P"+str(participant_num)+"_stim_post"+".bin")

# learning group
# yes difference pre vs post
#
print(f"Generating {n_participants} participants in the learning group...")
for i_participant in tqdm(range(n_participants), unit="participant"):
    sep_change  = 0.05 # pre to post change
    snap_change = 0   # pre to post change
    sep_ampl    = 1.0 + np.random.randn() * 0   # pre amplitude + across-participant variability
    snap_ampl   = 4.0 + np.random.randn() * 0   # pre amplitude + across-participant variability
    sep_noise   = 0.7 # within-participant variability
    snap_noise  = 0.7 # within-participant variability
    trial_duration = 120 + np.random.randint(-5,5)
    participant_num = i_participant + n_participants
    stim = create_stim(dur=trial_duration, sr=sample_rate, pulse_width=0.001, pulse_freq=3)
    cp3_pre   = create_SEP(stim, sep_peak=0.020, sep_period=0.008, sep_amplitude=sep_ampl, jitter_onset=2, noise_sd=sep_noise)
    write_binary_eeg(cp3_pre, raw_dir+"/learning_group/"+"P"+str(participant_num)+"_cp3_pre"+".bin")
    cp3_post  = create_SEP(stim, sep_peak=0.020, sep_period=0.008, sep_amplitude=sep_ampl + sep_change, jitter_onset=2, noise_sd=sep_noise)
    write_binary_eeg(cp3_post, raw_dir+"/learning_group/"+"P"+str(participant_num)+"_cp3_post"+".bin")
    snap_pre  = create_SEP(stim, sep_peak=0.005, sep_period=0.003, sep_amplitude=snap_ampl, jitter_onset=0, noise_sd=snap_noise)
    write_binary_eeg(snap_pre, raw_dir+"/learning_group/"+"P"+str(participant_num)+"_snap_pre"+".bin")
    snap_post = create_SEP(stim, sep_peak=0.005, sep_period=0.003, sep_amplitude=snap_ampl + snap_change, jitter_onset=0, noise_sd=snap_noise)
    write_binary_eeg(snap_post, raw_dir+"/learning_group/"+"P"+str(participant_num)+"_snap_post"+".bin")
    write_binary_eeg(stim, raw_dir+"/learning_group/"+"P"+str(participant_num)+"_stim_pre"+".bin")
    write_binary_eeg(stim, raw_dir+"/learning_group/"+"P"+str(participant_num)+"_stim_post"+".bin")






