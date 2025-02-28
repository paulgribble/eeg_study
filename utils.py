import numpy as np
import matplotlib.pyplot as plt
import math

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

