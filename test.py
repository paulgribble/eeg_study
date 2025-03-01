import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math
import os
from tqdm import tqdm
from scipy import stats

def read_binary_eeg(fname):
    with open(fname, 'rb') as f:
        data = np.fromfile(f, dtype='<f8') # little endian float64 on my apple silicon mac
    return data

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

def load_group(data_dir, participant_range):
	group_df = pd.DataFrame(columns=["participant", "n_stims", "cp3_change", "snap_change"])
	print(f"loading {len(participant_range)} participants from {data_dir}")
	for i_participant in tqdm(participant_range, unit="participant"):
		fname_prefix = data_dir + "P" + str(i_participant) + "_"
		stim      = read_binary_eeg(fname_prefix + "stim_pre.bin")
		cp3_pre   = read_binary_eeg(fname_prefix + "cp3_pre.bin")
		snap_pre  = read_binary_eeg(fname_prefix + "snap_pre.bin")
		cp3_post  = read_binary_eeg(fname_prefix + "cp3_post.bin")
		snap_post = read_binary_eeg(fname_prefix + "snap_post.bin")
		cp3_pre_c, n_stims   = clip_SEP(stim, cp3_pre)
		cp3_post_c, n_stims  = clip_SEP(stim, cp3_post)
		snap_pre_c, n_stims  = clip_SEP(stim, snap_pre)
		snap_post_c, n_stims = clip_SEP(stim, snap_post)
		cp3_pre_c_m = np.mean(cp3_pre_c,0)
		cp3_post_c_m = np.mean(cp3_post_c,0)
		snap_pre_c_m = np.mean(snap_pre_c,0)
		snap_post_c_m = np.mean(snap_post_c,0)
		cp3_change = (np.max(cp3_post_c_m) - np.min(cp3_post_c_m)) - (np.max(cp3_pre_c_m) - np.min(cp3_pre_c_m))
		snap_change = (np.max(snap_post_c_m) - np.min(snap_post_c_m)) - (np.max(snap_pre_c_m) - np.min(snap_pre_c_m))
		group_df.loc[i_participant] = [i_participant, n_stims, cp3_change, snap_change]
	return group_df

def plot_plus_stats(df, colname):
	fig = plt.figure(figsize=(2, 6))
	sns.boxplot(x='group', y=colname, data=df, 
				width=0.3, fliersize=0, color='lightgray', showmeans=True)
	sns.violinplot(x='group', y=colname, data=df, 
				inner=None, color='lightblue', alpha=0.5)
	sns.stripplot(x='group', y=colname, data=df, 
				jitter=True, size=6, alpha=0.5, color='black')
	plt.xlabel('GROUP')
	plt.ylabel(colname)
	plt.grid(True)
	plt.tight_layout()
	plt.show()
	result = stats.ttest_1samp(df[colname][df.group=="control"], 0)
	print(f"control vs zero: t({result.df})={result.statistic:.5f}, p={result.pvalue:.5f}")
	result = stats.ttest_1samp(df[colname][df.group=="learning"], 0)
	print(f"learning vs zero: t({result.df})={result.statistic:.5f}, p={result.pvalue:.5f}")
	result = stats.ttest_ind(df[colname][df.group=="learning"], df[colname][df.group=="control"])
	print(f"learning vs control: t({result.df})={result.statistic:.5f}, p={result.pvalue:.5f}")
	return fig


# SUMMARY FIGURES
#
control_df = load_group("raw_data/control_group/", range(0,15))
control_df["group"] = "control"
learning_df = load_group("raw_data/learning_group/", range(15,30))
learning_df["group"] = "learning"
all_df = pd.concat([control_df, learning_df], ignore_index=True)
#
fig_cp3 = plot_plus_stats(all_df, "cp3_change")
fig_snap = plot_plus_stats(all_df, "snap_change")
fig_cp3.set_figwidth(4)
fig_cp3.set_figheight(6)
fig_cp3.tight_layout()
fig_cp3.savefig("fig_cp3_summary.png",dpi=300)

# EXAMPLE TIMESERIES FIGURE
#
data_dir = "raw_data/learning_group/"
i_participant = 17
fname_prefix = data_dir + "P" + str(i_participant) + "_"
stim      = read_binary_eeg(fname_prefix + "stim_pre.bin")
cp3_pre   = read_binary_eeg(fname_prefix + "cp3_pre.bin")
cp3_post  = read_binary_eeg(fname_prefix + "cp3_post.bin")
cp3_pre_c, n_stims   = clip_SEP(stim, cp3_pre)
cp3_post_c, n_stims  = clip_SEP(stim, cp3_post)
cp3_pre_c_m = np.mean(cp3_pre_c,0)
cp3_post_c_m = np.mean(cp3_post_c,0)
t = (np.arange(np.shape(cp3_pre_c_m)[0]) / 5000) - 0.020
#
plt.figure(figsize=(10,3))
plt.plot(t, cp3_post_c_m,'r-',label="POST")
plt.plot(t, cp3_pre_c_m,'b-',label="PRE")
plt.plot([0,0],[-1,1],'k-')
plt.legend()
plt.xlabel('TIME (s)')
plt.ylabel('EVOKED RESPONSE')
plt.title(f'P{i_participant} (Learning Group): CP3 Electrode')
plt.tight_layout()
plt.savefig("fig_cp3_timeseries.png",dpi=300)

