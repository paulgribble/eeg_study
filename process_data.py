
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os
from tqdm import tqdm

def read_binary_eeg(fname):
	with open(fname, 'rb') as f:
		data = np.fromfile(f, dtype='<f8') # little endian float64 on my apple silicon mac
	return data

def slice_eeg(stim, eeg, sample_rate=5000, pre_slice=0.010, post_slice=0.090):
	stim_up      = np.where(np.diff(stim)==1)[0] + 1
	n_stims      = len(stim_up)
	pre_slice_n  = int(pre_slice * sample_rate)  # samples
	post_slice_n = int(post_slice * sample_rate) # samples
	eeg_sliced   = np.zeros((n_stims, pre_slice_n + post_slice_n))
	for i in range(n_stims):
		i1 = stim_up[i] - pre_slice_n
		i2 = stim_up[i] + post_slice_n
		eeg_sliced[i,:] = eeg[i1:i2]
	return eeg_sliced

def process_participant(participant, raw_folder, processed_folder):
	pre_slice   = 0.010 # (s)
	post_slice  = 0.090 # (s)
	sample_rate = 5000  # (Hz)
	# read data
	stim_pre  = read_binary_eeg(raw_folder+"/"+participant+"_stim_pre.bin")
	cp3_pre   = read_binary_eeg(raw_folder+"/"+participant+"_cp3_pre.bin")
	snap_pre  = read_binary_eeg(raw_folder+"/"+participant+"_snap_pre.bin")
	stim_post = read_binary_eeg(raw_folder+"/"+participant+"_stim_post.bin")
	cp3_post  = read_binary_eeg(raw_folder+"/"+participant+"_cp3_post.bin")
	snap_post = read_binary_eeg(raw_folder+"/"+participant+"_snap_post.bin")
	# slice data
	cp3_pre_sliced   = slice_eeg(stim_pre , cp3_pre  , sample_rate=sample_rate, pre_slice=pre_slice, post_slice=post_slice)
	snap_pre_sliced  = slice_eeg(stim_pre , snap_pre , sample_rate=sample_rate, pre_slice=pre_slice, post_slice=post_slice)
	cp3_post_sliced  = slice_eeg(stim_post, cp3_post , sample_rate=sample_rate, pre_slice=pre_slice, post_slice=post_slice)
	snap_post_sliced = slice_eeg(stim_pre , snap_post, sample_rate=sample_rate, pre_slice=pre_slice, post_slice=post_slice)
	# average over stims
	cp3_pre_sliced_m   = np.mean(cp3_pre_sliced, 0)
	cp3_post_sliced_m  = np.mean(cp3_post_sliced, 0)
	snap_pre_sliced_m  = np.mean(snap_pre_sliced, 0)
	snap_post_sliced_m = np.mean(snap_post_sliced, 0)
	# save to .csv files
	if not os.path.exists(processed_folder):
		os.makedirs(processed_folder)
	cp3  = np.vstack((cp3_pre_sliced_m, cp3_post_sliced_m)).T
	snap = np.vstack((snap_pre_sliced_m, snap_post_sliced_m)).T
	signals = np.hstack((cp3,snap))
	np.savetxt(processed_folder+"/"+participant+"_signals.csv" , signals , delimiter=",", header="cp3_pre,cp3_post,snap_pre,snap_post", comments='')
	# behavioural .csv data files
	behav_pre  = np.genfromtxt(raw_folder+"/"+participant+"_sequence_times_pre.csv", delimiter=",")
	behav_post = np.genfromtxt(raw_folder+"/"+participant+"_sequence_times_post.csv", delimiter=",")
	behav = np.hstack((behav_pre, behav_post))
	np.savetxt(processed_folder+"/"+participant+"_behaviour.csv", behav, delimiter=",", header="sequence1_pre, sequence2_pre, sequence3_pre, sequence1_post, sequence2_post, sequence3_post", comments='')
	# make a figure
	fig = plt.figure(layout="constrained", figsize=(10,6))
	gs = GridSpec(2,3, figure=fig)
	t = (np.arange(np.shape(cp3_pre_sliced_m)[0]) / sample_rate) - pre_slice
	ax1 = fig.add_subplot(gs[0,0:2])
	ax1.plot(t, cp3[:,1],'r-',label="POST")
	ax1.plot(t, cp3[:,0],'b-',label="PRE")
	ax1.plot([0,0],[-1,1],'k-')
	ax1.legend()
	ax1.set_ylabel('EVOKED RESPONSE')
	ax1.set_title(f'{raw_folder}/{participant}: CP3 Electrode')
	ax2 = fig.add_subplot(gs[1,0:2])
	ax2.plot(t, snap[:,1],'r-',label="POST")
	ax2.plot(t, snap[:,0],'b-',label="PRE")
	ax2.plot([0,0],[-1,1],'k-')
	ax2.legend()
	ax2.set_ylabel('EVOKED RESPONSE')
	ax2.set_xlabel('TIME (s)')
	ax2.set_title(f'{raw_folder}/{participant}: SNAP Electrode')
	ax3 = fig.add_subplot(gs[0:2,2])
	ax3.boxplot(behav, tick_labels=("S1","S1","S2","S2","S3","S3"))
	ax3.plot(np.random.rand(np.shape(behav)[0])*.04 + 1, behav[:,0], 'b.', alpha=0.4)
	ax3.plot(np.random.rand(np.shape(behav)[0])*.04 + 2, behav[:,3], 'r.', alpha=0.4)
	ax3.plot(np.random.rand(np.shape(behav)[0])*.04 + 3, behav[:,1], 'b.', alpha=0.4)
	ax3.plot(np.random.rand(np.shape(behav)[0])*.04 + 4, behav[:,4], 'r.', alpha=0.4)
	ax3.plot(np.random.rand(np.shape(behav)[0])*.04 + 5, behav[:,2], 'b.', alpha=0.4)
	ax3.plot(np.random.rand(np.shape(behav)[0])*.04 + 6, behav[:,5], 'r.', alpha=0.4)
	ymin,ymax = ax3.get_ylim()
	ax3.plot([2.5, 2.5], [ymin,ymax], 'k-', linewidth=0.5)
	ax3.plot([4.5, 4.5], [ymin,ymax], 'k-', linewidth=0.5)
	ax3.set_title('Behaviour')
	ax3.set_ylabel('TIME (s)')
	ax3.set_xlabel('GROUP')
	fig.savefig(f"{processed_folder}/{participant}.png",dpi=300)
	plt.close(fig)


# PROCESS THE DATA!

for i in tqdm(range(0, 15), unit=" participant "):
	process_participant("P"+str(i), "raw_data/control_group", "processed_data/control_group")

for i in tqdm(range(15,30), unit=" participant "):
	process_participant("P"+str(i), "raw_data/learning_group", "processed_data/learning_group")


