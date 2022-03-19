import numpy as np
import matplotlib.pyplot as plt
from   matplotlib.ticker import MaxNLocator
import math
import h5py

import torch
import torch.nn as nn
import torch.nn.functional as F

import snntorch as snn
import snntorch.spikegen  as spikegen
import snntorch.spikeplot as splt
from   snntorch import surrogate

from models import LCN, LCNSpikingHybrid, LCNSpikingHybrid2

# copy ONV 3 times because RGB is all the same (white ball)
class CopyRedChannel():
	def __call__(self, x):
		x = x.tile((3,))
		return x

def main():

	# Demonstrate LIF basics
	"""

	num_steps = 20

	lif1 = snn.Leaky(0.9, reset_mechanism ='subtract')

	# Initialize membrane, input, and output
	mem = torch.zeros(1)  # U=0 at t=0
	# cur_in = torch.Tensor([0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0])
	cur_in = torch.Tensor([0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, -1, -1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0])
	spk_out = torch.zeros(1)  # initialize output spikes

	# A list to store a recording of membrane potential
	mem_rec = [mem]
	spk_rec = [spk_out]

	# pass updated value of mem and cur_in[step]=0 at every time step
	for step in range(num_steps):
	  spk_out, mem = lif1(cur_in[step], mem)

	  # Store recordings of membrane potential
	  mem_rec.append(mem)
	  spk_rec.append(spk_out)

	# convert the list of tensors into one tensor
	mem_rec = torch.stack(mem_rec)

	f, ax = plt.subplots(3, sharex=True)

	ax[0].get_yaxis().set_visible(False)
	ax[0].xaxis.set_major_locator(MaxNLocator(integer=True))
	# ax[1].get_yaxis().set_visible(False)
	# ax[1].xaxis.set_major_locator(MaxNLocator(integer=True))
	ax[2].get_yaxis().set_visible(False)
	ax[2].xaxis.set_major_locator(MaxNLocator(integer=True))

	ax[0].set_title("Input Spikes", fontsize=24)

	colors = ['red' if x == 1 else 'blue' for x in cur_in]

	colors = []
	spk_plt = []
	time = []
	for i in range(24):
		if cur_in[i] < 0:
			spk_plt.append(1)
			time.append(i)
			colors.append('blue')
		elif cur_in[i] > 0:
			spk_plt.append(1)
			time.append(i)
			colors.append('red')


	# ax[0].scatter(time, spk_plt, c='black', marker='.')
	ax[0].scatter(time, spk_plt, c=colors, marker='.')

	# splt.raster(cur_in, ax[0], s=1.5, c=colors)
	# plt.show()

	ax[1].set_title("Membrane Voltage", fontsize=24)
	ax[1].axhline(y=lif1.threshold, color='r', linestyle='dashed')
	ax[1].plot(mem_rec)
	ax[1].set_ylabel("Voltage (V)", fontsize=20)
	ax[1].tick_params(axis='y', which='major', labelsize=16)
	# plt.show()
	# plt.clf()

	ax[2].set_title("Output Spikes", fontsize=24)

	colors = []
	spk_plt = []
	time = []
	for i, s in enumerate(spk_rec):
		if s > 0:
			spk_plt.append(1)
			time.append(i)
			colors.append('red')

	# ax[2].scatter(time, spk_plt, c='black', marker='.')
	ax[2].scatter(time, spk_plt, c=colors, marker='.')
	ax[2].set_xlabel("Time", fontsize=20)

	plt.xticks(fontsize=18)
	plt.yticks(fontsize=18)
	plt.show()
	# plt.clf()
	"""

	# Demonstrate Rate Coding Basics
	"""
	nSteps = 20
	time = np.arange(nSteps)

	inp = torch.Tensor([0])
	data0 = spikegen.rate(inp, num_steps=nSteps)

	inp = torch.Tensor([0.5])
	data1 = spikegen.rate(inp, num_steps=nSteps)

	inp = torch.Tensor([1])
	data2 = spikegen.rate(inp, num_steps=nSteps)

	f, ax = plt.subplots(3, sharex=True)
	ax[0].get_yaxis().set_visible(False)
	ax[0].xaxis.set_major_locator(MaxNLocator(integer=True))
	ax[1].get_yaxis().set_visible(False)
	ax[1].xaxis.set_major_locator(MaxNLocator(integer=True))
	ax[2].get_yaxis().set_visible(False)
	ax[2].xaxis.set_major_locator(MaxNLocator(integer=True))

	# f.suptitle("Rate Encoding for Different Values")

	ax[0].set_title("0", fontsize=24)
	splt.raster(data0, ax[0], s=15, c="black")

	ax[1].set_title("0.5", fontsize=24)
	splt.raster(data1, ax[1], s=15, c="black")

	ax[2].set_title("1", fontsize=24)
	ax[2].set_xlabel("Time", fontsize=20)
	plt.xticks(fontsize=18)
	splt.raster(data2, ax[2], s=15, c="black")

	plt.show()

	# neuron = snn.Synaptic(0, 1, spike_grad=surrogate.fast_sigmoid())
	# s, m, syn = neuron(inp)
	"""

	# Photoreceptor Image
	"""
	desktop = 'C:/Users/taasi/Desktop'
	coordinates = np.load(f'{desktop}/trainSNNs/coordinates.npy')
	print(coordinates.shape)
	fig, ax = plt.subplots()

	circle1 = plt.Circle((0, 0), 1.25, color='tan')
	ax.add_patch(circle1)
	ax.scatter(coordinates[:, :1], coordinates[:, 1:], c='black', marker='.')
	plt.axis('off')
	plt.show()
	# plt.savefig('retina_log_polar.pdf')
	plt.clf()
	plt.axis('on')
	"""

	# Surrogate Gradient Graphs
	"""
	plt.subplot(2, 2, 1)

	t = np.arange(-1, 1, 0.01)
	y = [0]*100 + [1]*100
	plt.step(t, y, c='blue')

	plt.subplot(2, 2, 2)
	# plt.axvline(x=0, c='red', linestyle='dashed', ymin=0.5, ymax=1)
	y = [0]*99 + [1] + [0]*100
	plt.step(t, y, c='red', linestyle='dashed')
	# plt.ylim(0, 2)
	# plt.show()


	sigmoid = lambda x: x / (1 + 25*np.abs(x))
	sigmoid_dt = lambda x: 1 / (1 + 25*abs(x))**2

	plt.subplot(2, 2, 3)
	plt.plot(t, sigmoid(t), c='blue')
	plt.subplot(2, 2, 4)
	plt.plot(t, sigmoid_dt(t), c='red', linestyle='dashed')
	plt.show()
	"""

	# See Data
	"""
	desktop = 'C:/Users/taasi/Desktop'
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	device = 'cpu'

	idx = 100
	nSteps = 20

	coordinates = np.load(f'{desktop}/trainSNNs/coordinates.npy')

	normal_x = np.load(f'{desktop}/trainSNNs/training_data/siggraph_data/data.npy', mmap_mode='r+')
	normal_y = np.load(f'{desktop}/trainSNNs/training_data/siggraph_data/labels.npy', mmap_mode='r+')

	delta_x = np.load(f'{desktop}/trainSNNs/training_data/siggraph_data/dataDelta.npy', mmap_mode='r+')
	delta_y = np.load(f'{desktop}/trainSNNs/training_data/siggraph_data/labelsDelta.npy', mmap_mode='r+')

	onv_normal_prev = normal_x[idx]
	onv_normal_cur  = normal_x[idx+1]
	onv_delta  = delta_x[idx]

	# should be the same
	angles_normal = normal_y[idx+1]
	angles_delta  = delta_y[idx]
	print(angles_normal, angles_delta, normal_y[idx])

	label = angles_delta

	print(f'Nonzero in previous ONV: {np.count_nonzero(onv_normal_prev)}')
	print(f'Nonzero in current ONV:  {np.count_nonzero(onv_normal_cur)}')
	print(f'Nonzero in delta ONV:    {np.count_nonzero(onv_delta)}')

	## Generate Plots

	onv_normal_prev = torch.Tensor(onv_normal_prev)
	onv_normal_cur  = torch.Tensor(onv_normal_cur)
	onv_delta       = torch.Tensor(onv_delta)

	cmap = plt.get_cmap('viridis', 20)

	plt.scatter(coordinates[:, :1], coordinates[:, 1:], c=onv_normal_prev, marker='.', cmap=cmap, vmin=-1, vmax=1)
	cbar = plt.colorbar(extend='min')
	cbar.ax.tick_params(labelsize=20)
	# plt.show()
	# plt.savefig('onv_normal_prev.pdf')  
	plt.clf()

	plt.scatter(coordinates[:, :1], coordinates[:, 1:], c=onv_normal_cur, marker='.', cmap=cmap, vmin=-1, vmax=1)
	cbar = plt.colorbar(extend='min')
	cbar.ax.tick_params(labelsize=20)
	# plt.show()
	# plt.savefig('onv_normal_cur.pdf')  
	plt.clf()

	plt.scatter(coordinates[:, :1], coordinates[:, 1:], c=onv_delta, marker='.', cmap=cmap, vmin=-1, vmax=1)
	cbar = plt.colorbar(extend='min')
	cbar.ax.tick_params(labelsize=20)
	# plt.show()
	# plt.savefig('onv_delta.pdf')  
	plt.clf()

	# ex37_LCNSpikingHybrid2_spiking_normal_100epoch_k25_L4_surrogate_fastSigmoid
	# ex37_LCNSpikingHybrid2_spiking_delta_100epoch_k25_L4_surrogate_fastSigmoid

	fastSigmoid = surrogate.fast_sigmoid()
	layers = 4
	model_dict = torch.load(f'{desktop}/trainSNNs/model_dicts/ex37_LCNSpikingHybrid2_spiking_normal_100epoch_k25_L4_surrogate_fastSigmoid', map_location=device)
	
	m4 = LCNSpikingHybrid2(4, 43200, 2, 25, 5, 5, 0, 1, False, spikeGrad=fastSigmoid)
	m4.load_state_dict(model_dict)
	m4.to(torch.float)
	m4.to(device)

	d = onv_delta

	d.to(torch.float)
	d.to(device)

	spikes = spikegen.rate(onv_normal_prev, num_steps=nSteps, gain=1)
	latency = spikegen.latency(onv_normal_prev, nSteps, 
												 tau=5, threshold=0.01, 
												 linear=True, normalize=True, clip=True)
	
	f, ax = plt.subplots(1, sharex=True)
	# ax.title.set_text(f'Rate Encoded Delta ONV #{idx}: {np.count_nonzero(onv_delta)}/14400')
	splt.raster(latency[:, 8000:], ax, s=1.5, c="black")
	ax.xaxis.set_major_locator(MaxNLocator(integer=True))
	ax.set_xlabel("Time", fontsize=24)
	ax.set_ylabel("Photoreceptor", fontsize=24)
	# plt.savefig(f'onv_{idx}')

	plt.xticks(fontsize=18)
	plt.yticks(fontsize=18)
	# plt.show()
	plt.clf()
	
	rgb = CopyRedChannel()
	"""

	# Run spiking model on data
	"""

	spikes = rgb(spikes[None, :])

	spikes.to(torch.float)
	spikes.to(device)

	membranes, spikes, out = m4(spikes)
	print("Out", out)

	nameModel = 'LCNSpikingHybrid_L4_delta'

	for l in range(layers):

		thresholds = m4.state_dict()[f'snn.lif{l}.threshold'].numpy()
		
		plt.title(f'Threshold Voltages from Layer {l} of {nameModel}')
		plt.hist(thresholds, bins=50)
		plt.savefig(f'threshold_{idx}_{nameModel}_{l}')
		plt.clf()
		# plt.show() 

		s = spikes[l].detach()
		m = membranes[l].detach().numpy()

		print(s.shape, m.shape)

		sNumpy = s.numpy()
		chartS = []
		for i in range(0, sNumpy.shape[1]):
		    curS = sNumpy[:, i]
		    if np.any(curS):
		        chartS.append(curS)

		chartS = np.array(chartS)

		spikeCounts = np.sum(chartS, axis=1)
		avgRate = np.average(spikeCounts)


		print("HI", avgRate)
		print("Spike", l, chartS.shape[0], "/", sNumpy.shape[1])

		# f, ax = plt.subplots(1, sharex=True)

		# splt.raster(torch.from_numpy(chartS), ax, s=1.5, c="black")
		# ax.title.set_text(f'Spikes from Layer {l} of LCNSpikingHybrid_L4: {chartS.shape[0]} / {sNumpy.shape[1]}')
		# plt.savefig(f'spikes_{idx}_{nameModel}_{l}')
		# # plt.clf()
		# plt.show()

		neuronIdx = []
		chartM = []
		chartT = []
		for i in range(0, m.shape[1]):
		    curM = m[:, i]
		    if not np.all(curM[0] == curM):
		        neuronIdx.append(i)
		        chartM.append(curM)
		        chartT.append(thresholds[i])

		chartM = np.array(chartM)
		chartT = np.array(chartT)
		print("Mem", l, chartM.shape[0], "/", m.shape[1])

		print(chartM.shape)

		snn.spikeplot.traces(torch.from_numpy(chartM.T), dim=(10,10))
		plt.show()
		# plt.clf()

		# plt.plot(chartM.T)
		# plt.title(f'Membrane Voltages from Layer {l} of LCNSpikingHybrid_L4: {chartM.shape[0]} / {m.shape[1]}')
		# plt.hlines(chartT, 0, 20)
		# plt.savefig(f'membranes_{idx}_{nameModel}_{l}')
		# plt.clf()
		# plt.show()
	"""

	# Run LCN on data
	"""

	# ex34_LCN_normal_100epoch_k25
	# ex34_LCN_delta_100epoch_k25

	model_dict = torch.load(f'{desktop}/trainSNNs/model_dicts/ex34_LCN_delta_100epoch_k25', map_location=device)
	
	m = LCN(43200, 2, 25, 5, 5, False)
	m.load_state_dict(model_dict)
	m.to(torch.float)
	m.to(device)

	d = onv_delta
	d = d[None, :]
	d = rgb(d)

	d.to(torch.float)
	d.to(device)

	out, act = m(d)

	print(out)

	for i in range(5):
		print(torch.count_nonzero(act[i]), act[i].size())
	"""

	# Tensorboard Loss Plots
	"""
	data = 'normal'
	phase = 'train'
	num = 2
	model = 'LCN'

	cur = f'tensorboard/run-ex{num:02d}_{model}_{data}-loss_{phase}.csv'

	# d = np.loadtxt(cur, names=True)
	d = np.genfromtxt(cur, dtype=float, delimiter=',', names=True)
	print(d["Step"])
	print(d["Value"])
	"""

	phases = ['train', 'val']

	lines = {
		'train': 'solid',
		'val': 'dashed'
	}

	t = np.arange(100)

	# LCN Baseline
	"""
	data = ['normal', 'delta']
	num = 2
	model = 'LCN'

	colors = {
		'normal': 'r', 
		'delta': 'b'
	}

	for p in phases:
		for d in data:
			cur = f'tensorboard/run-ex{num:02d}_{model}_{d}-loss_{p}.csv'
			curData = np.genfromtxt(cur, dtype=float, delimiter=',', names=True)
			plt.plot(t, curData["Value"], c=colors[d], linestyle=lines[p], label=f'{model}_{d}_{p}')

	plt.title('LCN Normal vs Delta ONV')
	plt.xlabel('Epoch')
	plt.ylabel('Loss - MSE')
	plt.legend()
	plt.show()
	"""

	# names = ['LiNet', 'Hybrid SNN', 'SNN']
	names = ['LiNet', 'SNN']
	phases_long = ['training', 'validation']

	# Normal ONV
	"""
	# models = ['LCN', 'LCNSpikingHybrid_L1', 'LCNSpikingHybrid_L2', 'LCNSpikingHybrid_L3', 'LCNSpikingHybrid_L4']
	# models = ['LCN', 'LCNSpikingHybrid_L4', 'LCNSpikingHybrid2_L4']
	models = ['LCN', 'LCNSpikingHybrid2_L4']
	# nums   = [2] + [22]*4
	# nums = [2] + [35] + [37]
	nums = [2] + [37]
	data   = ['normal']
	colors = ['r', 'g', 'b', 'y', 'purple']

	for i, m in enumerate(models):
		n = nums[i]

		for j, p in enumerate(phases):
			for d in data:
				cur = f'tensorboard/run-ex{n:02d}_{m}_{d}-loss_{p}.csv'
				curData = np.genfromtxt(cur, dtype=float, delimiter=',', names=True)
				plt.plot(t, curData["Value"], c=colors[i], linestyle=lines[p], label=f'{names[i]} {phases_long[j]} loss')

	# plt.title('Normal ONV')
	plt.xlabel('Epoch', fontsize=24)
	plt.ylabel('Loss - MSE', fontsize=24)
	plt.xticks(fontsize=18)
	plt.yticks(fontsize=18)
	plt.legend(fontsize=18)
	plt.show()
	"""


	# Delta ONV
	"""
	# models = ['LCN', 'LCNSpikingHybrid_L1', 'LCNSpikingHybrid_L2', 'LCNSpikingHybrid_L3', 'LCNSpikingHybrid_L4']
	# models = ['LCN', 'LCNSpikingHybrid_L4', 'LCNSpikingHybrid2_L4']
	models = ['LCN', 'LCNSpikingHybrid2_L4']
	# nums   = [2] + [22]*4
	# nums = [2] + [35] + [37]
	nums = [2] + [37]
	data   = ['delta']
	colors = ['r', 'g', 'b', 'y', 'purple']


	for i, m in enumerate(models):
		n = nums[i]

		for p in phases:
			for d in data:
				cur = f'tensorboard/run-ex{n:02d}_{m}_{d}-loss_{p}.csv'
				curData = np.genfromtxt(cur, dtype=float, delimiter=',', names=True)
				plt.plot(t, curData["Value"], c=colors[i], linestyle=lines[p], label=f'{names[i]} {phases_long[j]} loss')

	# plt.title('Delta ONV')
	plt.xlabel('Epoch', fontsize=24)
	plt.ylabel('Loss - MSE', fontsize=24)
	plt.xticks(fontsize=18)
	plt.yticks(fontsize=18)
	plt.legend(fontsize=18)
	plt.show()
	"""


if __name__ == "__main__":
	main()


"""
Nonzero in previous ONV: 9486

torch.Size([20, 8640]) (20, 8640)
Spike 0 2853 / 8640
Mem 0 1896 / 8640
(1896, 20)
torch.Size([20, 1728]) (20, 1728)
Spike 1 706 / 1728
Mem 1 551 / 1728
(551, 20)
torch.Size([20, 345]) (20, 345)
Spike 2 100 / 345
Mem 2 182 / 345
(182, 20)
torch.Size([20, 69]) (20, 69)
Spike 3 8 / 69
Mem 3 30 / 69
(30, 20)
"""

"""
Nonzero in current ONV:  94

torch.Size([20, 8640]) (20, 8640)
Spike 0 820 / 8640
Mem 0 196 / 8640
(196, 20)
torch.Size([20, 1728]) (20, 1728)
Spike 1 425 / 1728
Mem 1 116 / 1728
(116, 20)
torch.Size([20, 345]) (20, 345)
Spike 2 106 / 345
Mem 2 56 / 345
(56, 20)
torch.Size([20, 69]) (20, 69)
Spike 3 12 / 69
Mem 3 32 / 69
(32, 20)
"""

"""
Nonzero in delta ONV:    9580

torch.Size([20, 8640]) (20, 8640)
Spike 0 826 / 8640
Mem 0 208 / 8640
(208, 20)
torch.Size([20, 1728]) (20, 1728)
Spike 1 425 / 1728
Mem 1 121 / 1728
(121, 20)
torch.Size([20, 345]) (20, 345)
Spike 2 105 / 345
Mem 2 59 / 345
(59, 20)
torch.Size([20, 69]) (20, 69)
Spike 3 12 / 69
Mem 3 31 / 69
(31, 20)
"""