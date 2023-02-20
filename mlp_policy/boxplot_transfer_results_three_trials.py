#!/usr/bin/env python
# coding: utf-8

'''

Source code for paper "Learning modular robot control policies" in Transactions on Robotics
MLP comparisons boxplot creation
Julian Whitman, Dec. 2022. 

'''


import numpy as np
import torch
import matplotlib.pyplot as plt
import os

font = {'family':'serif', 'serif': ['Times New Roman']}
plt.rc('font',**font)
# to get the font change to work, I used:
# sudo apt install msttcorefonts -qq
# rm ~/.cache/matplotlib -rf
# then ran -- # python -c "import matplotlib" -- to let it rebuild the font cache

x_label_size = 11
def shift_left(ax, dy):
    pos = ax.get_position() 
    ax.set_position([pos.x0-dy, pos.y0, pos.width, pos.height])


fnames_wt = [os.path.join(os.path.split(os.getcwd())[0], 
        'modular_policy/saved/with_tripod'+str(i)+'/transfer_data/transfer_results.ptx') for i in [1,2,3]]
fnames_nt = [os.path.join(os.path.split(os.getcwd())[0], 
        'modular_policy/saved/no_tripod'+str(i)+'/transfer_data/transfer_results.ptx') for i in [1,2,3]]
fnames_hc = [os.path.join(os.path.split(os.getcwd())[0], 
        'mlp_policy/saved/hc_tripod'+str(i)+'/transfer_data/transfer_results.ptx') for i in [1,2,3]]
fnames_st = [os.path.join(os.path.split(os.getcwd())[0], 
        'mlp_policy/saved/shared_trunk_tripod'+str(i)+'/transfer_data/transfer_results.ptx') for i in [1,2,3]]
print(fnames_wt)

fig, axs = plt.subplots(1, 4, sharey=True)
fig.set_size_inches(7, 2)

axs[0].set_title('Modular policy\nwith gait style')
# axs[0].set_title('GNN with gait style', wrap=True)
train_data = []
transfer_data= []
for i in range(3):
    fname=  fnames_wt[i]
    folder = os.path.dirname(fname)
    data = torch.load(fname)
    vb = np.array(data['vel_baseline_list'])
    v = np.array(data['vel_metric_list'])
    metric_all = (vb-v)/vb
    train_data.append( metric_all[data['seen_inds']] )
    transfer_data.append( metric_all[data['unseen_inds']])

axs[0].boxplot(train_data+transfer_data,
           labels=['1', '2\nTraining', '3', '1', '2\nTransfer', '3'],
           whis=(0,100),
           widths=[0.4]*6)# does not calculate outliers
axs[0].axvline(x=3.5, color = 'black', linewidth = '0.5', linestyle='dashed')
# axs[0].set_ylabel('Velocity matching metric\n (higher is better)')
axs[0].set_ylabel('Velocity matching metric', fontsize='large')
plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8])
shift_left(axs[0], 0.02)
axs[0].tick_params(axis='x', labelsize=x_label_size)


axs[1].set_title('Modular policy\nwithout gait style')
train_data = []
transfer_data= []
for i in range(3):
    fname=  fnames_nt[i]
    folder = os.path.dirname(fname)
    data = torch.load(fname)
    vb = np.array(data['vel_baseline_list'])
    v = np.array(data['vel_metric_list'])
    metric_all = (vb-v)/vb
    train_data.append( metric_all[data['seen_inds']] )
    transfer_data.append( metric_all[data['unseen_inds']])

axs[1].boxplot(train_data+transfer_data,
           labels=['1', '2\nTraining', '3', '1', '2\nTransfer', '3'],
           whis=(0,100),
           widths=[0.4]*6)# does not calculate outliers
axs[1].axvline(x=3.5, color = 'black', linewidth = '0.5', linestyle='dashed')
shift_left(axs[1], 0.04)
axs[1].tick_params(axis='x', labelsize=x_label_size)



axs[2].set_title('Hardware\nconditioned')
train_data = []
transfer_data= []
for i in range(3):
    fname=  fnames_hc[i]
    folder = os.path.dirname(fname)
    data = torch.load(fname)
    vb = np.array(data['vel_baseline_list'])
    v = np.array(data['vel_metric_list'])
    metric_all = (vb-v)/vb
    train_data.append( metric_all[data['seen_inds']] )
    transfer_data.append( metric_all[data['unseen_inds']])

axs[2].boxplot(train_data+transfer_data,
           labels=['1', '2\nTraining', '3', '1', '2\nTransfer', '3'],
           whis=(0,100),
           widths=[0.4]*6)# does not calculate outliers
axs[2].axvline(x=3.5, color = 'black', linewidth = '0.5', linestyle='dashed')
shift_left(axs[2], 0.06)
axs[2].tick_params(axis='x', labelsize=x_label_size)



axs[3].set_title('Shared\ntrunk')
train_data = []
for i in range(3):
    fname=  fnames_hc[i]
    folder = os.path.dirname(fname)
    data = torch.load(fname)
    vb = np.array(data['vel_baseline_list'])
    v = np.array(data['vel_metric_list'])
    metric_all = (vb-v)/vb
    train_data.append( metric_all )
axs[3].boxplot(train_data,
           labels=['1', '2\nTraining', '3'],
           whis=(0,100),
           widths=[0.4]*3)# does not calculate outliers
axs[3].set_aspect(7.5)
shift_left(axs[3], 0.125)
axs[3].tick_params(axis='x', labelsize=x_label_size)


# for ax in fig.get_axes():
#     ax.label_outer()


# file_out = os.path.join(folder,'boxplot_three_trials.pdf')
file_out = 'boxplot_three_trials.pdf'
fig.savefig(file_out , bbox_inches='tight')
print('created ' + file_out)

plt.show()


