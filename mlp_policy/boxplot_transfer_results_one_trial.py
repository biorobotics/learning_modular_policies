#!/usr/bin/env python
# coding: utf-8

'''

Source code for paper "Learning modular robot control policies" in Transactions on Robotics
MLP comparisons
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



def shift_left(ax, dy):
    pos = ax.get_position() 
    ax.set_position([pos.x0-dy, pos.y0, pos.width, pos.height])

def print_best_worst(metric_val, urdf_names):
    worst_ind = np.argmin(metric_val)
    best_ind = np.argmax(metric_val)
    print('Best: ' + urdf_names[best_ind])
    print('Worst: ' + urdf_names[worst_ind])



fname1 = os.path.join(os.path.split(os.getcwd())[0], 
        'modular_policy/saved/with_tripod3/transfer_data/transfer_results.ptx')
fname2 = os.path.join(os.path.split(os.getcwd())[0], 
        'modular_policy/saved/no_tripod3/transfer_data/transfer_results.ptx')
fname3 = os.path.join(os.path.split(os.getcwd())[0], 
        'mlp_policy/saved/hc_tripod3/transfer_data/transfer_results.ptx')
fname4 = os.path.join(os.path.split(os.getcwd())[0], 
        'mlp_policy/saved/shared_trunk_tripod3/transfer_data/transfer_results.ptx')



folder = os.path.dirname(fname1)
data1 = torch.load(fname1)
vb = np.array(data1['vel_baseline_list'])
v = np.array(data1['vel_metric_list'])
metric_all = (vb-v)/vb
metric_seen1 = metric_all[data1['seen_inds']]
metric_unseen1 = metric_all[data1['unseen_inds']]

folder = os.path.dirname(fname2)
data2 = torch.load(fname2)
vb = np.array(data2['vel_baseline_list'])
v = np.array(data2['vel_metric_list'])
metric_all = (vb-v)/vb
metric_seen2 = metric_all[data2['seen_inds']]
metric_unseen2 = metric_all[data2['unseen_inds']]

folder = os.path.dirname(fname4)
data4 = torch.load(fname4)
vb = np.array(data4['vel_baseline_list'])
v = np.array(data4['vel_metric_list'])
metric_all4 = (vb-v)/vb

folder = os.path.dirname(fname3)
data3 = torch.load(fname3)
vb = np.array(data3['vel_baseline_list'])
v = np.array(data3['vel_metric_list'])
metric_all3 = (vb-v)/vb
metric_seen3 = metric_all3[data3['seen_inds']]
metric_unseen3 = metric_all3[data3['unseen_inds']]

fig, axs = plt.subplots(1, 4, sharey=True)
fig.set_size_inches(7, 2)
# fig.suptitle('Sharing x per column, y per row')
axs[0].set_title('GNN with\ngait style')
# axs[0].set_title('\n'.join(wrap('GNN with gait style', 10)))
axs[0].boxplot([metric_seen1, metric_unseen1],
           labels=['Training', 'Transfer'],
           whis=(0,100),
           widths=[0.4, 0.4])# does not calculate outliers
axs[0].set_ylabel('Velocity matching metric\n (higher is better)')
print('GNN with gait style training set')
print_best_worst(metric_seen1, [data1['urdf_names'][j] for j in data1['seen_inds']])
print('GNN with gait style transfer set')
print_best_worst(metric_unseen1, [data1['urdf_names'][j] for j in data1['unseen_inds']])



axs[1].set_title('GNN without\ngait style')
axs[1].boxplot([metric_seen2, metric_unseen2],
           labels=['Training', 'Transfer'],
           whis=(0,100),
           widths=[0.4, 0.4])# does not calculate outliers
shift_left(axs[1], 0.02)
print('GNN with gait style training set')
print_best_worst(metric_seen2, [data2['urdf_names'][j] for j in data2['seen_inds']])
print('GNN with gait style transfer set')
print_best_worst(metric_unseen2, [data2['urdf_names'][j] for j in data2['unseen_inds']])



axs[2].set_title('Hardware\nconditioned')
axs[2].boxplot([metric_seen3, metric_unseen3],
           labels=['Training', 'Transfer'],
           whis=(0,100),
           widths=[0.4, 0.4])# does not calculate outliers
shift_left(axs[2], 0.04)
print('Hardware conditioned training')
print_best_worst(metric_seen3, [data3['urdf_names'][j] for j in data3['seen_inds']])
print('Hardware conditioned transfer')
print_best_worst(metric_unseen3, [data3['urdf_names'][j] for j in data3['unseen_inds']])



axs[3].set_title('Shared\ntrunk')
axs[3].boxplot([metric_all4],
           labels=['Training'],
           whis=(0,100),
           widths=[0.4])# does not calculate outliers
# axs[3].set_aspect(2.2)
# shift_left(axs[3], 0.1)
print('Shared trunk training')
print_best_worst(metric_all4, data4['urdf_names'])


for ax in fig.get_axes():
    ax.label_outer()


# file_out = os.path.join(folder,'boxplot.pdf')
# fig.savefig(file_out , bbox_inches='tight')
# print('created ' + file_out)

plt.show()


