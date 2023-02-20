#!/usr/bin/env python
# coding: utf-8

'''
Source code for paper "Learning modular robot control policies" in Transactions on Robotics
Julian Whitman, Dec. 2022. 
MLP comparisons
Create boxplots from the results
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



# # Create one set of boxplots
# fname = '/home/cobracommander/modular_mbrl/mbrl_v8_test9/zero_shot_data/transfer_results.ptx'
# # fname = '/home/cobracommander/modular_mbrl/mbrl_v8_test9NT2/data_seen/transfer_results.ptx'
# folder = os.path.dirname(fname)
# data = torch.load(fname)
# vb = np.array(data['vel_baseline_list'])
# v = np.array(data['vel_metric_list'])
# metric_all = (vb-v)/vb
# metric_seen = metric_all[data['seen_inds']]
# metric_unseen = metric_all[data['unseen_inds']]
# fig1, ax1 = plt.subplots()
# fig1.set_size_inches(4, 2)
# # ax1.set_title('Zero-shot policy transfer')
# ax1.set_title('With gait style objective')
# ax1.boxplot([metric_seen, metric_unseen],
#            labels=['Training (12 designs)', 'Transfer (132 designs)'],
#            whis=(0,100))# does not calculate outliers
# ax1.set_ylabel('Velocity matching metric')
# fig1.savefig(os.path.join(folder,'boxplot.pdf'))
# plt.show()

# # Create two sets of boxplots

fname1 = '/home/cobracommander/modular_mbrl/mbrl_shared_trunk3/zero_shot_data/transfer_results.ptx'
# fname2 = '/home/cobracommander/modular_mbrl/mbrl_v8_test9NT2/data_seen/transfer_results.ptx'

folder = os.path.dirname(fname1)
data = torch.load(fname1)
vb = np.array(data['vel_baseline_list'])
v = np.array(data['vel_metric_list'])
metric_all = (vb-v)/vb
# metric_seen1 = metric_all[data['seen_inds']]
# metric_unseen1 = metric_all[data['unseen_inds']]

# folder = os.path.dirname(fname2)
# data = torch.load(fname2)
# vb = np.array(data['vel_baseline_list'])
# v = np.array(data['vel_metric_list'])
# metric_all = (vb-v)/vb
# metric_seen2 = metric_all[data['seen_inds']]
# metric_unseen2 = metric_all[data['unseen_inds']]

fig, (ax1, ax2) = plt.subplots(1, 2)
fig.set_size_inches(5, 2)
# fig.suptitle('Sharing x per column, y per row')
ax1.set_title('Shared trunk, 12 robots')
ax1.boxplot(metric_all,
           labels=['Training'],
           whis=(0,100))# does not calculate outliersax2.plot(x, y**2, 'tab:orange')
# ax2.set_title('Without gait style objective')
# ax2.boxplot([metric_seen2, metric_unseen2],
#            labels=['Training', 'Transfer'],
#            whis=(0,100))# does not calculate outliersax4.plot(x, -y**2, 'tab:red')
ax1.set_ylabel('Velocity matching metric\n (higher is better)')
for ax in fig.get_axes():
    ax.label_outer()
# ax1.sharey(ax2)
file_out = os.path.join(folder,'boxplot.pdf')
fig.savefig(file_out)
print('created ' + file_out)
plt.show()


