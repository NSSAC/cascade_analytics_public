DESC='''
Given a contact network, obtain unique edges.

Input format: 
    targetPID,targetActivity,sourcePID,sourceActivity,duration
    314737001,1:2,314747928,1:3,420
Output format:
    3147370010314747928,23

By: AA
'''

import logging
import pandas as pd
from pdb import set_trace
import numpy as np
from time import time

class timer:
    def __init__(self, unit='minutes'):
        self.unit = unit
        self.start = time()
        self.current = self.start
        self.diff = None

    def update(self):
        curr = time()
        self.diff = curr - self.current
        self.current = curr

    def display(self):
        if self.unit == 'minutes':
            print(f'Time in minutes: {self.diff/60: .0f}')
        elif self.unit == 'seconds':
            print(f'Time in seconds: {self.diff: .0f}')
        else:
            print(f'Unsupported time unit {self.unit}.')

    def update_and_display(self):
        self.update()
        self.display()

# NETWORK = '/project/biocomplexity/cascade_analytics/2023-KDD/network/TN/tn_contact_network_m5_M40_a1000.txt'
# NETWORK = '/project/biocomplexity/cascade_analytics/2023-KDD/network/VA/network.txt'
NETWORK = '../examples/network/contact_network.txt'

tim = timer(unit='seconds')
print('Reading network ...')
#net = pd.read_csv(NETWORK, nrows=100000, skiprows=1, usecols=['targetPID', 'sourcePID'])
net = pd.read_csv(NETWORK, skiprows=1, usecols=['targetPID', 'sourcePID'])
tim.update_and_display()

print('Order edges ...')
swap_ind = net.targetPID < net.sourcePID
net.loc[swap_ind, ['targetPID', 'sourcePID']] = \
        net.loc[swap_ind, ['sourcePID', 'targetPID']].values
tim.update_and_display()

print('Drop duplicates ...')
print('Before:', net.shape)
net = net.drop_duplicates()
print('After:', net.shape)
tim.update_and_display()

print('Writing to feather file ...')
net.reset_index(drop=True).to_feather('unique_edges.feather')
tim.update_and_display()
