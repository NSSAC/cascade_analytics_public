DESC='''
Given a contact network, obtain the map of edges and their primary source-target
activity types.

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

TARGET_SHIFT = 10000000000
#NETWORK = '/project/biocomplexity/cascade_analytics/2023-KDD/network/tn_contact_network_m5_M40_a1000.txt'
#NETWORK = '/project/biocomplexity/cascade_analytics/2023-KDD/network/VA/network.txt'
NETWORK = '../examples/network/contact_network.txt'

tim = timer(unit='seconds')
print('Reading network ...')
#net = pd.read_csv(NETWORK, nrows=100000, skiprows=1)
net = pd.read_csv(NETWORK, skiprows=1)
tim.update_and_display()

print('Preparing map file ...')
df = pd.DataFrame()
df['edge'] = net.targetPID * TARGET_SHIFT + net.sourcePID
df['activity'] = net.targetActivity.str.replace('.*:', '', regex=True) + \
        net.sourceActivity.str.replace('.*:', '', regex=True)
df['duration'] = net.duration
df.activity = df.activity.astype(int)
tim.update_and_display()

# Get max. duration edges only
print('Getting max. duration ...')
ind_max = df.sort_values('duration', ascending=False)['edge'].drop_duplicates().index
df = df.loc[ind_max].reset_index(drop=True)
tim.update_and_display()

print('Writing to feather file ...')
df[['edge', 'activity']].to_feather('edge_activity_map.feather')
