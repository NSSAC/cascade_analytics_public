'''
Study-specific methods for modifying cascades.

AA
'''

import logging
from multiprocessing import Process, Manager
import numpy as np
import pandas as pd
from pdb import set_trace
from time import time

import cascade as ca
import simulator as sim

SUPPORTED_STUDIES = ['TestEpiHiper', 'MultiVariant', 'KDD']
NODE_SHIFT = 10000000000

# A class is defined for every study. But these classes have some common
# mandatory methods.
## AA: need to figure out how to enforce common mandatory methods.
### The TestEpiHiper class
class TestEpiHiper:
    def __init__(self, **kwargs):
        self.processes = []
        self.cascade_features = kwargs['cascade_features']
        self.features_for_property = kwargs['features_for_property']

        logging.info(f'Initiating simulator ...')
        self.simulator = sim.EpiHiper()

        with Manager() as manager:
            return_dict = manager.dict()
            p = Process(target=self.simulator.read_node_features, 
                    args=(kwargs['node_features'], return_dict))
            p.start()
            p.join()
            self.nodes = return_dict['node_features']

        return

    def sim_to_cascade(self, file_name, ID, return_dict):
        cc = self.simulator.sim_to_cascade(file_name, ID, 
                self.cascade_features,
                nodes = self.nodes)
        cc.edges = cc.edges.drop('state', axis=1)
        return_dict[ID] = cc
        return

### The MultiVariant class
class MultiVariant:
    def __init__(self, **kwargs):
        self.processes = []
        self.cascade_features = kwargs['cascade_features']
        self.features_for_property = kwargs['features_for_property']

        logging.info(f'Initiating simulator ...')
        self.simulator = sim.EpiHiper()

        # Here, we read all necessary files like network, node features, etc.
        return

    def sim_to_cascade(self, file_name, ID, return_dict):
        self.state_features = pd.DataFrame({
                'var1R': [1, 1, 'recovered from variant 1'],
                'var2R': [2, 2, 'recovered from variant 2'],
                'var1E': [1, 1, 'exposed by variant 1'],
                'var1Isymp': [1, 1, 'symptomatic infected variant 1'],
                'var1Iasymp': [1, 1, 'asymptomatic infected variant 1'],
                'var2E': [2, 2, 'exposed variant 2'],
                'var2Isymp': [2, 2, 'variant 2 infected symptomatic'],
                'var2Iasymp': [2, 2, 'variant 2 infected asymptomatic'],
                'var1var2E': [1, 21, 'exposed by variant 1 after recovered from variant 2'],
                'var1var2Isymp': [1, 21, 'variant 1 symptomatic infected after recovered from variant 2'],
                'var1var2Iasymp': [1, 21, 'variant 1 asymptomatic infected after recovered from variant 2'],
                'var2var1E': [2, 12, 'exposed by variant 2 after recovered from variant 1'],
                'var2var1Isymp': [2, 12, 'variant 2 symptomatic infected after recovered from variant 1'],
                'var2var1Iasymp': [2, 12, 'variant 2 asymptomatic infected after recovered from variant 1']}).transpose().rename(columns={
                    0: 'variant',
                    1: 'variant_sequence'})

        # Only seeding and transmission events are selected.
        # These are study dependent as EpiHiper states can be defined by the user.
        states_to_consider = ['var1E', 'var2E', 'var1var2E', 'var2var1E']

        cascade = self.simulator.sim_to_cascade(file_name, ID, 
                self.cascade_features, 
                states_to_consider=states_to_consider)

        cascade.edges['variant'] = cascade.edges.state.map(
                self.state_features.variant)
        cascade.edges['variant_sequence'] = cascade.edges.state.map(
                        self.state_features.variant_sequence)

        # AA: for speedup, we are removing or converting any object arrays
        cascade.edges = cascade.edges.drop('state', axis=1).astype(
                {'variant': 'int', 'variant_sequence': 'int'})
        if (cascade.edges.dtypes==object).any():
            raise TypeError('Found "object" type in one of the columns. This is uniintented, but not really a problem. It might significantly slow down downstream processes though.')

        return_dict[ID] = cascade
        return


### KDD: for lack of better name
class KDD:
    def __init__(self, **kwargs):
        self.processes = []
        self.cascade_features = kwargs['cascade_features']
        self.features_for_property = kwargs['features_for_property']
        self.coverage = 1
        self.time_horizon = None
        self.random_seed = None

        if 'edge_features' not in kwargs['study_inputs'].keys():
            raise Exception('Edge features file is mandatory.')
        if 'network' not in kwargs['study_inputs'].keys():
            raise Exception('Network file is mandatory.')

        if 'random_seed' in kwargs['study_inputs'].keys():
            self.random_seed = kwargs['study_inputs']['random_seed']
        if 'coverage' in kwargs['study_inputs'].keys():
            self.coverage = kwargs['study_inputs']['coverage']
        if 'time_horizon' in kwargs['study_inputs'].keys():
            self.time_horizon = kwargs['study_inputs']['time_horizon']

        # These take some time to load, hence, doing it in the end.
        self.edge_features = pd.read_feather(kwargs['study_inputs'
            ]['edge_features']).set_index('edge').squeeze()
        self.network = pd.read_feather(kwargs['study_inputs'
            ]['network']).rename(columns={
                'targetPID': 'u',
                'sourcePID': 'v'})
        self.labeled_path_features = {'activity': 'edge'}
        self.max_num_hops = 4

        self.cascade_features['coverage'] = self.coverage
        self.cascade_features['random_seed'] = self.random_seed

        logging.info(f'Initiating simulator ...')
        self.simulator = sim.EpiHiper()

        return

    def sim_to_cascade(self, file_name, ID, return_dict):
        cc = self.simulator.sim_to_cascade(file_name, ID, 
                self.cascade_features,
                time_horizon=self.time_horizon)
        cc.edges = cc.edges.drop('state', axis=1)

        # Sample cascade before doing anything else
        ca.sample_cascade(cc, coverage=self.coverage, 
                seed=self.random_seed)

        if self.edge_features is not None:
            cc.edges['activity'] = 0
            edges_ind = cc.edges.parent != -1
            cc.edges['combined'] = cc.edges.node * NODE_SHIFT + cc.edges.parent
            real_edges = cc.edges[edges_ind]
            cc.edges.loc[edges_ind, 'activity'] = \
                    real_edges.combined.map(self.edge_features)
            cc.edges = cc.edges.drop('combined', axis=1)

        cc.boundary = ca.extract_boundary_nodes(cc.nodes, self.network)

        return_dict[ID] = cc
        return

    def delete_attributes(self):
        del self.network
        del self.edge_features

