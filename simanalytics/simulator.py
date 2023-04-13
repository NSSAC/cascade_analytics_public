'''
Simulator-specific methods for reading, writing, and modifying diffusion cascades.

AA
'''

import logging
import numpy as np
import pandas as pd
from pdb import set_trace
from time import time

import cascade

SUPPORTED_SIMULATORS = ['EpiHiper']

# A class is defined for every simulator. But these classes have some common
# mandatory methods.
## AA: need to figure out how to enforce common mandatory methods.
### The EpiHiper class
class EpiHiper:
    def __init__(self):
        return

    def sim_to_cascade(self, file_name, ID, 
            cascade_features, 
            nodes=None,
            states_to_consider=None,
            time_horizon=None):
        edges = pd.read_csv(file_name)
        if states_to_consider:
            edges = edges[edges.exit_state.isin(states_to_consider)]
        edges = edges.rename(columns={
            'tick': 'time',
            'contact_pid': 'parent',
            'pid': 'node',
            'exit_state': 'state'
            })
        if nodes is None:
            logging.warning('Found no node features.')
            nodes = cascade.extract_nodes_from_edgelist(edges)
        edges = cascade.source_target_time_to_edges(edges, nodes)
        if time_horizon is not None:
            edges = edges[edges.inf_time<=time_horizon]

        _cascade_features = cascade_features.copy()
        _cascade_features['ID'] = ID
    
        return cascade.Cascade(nodes, edges, _cascade_features)

    def read_node_features(self, infile, return_dict):
        logging.info(f'Reading node features file {infile} ...')
    
        df = pd.read_csv(infile, skiprows=1)
        df.age_group = df.age_group.replace({'o': 'a'})
        df = df.rename(columns={'pid': 'node'})
        return_dict['node_features'] = df
        return 

