DESC="""Defines cascade graph class and provides helper functions. A
cascade graph consists of
(1) node list with features, and
(2) time-expanded directed edge list with features.
Even though this is very generic, more constraints or properties will be added
in the future.

By: AA
"""

import argparse
from json import load, dump
import logging
import networkx as nx
import numpy as np
import pandas as pd
from pdb import set_trace

# AA: probably needs to become simulator specific
SEED_IND = -1

FORMAT="%(levelname)s:%(filename)s:%(funcName)s:%(message)s"


# Cascade class that defines the structure of the cascade.
# AA: Currently, there is only one class
# Inputs nodes and edges are pandas dataframes with some mandatory columns.
# Cascade features is a dictionary with some mandatory keys. It can for example
# store representative input parameter values from experiments.
class Cascade:
    def __init__(self, nodes=None, edges=None, cascade_features=None):
        self.mandatory_node_features = ['node']
        self.mandatory_edge_features = \
                ['node', 'parent', 'parent_inf_time', 'inf_time']
        self.mandatory_cascade_features = ['ID']

        self.boundary = None
        self.nodes = nodes
        self.edges = edges
        self.cascade_features = cascade_features

        logging.debug('Checking for duplicate nodes ...') 
        if not self.nodes['node'].is_unique:
            logging.error('Found duplicate nodes.')
            raise

        logging.debug('Checking for isolated nodes and removing them ...') 
        non_isolated_nodes = extract_nodes_from_edgelist(edges)
        isolated_nodes = set(nodes.node).difference(non_isolated_nodes.node)
        if isolated_nodes:
            logging.warning(f'Found {len(isolated_nodes)} isolated nodes. Removing them.')
            self.nodes = self.nodes[~nodes.node.isin(isolated_nodes)]

        logging.debug('Checking for mandatory features ...')
        try:
            for attrib in self.mandatory_node_features:
                if not attrib in nodes.columns:
                    logging.error(f"Mandatory column '{attrib}' not found in nodes.")
                    raise
        except Exception as err:
            logging.error(f"While checking for mandatory columns in nodes, found the following error:\n{err}\nNeed to pass a dataframe 'nodes' with mandatory columns {self.mandatory_node_features}.")
            raise

        try:
            for attrib in self.mandatory_edge_features:
                if not attrib in edges.columns:
                    logging.error(f"Mandatory column '{attrib}' not found in nodes.")
                    raise
        except Exception as err:
            logging.error(f"While checking for mandatory columns in edges, found the following error:\n{err}\nNeed to pass a dataframe 'edges' with mandatory columns {self.mandatory_edge_features}..")
            raise

        try:
            for attrib in self.mandatory_cascade_features:
                if not attrib in self.cascade_features.keys():
                    logging.error(f"Mandatory key '{attrib}' not found in features.")
                    raise
        except Exception as err:
            logging.error(f"While checking for mandatory columns in cascade features, found the following error:\n{err}\nNeed to pass a dictionary 'features' with mandatory keys {self.mandatory_cascade_features}.")
            raise

        logging.debug('Checking for common feature names between node and edge feature lists ...')
        common_attribs = set(nodes.columns).intersection(edges.columns)
        if common_attribs != {'node'}:
            logging.error(f'Feature names {common_attribs} are common to nodes and edges. They have to be distinct.')
            raise

        # The cascade must be acyclic regardless of the underlying diffusion
        # process as it is time-expanded. We check for this.
        logging.debug('Checking if cascade is acyclic ...')
        _edges = pd.DataFrame()
        _edges['source'] = edges[['parent', 'parent_inf_time']].apply(
                tuple, axis=1)
        _edges['target'] = edges[['node', 'inf_time']].apply(
                tuple, axis=1)
        G = nx.from_pandas_edgelist(_edges, create_using=nx.DiGraph())

        if not nx.is_directed_acyclic_graph(G):
            logging.error(f'The cascade {cascade.cascade_features["ID"]} is not acyclic.')
            raise
        logging.debug('Done')
        return

    # write cascade to json file that contains nodes, edges, and features.
    def write_to_file(self, file_name=None):
        cascade_dict = {}
        cascade_dict['nodes'] = self.nodes.to_dict(orient='list')
        cascade_dict['edges'] = self.edges.to_dict(orient='list')
        cascade_dict['cascade_features'] = self.cascade_features
    
        if file_name is None:
            return cascade_dict
        else:
            with open(file_name,'w') as f:
                dump(cascade_dict, f, indent=4)
        return

# Get node list from edge list
### If a node is a seed, then the parent for this node will be a seed indicator
### like -1.
def extract_nodes_from_edgelist(edges, seed_ind=SEED_IND):
    nodes = pd.concat([edges.parent, edges.node]
            ).drop_duplicates().to_frame(name='node')
    return nodes[nodes['node'] != seed_ind]

# Read cascade from json file that contains nodes, edges, and features with 
# necessary mandatory fields.
def read_from_file(file_name):
    with open(file_name) as f:
        cascade_object = load(f)
    
    logging.info(f'File: {file_name} ...')
    logging.info('Reading nodes ...')
    nodes = pd.DataFrame(cascade_object['nodes'])
    logging.info('Reading edges ...')
    edges = pd.DataFrame(cascade_object['edges'])
    logging.info('Reading cascade features ...')
    cascade_features = cascade_object['cascade_features']

    return Cascade(nodes=nodes, edges=edges, cascade_features=cascade_features)
     
# Convert temporal edge list (parent,node,time) to 
# (parent,parent_inf_time,node,inf_time)
def source_target_time_to_edges(edges, nodes):
    ### get sorted list of time steps
    ticks = edges.time.drop_duplicates().sort_values().tolist()

    node_list = nodes['node'].tolist()
    node_tick = pd.Series(index=nodes['node'], data=np.ones(nodes.shape[0])*-1)
    node_tick.loc[-1] = -1

    for tick in ticks:
        curr_ind = edges.time==tick

        # assign tick to parent
        edges.loc[curr_ind, 'parent_inf_time'] = \
                edges[curr_ind].parent.map(node_tick)

        # update current tick for all node nodes
        target_tick_map = edges[curr_ind][
                ['node', 'time']].set_index('node').squeeze()

        node_tick.loc[edges[curr_ind].node] = target_tick_map
    
    if (edges[edges.parent!=-1].parent_inf_time==-1).sum():
        raise ValueError('Found some nodes with time of infection=-1.')

    edges.parent_inf_time = edges.parent_inf_time.astype(int)
    edges = edges.rename(columns={'time': 'inf_time'})
    
    return edges

# Obtain boundary nodes
def extract_boundary_nodes(nodes, network):
    nodes_map = pd.Series(index=nodes.node.values, 
            data=np.ones(nodes.shape[0], dtype=int)*-1)
    u_boundary = network[network.u.map(nodes_map).isna()].u.value_counts()
    v_boundary = network[network.v.map(nodes_map).isna()].v.value_counts()
    boundary = pd.concat([u_boundary, v_boundary]).reset_index().rename(
            columns={'index': 'node', 0: 'out_degree'})
    return boundary

# Sample a cascade
# AA: currently tested for only SEIR
def sample_cascade(cascade, coverage=1, seed=0):
    if coverage == 1:
        logging.warning('Since coverage is 1, no sampling done.')
        return cascade
    logging.info(f'Sampling with coverage={coverage} and seed={seed} ...')
    
    nodes = cascade.nodes.sample(frac=coverage, random_state=seed)
    edges = cascade.edges

    node_map = pd.Series(index=nodes.node, 
            data=np.ones(nodes.shape[0], dtype=int))
    node_map[-1] = 1    # We want to keep seeding events for now
    
    # Creating the induced graph
    node_ind = ~edges.node.map(node_map).isna()
    not_parent_ind = edges.parent.map(node_map).isna()

    # Handling orphan nodes
    # If parent not observed, make it -1
    edges.loc[not_parent_ind, ['parent', 'parent_inf_time']] = -1
    # At this point, every parent should be accounted for
    edges = edges[node_ind]

    cascade.nodes = nodes
    cascade.edges = edges
    return
