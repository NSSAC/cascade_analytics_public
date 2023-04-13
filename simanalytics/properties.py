DESC = '''
Computing properties of cascades.

AA
'''

import argparse
import dask.dataframe as dd
import itertools
import json
import logging
import numpy as np
import pandas as pd
from pdb import set_trace
# import snap
from time import time

import cascade as ca

PROPERTIES=[ \
        'number_of_distinct_nodes',
        'number_of_nodes',
        'number_of_edges',
        'out_degree',
        'labeled_paths',
        'boundary_outdegree'
        ]
# AA: probably this should be study-specific also. But, some code will break if 
# some sets are empty.
MANDATORY_GROUPBY_FEATURES = {
        'number_of_nodes': {'inf_time'},
        'number_of_edges': {'inf_time'},
        'out_degree': {'parent_level', 'parent_inf_time'}
        }
DIAMETER_NUM_STARTING_NODES=100
FORMAT="%(levelname)s:%(filename)s:%(funcName)s:%(message)s"
TOP_TICK_LOCN = 50

# This circus is to avoid strings/objects in huge dataframes
# ASSUMPTION: There are not more than these many ticks.
NODE_TIME_OFFSET = 1000000
# ASSUMPTION: There are not more than these many labels.
LABEL_OFFSET = 10000

class Time:
    def __init__(self):
        self.current_time = time()
        self.total_time = 0

    def update_time(self):
        current_period = time() - self.current_time
        self.total_time += current_period
        return current_period, self.total_time

def prepare_features_list(features_for_property, 
        node_features, edge_features):
    properties = features_for_property.keys()
    for property in properties:
        if property not in PROPERTIES:
            logging.error(f'"{property}" not in supported properties: {PROPERTIES}.')
            raise ValueError

    # Checking if all features specified are present in either node or edge 
    # feature set.
    features = set()
    for fe in features_for_property.values():
        features = features.union(set(fe))

    # AA: not sure if this is clean enough
    try:
        features.remove('level')
    except KeyError:
        pass

    non_fe = features.difference(set(node_features).union(
        set(edge_features)))

    if non_fe:
        raise ValueError(f'The following feature(s) {non_fe} are not present in either node or edge feature list.')

    # assigning empty lists for unspecified properties
    for prop in PROPERTIES:
        if not prop in properties:
            features_for_property[prop] = []
    return


# extract levels in BFS with seeds as roots
def bfs_levels(nodes, edges, ticks):
    logging.info(f'Computing BFS levels for each (node,time) pair ...')

    ### initiate levels
    ### Assumption is that at tick=1, contact_pid=-1
    edges['level'] = np.nan 
    edges['parent_level'] = np.nan 

    node_level = pd.Series(index=nodes['node'], data=np.ones(nodes.shape[0])*-1)
    node_level.loc[-1] = -1 

    for tick in ticks:
        curr_ind = edges.inf_time==tick

        # assign level to parents
        edges.loc[curr_ind, 'parent_level'] = \
                edges[curr_ind].parent.map(node_level)

        # update current level for all nodes: level(parent)+1
        level_map = edges[curr_ind]['parent_level'] + 1

        # update level for current nodes
        edges.loc[curr_ind, 'level'] = level_map
        node_level.loc[edges[curr_ind].node] = level_map.values

    edges.loc[edges.parent==-1, 'parent_level'] = -1
    if edges.isnull().values.any():
        raise ValueError('Found some unassigned levels.')

    edges.level = edges.level.astype(int)
    edges.parent_level = edges.parent_level.astype(int)
    
    return

# Convert integer coded labeled paths to string
def convert_coded_paths_to_string(code_, inverse_feature_map=None,
        feature_type=None):
    label = ''
    code = code_
    while code:
        ## if inverse_feature_map is None:
        ##     code_label = str(code % LABEL_OFFSET)
        ## else:
        # AA: Why does rem give a float number
        code_label = str(inverse_feature_map[int(code % LABEL_OFFSET)])
        label = f'-{code_label}{label}'
        code //= LABEL_OFFSET
    if feature_type == 'node':
        return label[1:]
    elif feature_type == 'edge':
        rem_len = len(code_label) + 2   # remove first label + 2 for two -
        return label[rem_len:]
    else:
        raise ValueError(f'Unsupported feature type "{feature_type}".')

# For now, it only supports aggregating by a single feature.
def labeled_paths(orig_edges, feature, feature_type, k):
    labeled_paths_list = []

    # AA: This should be done beforehand, not here.
    if feature is not None:
        #inverse_feature_map = None
        inverse_feature_map = orig_edges[feature
                ].drop_duplicates().reset_index(drop=True)
        inverse_feature_map.index += 1 # this is to take care of remainder operation
        feature_map = pd.Series(inverse_feature_map.index.values, 
                index=inverse_feature_map)
        orig_edges[feature] = orig_edges[feature].map(feature_map)

        # Create node-tick-feature map
        node_tick_feature = pd.Series(
                index=orig_edges.node*NODE_TIME_OFFSET+orig_edges.inf_time,
                data=orig_edges[feature].values)

    # Create an edge data frame with just enough information
    ### node-tick pairs and time
    edges = pd.DataFrame(columns=['st', 'tt', 'start_time', 'end_time'])
    _orig_edges = orig_edges[orig_edges.parent!=-1]
    edges.st = _orig_edges.parent * NODE_TIME_OFFSET + _orig_edges.parent_inf_time
    edges.tt = _orig_edges.node * NODE_TIME_OFFSET + _orig_edges.inf_time
    edges.start_time = _orig_edges.parent_inf_time
    edges.end_time = _orig_edges.inf_time

    paths = pd.DataFrame()
    for i in range(1,k+1):
        if i == 1:
            paths['t'] = edges.start_time
            paths['endt'] = edges.end_time
            paths[0] = edges.st
            paths[1] = edges.tt
        else:
            paths = paths.drop('endt', axis=1).merge(
                    edges, left_on=i-1, right_on='st', how='left'
                    ).drop('st', axis=1).rename(columns={'tt': i})

            # leftmost tick
            paths = paths[~paths[i].isnull()].drop('start_time', axis=1).rename(
                    columns={'end_time': 'endt'}).astype(int)
        if feature is not None:
            labeled_paths = pd.DataFrame({
                'start_time': paths.t.tolist(), 
                'path': paths[0].map(node_tick_feature),
                'time_diff': (paths.endt-paths.t).tolist()},
                index=paths.index)
            for j in range(1,i+1):
                # AA: This is a debug point to catch parent issues. NaNs can be
                # detected here.
                #try:
                labeled_paths['path'] = labeled_paths['path']*LABEL_OFFSET + \
                        paths[j].map(node_tick_feature)
                #except:
                #    set_trace()
        else:
            labeled_paths = pd.DataFrame({'start_time': paths.t.tolist(), 
                    'time_diff': (paths.endt-paths.t).tolist()},
                    index=paths.index)
        
        path_counts = labeled_paths.drop('time_diff', axis=1
                ).value_counts().reset_index().rename(columns={0: 'value'})
        if feature is not None:
            path_counts['path'] = path_counts.path.apply(
                    convert_coded_paths_to_string,
                    inverse_feature_map=inverse_feature_map,
                    feature_type=feature_type)
        path_counts['path_len'] = i

        labeled_paths_list.append(path_counts)
    return pd.concat(labeled_paths_list)

# Computing cascade properties
### features_for_property specifies a list of features for each property. Every
### property will be computed for each distinct tuple of values corresponding to 
### the features list. 
def compute_cascade_properties(cascade, property_attributes, return_dict):
    
    features_for_property = property_attributes['features_for_property']
    max_num_hops = property_attributes['max_num_hops']
    labeled_path_features = property_attributes['labeled_path_features']

    props = {}
    current_time = time()
    total_time = 0
    cid = cascade.cascade_features['ID']
    logging.info(f'Computing properties for cascade {cid} ...')

    # Prep work
    nodes = cascade.nodes
    edges = cascade.edges
    ticks = edges.inf_time.drop_duplicates().sort_values().tolist()

    ### verify features list for each property
    prepare_features_list(features_for_property, 
            nodes.columns.tolist(), 
            edges.columns.tolist())

    ### Find levels for each (node,time) pair
    bfs_levels(nodes, edges, ticks)

    ## ### Map nodes to features using the node features data
    ## if nodes.columns.tolist() != ['name']:   # checking for features other than 'name'
    ##     nodes = nodes.set_index('name', drop=True)
    ##     edges = pd.merge(edges, nodes, 
    ##             left_on='target', right_index=True, how='left')

    # First we do all pandas operations
    logging.info(f'Basic properties ...')
    nodes = nodes.set_index('node') # sometimes, only the name column is present
    props['ID'] = cid

    ## update_time('preliminaries', 
    ## props['time']['preliminaries'] = time() - current_time
    ## current_time = time()

    # Node counts
    # Some counts depend on nodes alone, while others depend on node-tick pairs.
    # AA: In the future, this should come from the cascade. Right now, edges has 
    # all the node-tick information required.
    if not nodes.empty:
        edges = edges.merge(nodes, 
                left_on='node', right_index=True)

    ### Number of distinct nodes
    props['number_of_distinct_nodes'] = pd.Series(name='value', 
            data=nodes.shape[0]).to_frame()

    # Number of nodes by specified node features
    groupby_features = list(
            set(features_for_property['number_of_nodes']).union(
                MANDATORY_GROUPBY_FEATURES['number_of_nodes']))
    if groupby_features:
        # AA: check if these are unique counts
        props['number_of_nodes'] = edges.groupby(
                groupby_features).size().reset_index().rename(columns={0: 'value'})

    # Edges: Note that the validity of the edge feature has to be checked by the
    # user.
    edges_minus_seeds = edges[edges.parent!=-1]
    groupby_features = list(
            set(features_for_property['number_of_edges']).union(
                MANDATORY_GROUPBY_FEATURES['number_of_nodes']))
    if groupby_features:
        props['number_of_edges'] = edges_minus_seeds.groupby(
                groupby_features).size().reset_index().rename(columns={0: 'value'})
    else:
        props['number_of_edges'] = pd.Series(name='value', 
                data=edges_minus_seeds.shape[0]).to_frame()

    # Out-degree by time and level
    groupby_features = list(
            set(features_for_property['out_degree']).union(
                MANDATORY_GROUPBY_FEATURES['out_degree']).union({'parent'}))
    groupby_features = list(groupby_features)
    out_degree = edges_minus_seeds.groupby(groupby_features
            ).size().reset_index().rename(columns={0: 'out_degree'})

    groupby_features.append('out_degree')
    groupby_features.remove('parent')
    props['out_degree'] = out_degree.groupby(list(groupby_features)
            ).size().reset_index().rename(columns={0: 'value'})
    del out_degree
    
    # labeled paths
    logging.info(f'Labeled paths ...')
    if labeled_path_features:
        for feature, feature_type in labeled_path_features.items():
            props['labeled_paths'] = labeled_paths(edges, feature, feature_type,
                    max_num_hops)
    else:
        props['labeled_paths'] = labeled_paths(edges, None, None, max_num_hops)

    # boundary nodes
    if cascade.boundary is not None:
        props['boundary_outdegree'] = \
                cascade.boundary.groupby('node').sum().value_counts().sort_index(
                        ).reset_index().rename(columns={
                            'out_degree': 'boundary_outdegree',
                            0: 'value'})

    logging.info(f'Done')
    return_dict[cid] = props

    return

def aggregate_properties(cascade_properties, property_names):

    aggp = {}
    for p in list(cascade_properties)[0].keys():
        if p not in property_names:
            continue
        logging.info(f'Aggregating "{p}" ...')
        plist = []
        for c in cascade_properties:
            c[p]['cascade'] = c['ID']
            plist.append(c[p])
        df = pd.concat(plist)
        ## groupby_columns = df.columns.tolist()
        ## groupby_columns.remove('value')
        ## if groupby_columns:
        ##     adf = df.groupby(groupby_columns).describe().fillna(0)
        ##     adf.columns = [y for x,y in adf.columns.to_flat_index().tolist()]
        ##     adf = adf.reset_index()
        ## else:
        ##     adf = df.describe().transpose()
            
        ## aggp[p] = adf.to_dict(orient='list')
        aggp[p] = df #df.to_dict(orient='list')
    return aggp

def main():
    # parser
    parser=argparse.ArgumentParser(description=DESC, 
            formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument("-c", "--cascade_file_name", 
            required=True,
            help="JSON cascade file.")

    parser.add_argument("-a", "--features_for_property", 
            default={},
            type=json.loads,
            help="Dictionary in JSON format: { 'property': ['attibute1', 'feature2']. Every property will be computed for each distinct tuple of values corresponding to the features list.")

    parser.add_argument("-o", "--output_file_name", 
            default='out.json',
            help="Properties in JSON format")

    # Run-related
    parser.add_argument("-d", "--debug", action="store_true")
    parser.add_argument("-q", "--quiet", action="store_true")

    args = parser.parse_args()

    main_start = time()

    # Set logger
    if args.debug:
       logging.basicConfig(level=logging.DEBUG,format=FORMAT)
    elif args.quiet:
       logging.basicConfig(level=logging.WARNING,format=FORMAT)
    else:
       logging.basicConfig(level=logging.INFO,format=FORMAT)

    # Load cascade
    cg = ca.read_from_file(args.cascade_file_name)

    # Compute properties
    props = compute_cascade_properties(cg, args.features_for_property)

    # Write to file
    with open(args.output_file_name,'w') as f:
        json.dump(props, f, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    main()
