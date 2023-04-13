DESC = """
This program is a front-end for computing measures for a set of cascades.
It is assumed that these cascades are different stochastic realizations of
the same simulation model, i.e., the model input parameters are identical
for these cascades. Currently, it is not clear whether this needs to be
study-specific. At the least, it can serve as a template for study-specific
scripts.

Three types of code segment can be found here:
    1. Simulator-specific
    2. Study-specific
    3. General

1 and 2 correspond to generating cascades in a standard form, while 3 corresponds
to computing properties of cascades in general form.

By AA
"""

import argparse
import glob
import json
import logging
from multiprocessing import Process, Manager, cpu_count
import numpy as np
import os
import pandas as pd
from pdb import set_trace
from shutil import rmtree
from sys import exit
from time import time
from zipfile import ZipFile, ZIP_DEFLATED

import cascade as ca
import properties as prop
import simulator as sim
import study as st

FORMAT="%(levelname)s:%(filename)s:%(funcName)s:%(message)s"

# Basic parser
def argument_parser(desc=''):
    parser=argparse.ArgumentParser(description=desc, 
            formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('--study', 
            required=True,
            choices=st.SUPPORTED_STUDIES,
            help='Study, which provides the context.')

    parser.add_argument('-s', '--simulation_outcomes', 
            nargs='*', 
            required=True,
            help='Simulation instances corresponding with the same cascade features.')
    
    # Read cascade features
    parser.add_argument('-c', '--cascade_features', 
            required=True,
            type=json.loads,
            help='Every cascade will be associated with features and their corresponding values. Dictionary in JSON format: dict(parameter=value, ...). At minimum, required experiment cell features and their values should be specified here.')

    # Attributes for properties
    parser.add_argument("-a", "--features_for_property", 
            type=json.loads,
            default={},
            help="Dictionary in JSON format: { 'property': ['feature1', 'feature2'] ... }. Every measure will be computed for each distinct tuple of values corresponding to the features list.")

    parser.add_argument('--study_inputs',
            type=json.loads,
            default={},
            help='Dictionay in JSON format: { "feature": "value", ...}')

    # Output
    parser.add_argument("-o", "--properties_file_prefix",
            help="The json/zip file that contains network properties",
            default='out')
    parser.add_argument("--output_format",
            choices=['json', 'parquet'],
            help="Contains network properties",
            default='json')

    # Run-related
    ## parser.add_argument("--processes_per_round", type=int,
    ##         default=10,
    ##         help='This is for memory management. Hopefully, next version will not have it.')
    parser.add_argument("--no_parallel", action="store_true",
            help='Replaces all multiprocessing code with serial code.')
    parser.add_argument("-d", "--debug", action="store_true")
    parser.add_argument("-q", "--quiet", action="store_true")
    parser.add_argument("--no_time", action="store_true", help="Do not display time taken")
    ## parser.add_argument("--num_processes", 
    ##         type=int, 
    ##         default=NUM_PROCESSES, 
    ##         help="Number of processes")
    
    return parser

def spawn_jobs(processes_per_round=1000, func=None, args=None, no_parallel=False):
    round = 1
    processes_to_track = []
    if not no_parallel:
        with Manager() as manager:
            return_dict = manager.dict()
            for pnum, arr in enumerate(args):
                arr.append(return_dict)
                p = Process(target=func, args=arr)
                p.start()
                processes_to_track.append(p)

                # Pause submission till the current batch is done
                if not (pnum+1) % processes_per_round:
                    logging.info(f'Round {round}: total submitted {pnum+1} ...')
                    round += 1
                    for p in processes_to_track:
                        p.join()
                    processes_to_track = []
            for p in processes_to_track:
                p.join()
            result = return_dict.values()
    else:
        logging.warning('SERIAL CODE ON!!')
        return_dict = dict()
        for arr in args:
            func(*arr, return_dict)
        result = return_dict.values()
    return result

def get_procs_per_round(file_list):
    # File size
    s = 0
    for f in file_list:
        s += os.path.getsize(f)
    s /= 10**9
    logging.info(f'Total size: {s}GB')

    # A linear function based on (12GB,5 processes) and (4GB, 15 processes)
    return 5 #int(s * -4.6 / 4 + 20) #-5/4

def write_to_file(prop, file_prefix, mode='json'):
    logging.info(f'Dumping to {file_prefix}.{mode}.zip ...')
    if mode == 'json':
        _prop = dict()
        for key in prop.keys():
            if isinstance(prop[key], pd.DataFrame):
                _prop[key] = prop[key].to_dict(orient='list')
            elif isinstance(prop[key], dict):
                _prop[key] = prop[key]
            else:
                raise TypeError('Unsupported data type.')
        with ZipFile(f'{file_prefix}.{mode}.zip', 'w', 
                ZIP_DEFLATED) as zf:
            with zf.open(f'{file_prefix}.json', 'w') as json_file:
                data_bytes = json.dumps(
                        _prop, ensure_ascii=False, 
                        allow_nan=False, indent=4).encode('utf-8')
                json_file.write(data_bytes)
    elif mode == 'parquet':
        os.mkdir(file_prefix)
        for key in prop.keys():
            if isinstance(prop[key], pd.DataFrame):
                tab = prop[key]
            elif isinstance(prop[key], dict):
                try:
                    tab = pd.DataFrame(prop[key])
                except ValueError:
                    tab = pd.Series(prop[key]).to_frame().rename(columns={
                        0: 'value'})
            else:
                raise TypeError('Unsupported data type.')
            if (tab.dtypes == object).any():
                logging.warning(
                        f'"{key}" has object data type; converting to string.')
                tab = tab.astype(str)
            tab.to_parquet(f'{file_prefix}/{key}.parquet')
        with ZipFile(f'{file_prefix}.{mode}.zip', 'w', 
                ZIP_DEFLATED) as zf:
            for f in glob.glob(f'{file_prefix}/*parquet'):
                zf.write(f, os.path.basename(f))
        rmtree(file_prefix)

def main():

    start = time()

    # General parser
    parser = argument_parser(DESC)

    # Study-specific stuff comes here. Right now, there is nothing.
    ## AA: Need to figure this out.
    ## AA: May be nothing needs to be done here.

    args = parser.parse_args()

    # Checking if results are already present
    if args.output_format == 'parquet':
        if os.path.exists(args.properties_file_prefix):
            raise FileExistsError(f'Folder {args.properties_file_prefix} already exists.')

    # AA: need a better way to manage time
    main_start = time()

    # Set logger
    if args.debug:
       logging.basicConfig(level=logging.DEBUG,format=FORMAT)
    elif args.quiet:
       logging.basicConfig(level=logging.WARNING,format=FORMAT)
    else:
       logging.basicConfig(level=logging.INFO,format=FORMAT)

    # Get number of rounds
    logging.info(f'Number of available cores: {cpu_count()}')
    rounds = get_procs_per_round(args.simulation_outcomes)
    logging.info(f'Number of processes per round: {rounds}')

    # Initiate study
    logging.info(f'Initiating study ...')
    study_class = getattr(st, args.study)
    study = study_class(**vars(args))

    # Generate cascades
    logging.info(f'Reading simulation outcomes ...')
    logging.info(f'Number of files to process: {len(args.simulation_outcomes)}')
    args_list = [[f,id] for id,f in enumerate(args.simulation_outcomes)]
    cascades = spawn_jobs(processes_per_round=5, 
            func=study.sim_to_cascade,
            args=args_list, no_parallel=args.no_parallel)
    study.delete_attributes()
    try:
        print(study.network)
    except Exception as err:
        print(err)

    # Compute properties
    logging.info(f'Computing properties of cascades ...')
    property_attributes = {
            'features_for_property': study.features_for_property,
            'labeled_path_features': study.labeled_path_features,
            'max_num_hops': study.max_num_hops
            }
    args_list = [[c, property_attributes] for c in cascades]
    properties = spawn_jobs(processes_per_round=5, 
            func=prop.compute_cascade_properties,
            args=args_list, no_parallel=args.no_parallel)

    # Combine
    logging.info('Combining properties of all cascades ...')

    # Some properties share a common methodology for aggregation
    ### Currently, all properties can be aggregated this way.
    out_dict = prop.aggregate_properties(properties, prop.PROPERTIES)
    out_dict['cascade_features'] = study.cascade_features
    out_dict['cascade_features']['compute_time'] = int((time() - start)//60)

    # Dumping to file
    write_to_file(out_dict, args.properties_file_prefix, mode=args.output_format)

    logging.info(f'Time: {out_dict["cascade_features"]["compute_time"]} minutes')
    logging.info(f'Done')

if __name__=='__main__':
    main()
