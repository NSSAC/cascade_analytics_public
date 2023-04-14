import os
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import json
from utils import read_cells, add_scenarios_to_table, determine_scenarios, generate_epicurve_figure, generate_labeled_path_counts, aggregate_node_count_total, plot_node_count_total, generate_multi_feature_figure, aggregate_unlabeled_path, plot_unlabeled_path, plot_out_degree, aggregate_outdegree, aggregate_boundary_degree, plot_boundary_degree
DESC = """Read cell files and produce figures describing some statistics"""

def argument_parser(desc=''):
    parser=argparse.ArgumentParser(description=desc, 
            formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('-c', '--cell-files', 
            nargs='*', 
            required=True,
            help='Cell parque/JSON files')

    parser.add_argument("--input-format",
            choices=['json', 'parquet'],
            help="whether input is a JSON file or a parquet file",
            default='parquet')

    parser.add_argument("--scenario-label-color",
            help="A JSON object containing mapping from scenario name to label and color",
            type=json.loads, required=True)

    parser.add_argument('--scenario-order', 
            nargs='*', 
            required=True,
            help='Order of scenarios in figures')
    # Output
    parser.add_argument("-o", "--output-folder",
            help="Folder to store resulting figures",
            default='out')
    
    return parser

TAB_COLORS=['tab:red', 'tab:orange', 'tab:green', 'tab:blue', 'tab:purple', 'tab:pink', 'tab:gray']

def main(args):
    output_dir = args.output_folder
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    all_cells = read_cells(args.input_format, args.cell_files, 8) 
    scenarios = determine_scenarios(all_cells)
    scenario_map = {s:args.scenario_label_color[s]['label'].replace('&','\n') for s in scenarios}
    color_map = {
        args.scenario_label_color[s]['label'].replace('&','\n'):args.scenario_label_color[s]['color'] 
        for s in scenarios}
    scenario_order = [x.replace('&','\n') for x in args.scenario_order]
    color_order = [color_map[x] for x in scenario_order]
    add_scenarios_to_table(all_cells, 'labeled_paths')
    add_scenarios_to_table(all_cells, 'en_labeled_paths')
    add_scenarios_to_table(all_cells, 'boundary_outdegree')
    add_scenarios_to_table(all_cells, 'out_degree')
    add_scenarios_to_table(all_cells, 'number_of_nodes')
    add_scenarios_to_table(all_cells, 's_en_labeled_paths')

    #merged_labeled_paths = merge_all_tables_of_type(all_cells, 'labeled_paths')
    label_tables = [(name, cell['labeled_paths']) for name, cell in all_cells.items()]
    s_en_label_tables = [(name, cell['s_en_labeled_paths']) for name, cell in all_cells.items()]
    boundary_degree_tables = [(name, cell['boundary_outdegree']) for name, cell in all_cells.items()]
    node_count_tables = [(name, cell['number_of_nodes']) for name, cell in all_cells.items()]
    outdegree_tables  = [(name, cell['out_degree']) for name, cell in all_cells.items()]
    generate_epicurve_figure(node_count_tables, scenario_map, scenario_order, color_order, 'Epicurve', output_dir)
    generate_labeled_path_counts(s_en_label_tables, scenario_map, scenario_order, output_dir)


    grouped_node_counts_total = aggregate_node_count_total(node_count_tables, scenario_map, scenario_order)
    merged_path_table_grouped = aggregate_unlabeled_path(label_tables, scenario_map, scenario_order)
    grouped_merged_outdegree = aggregate_outdegree(outdegree_tables, scenario_map, scenario_order)
    merged_normalized_boundary_degree, boundary_col_name, normal_boundary_col_name = aggregate_boundary_degree(boundary_degree_tables, scenario_map, scenario_order)
    dataframes = {
        'node_count_total':{
            0: grouped_node_counts_total,
            'function': plot_node_count_total,
            'name':'Total infected nodes',
            'params':{
                'color_order':color_order,
                'scenario_order': scenario_order
            }
        },
        'unlabeled_path_count':{
            0: merged_path_table_grouped,
            'function': plot_unlabeled_path,
            'name':'Path frequency',
            'params':{
                'color_order':color_order,
                'scenario_order': scenario_order,
            }
        },
        'out_degree':{
            0:grouped_merged_outdegree,
            'function': plot_out_degree,
            'name': 'Out degree',
            'params':{
                'color_order':color_order,
                'scenario_order': scenario_order,
                'min': 4,
                'max': 6
            }
        },
        'boundary_degree':{
            0: merged_normalized_boundary_degree ,
            'function': plot_boundary_degree,
            'name':'In degree of boundary nodes',
            'params':{
                'color_order':color_order,
                'scenario_order': scenario_order,
                'col_name': boundary_col_name,
                'normal_col_name':normal_boundary_col_name,
                'max':3
            }
    },
    }
    features = [key for key in dataframes.keys()]#, 'level_count']#, 'boundary_degree']#,'level_count', 'labeled_path',]
    generate_multi_feature_figure(dataframes, features, output_dir)



if __name__=='__main__':
    args = argument_parser(DESC).parse_args()
    main(args)