import pdb
import contextlib
from tempfile import mkdtemp 
import argparse
import json
from zipfile import ZipFile
from typing import List, Dict, Tuple
import tempfile
import pandas as pd
import os.path
from collections import defaultdict
from multiprocessing import Process, Manager, Pool
from matplotlib import pyplot as plt
import seaborn as sns

"""
Change the data type of `col` in `table` to be int if it is changeable (all values are digits)
"""
def fix_table_int(table, col):
    if col in table.columns:
        if all(table[col].astype(str).str.isdigit()):
            table[col] = table[col].astype(int)
        return table
    else:
        return table
"""
Change the data type of `col` in `table` to be in `typ`. However, does not do
safety check to confirm data is convertable.
"""
def fix_table_type(table, col, typ):
    if col in table.columns:
        table[col] = table[col].astype(typ)
        return table
    else:
        return table

def replace_multiple(orig, chars_to_remove, char_to_replace_with):
    for char in chars_to_remove:
        orig = orig.replace(char, char_to_replace_with)
    return orig

def create_essential_nonessential_paths(table: pd.DataFrame):
    ESSENTIAL=['1','2','5','6']
    NONESSENTIAL=['3','4','7']

    cols = table.columns.to_list()
    cols.remove('path')
    cols.append('en_path')
    cols_no_value = [x for x in cols]
    cols_no_value.remove('value')
    new_table = table
    new_table['en_path'] = new_table['path'].apply(replace_multiple, args=(ESSENTIAL, 'e'))
    new_table['en_path'] = new_table['en_path'].apply(replace_multiple, args=(NONESSENTIAL, 'n'))
    new_table = new_table[cols].groupby(cols_no_value, group_keys=True).sum().reset_index()
    return new_table


def create_simple_essential_nonessential_paths(table: pd.DataFrame):
    ES=['ee']
    HYB=['ne','en']
    NE=['nn']
    cols = table.columns.to_list()
    cols.remove('en_path')
    cols.append('s_en_path')
    cols_no_value = [x for x in cols]
    cols_no_value.remove('value')
    new_table = table
    new_table['s_en_path'] = new_table['en_path'].apply(replace_multiple, args=(ES, 'E'))
    new_table['s_en_path'] = new_table['s_en_path'].apply(replace_multiple, args=(HYB, 'M'))
    new_table['s_en_path'] = new_table['s_en_path'].apply(replace_multiple, args=(NE, 'N'))
    new_table = new_table[cols].groupby(cols_no_value, group_keys=True).sum().reset_index()
    return new_table
def read_a_cell(cell_format, file_name):
    """
    Reads a list of cell zipped archives, each containing a set of parquets.
    Each parquet is a table for a specific measure
    Returns the following:
        (
            cell_name (str),
            {
                table_name (str): pd.DataFrame,
                ...
            }
        )
    """
    tables = {}
    with ZipFile(file_name, 'r') as zipped:
        files_in_zip = zipped.namelist()
        for table in files_in_zip:
            table_name = table.split("."+cell_format)[0]
            with zipped.open(table) as cell:
                if cell_format == 'json':
                    df = pd.read_json(cell)
                elif cell_format == 'parquet':
                    df = pd.read_parquet(cell)
                else:
                    raise ValueError("Cell format is either JSON or parque. You passed "+cell_format)
                fix_table_int(df, 'value')
                fix_table_int(df, 'path_len')
                fix_table_int(df, 'cascade')
                #fix_table_int(df, 'inf_time')
                #fix_table_int(df, 'parent_inf_time')
                fix_table_type(df, 'start_time', float)
                fix_table_type(df, 'time_diff', float)
                tables[table_name] = df
    # create non-essential/essential paths
    if 'labeled_paths' in tables.keys():
        # tables['labeled_paths'] = clear_empty_paths_and_recount(tables['labeled_paths'])
        tables['en_labeled_paths'] = create_essential_nonessential_paths(tables['labeled_paths'])
        tables['s_en_labeled_paths'] = create_simple_essential_nonessential_paths(tables['en_labeled_paths'])
    return os.path.split(file_name)[1], tables

def read_cells(cell_format, files:List, num_workers=1):
    pool = Pool(processes=num_workers)
    cells = {
        name: tables
        for name, tables in pool.starmap(read_a_cell, [(cell_format, f) for f in files])}
    return cells

def aggregate_out_degree(table: pd.DataFrame, bin_size):
    agg = table.groupby(['cascade', 'out_degree'], group_keys=True).value.sum().groupby([lambda x: x[0], lambda x: x[1]//bin_size]).sum().reset_index()
    rows = sorted(table['cascade'].unique())
    cols = agg['level_1'].unique()
    cols_names = ["out_degree_"+str(d*bin_size)+"_to_"+str((1+d)*bin_size) for d in cols]
    col_2_col_name = {a:b for a, b in zip(cols, cols_names)}
    output = pd.DataFrame(0, columns=cols_names, index=rows)
    for _, row in agg.iterrows():
        output.at[row['level_0'], col_2_col_name[row['level_1']]] = row['value']
    return output

def aggregate_labeled_path_count(table: pd.DataFrame, path_col, excluded_path_string, path_lengths):
    if excluded_path_string is not None:
        count_table =table[(table['path_len'].isin(path_lengths)) & ~(table[path_col].str.contains(excluded_path_string))][['cascade', path_col, 'value', 'path_len']].groupby(['cascade', 'path_len', path_col],group_keys=True).sum().reset_index()
    else:
        count_table =table[(table['path_len'].isin(path_lengths))][['cascade', path_col, 'value', 'path_len']].groupby(['cascade', 'path_len', path_col],group_keys=True).sum().reset_index()
    agg = count_table.copy()
    rows = sorted(table['cascade'].unique())
    cols = agg[path_col].unique()
    cols_names = ["cnt_"+path_col+"_"+d for d in cols]
    col_2_col_name = {a:b for a, b in zip(cols, cols_names)}
    output = pd.DataFrame(0, columns=cols_names, index=rows)
    for _, row in agg.iterrows():
        output.at[row['cascade'], col_2_col_name[row[path_col]]] = row['value']
    return output

def aggregate_labeled_path_averages(table: pd.DataFrame, path_col, excluded_path_string, path_lengths):
    total_table =  table[(table['path_len'].isin(path_lengths))][['cascade', path_col, 'value', 'path_len']].copy().groupby(['cascade','path_len']).value.sum().reset_index()    
    if excluded_path_string is not None:
        count_table =table[(table['path_len'].isin(path_lengths)) & ~(table[path_col].str.contains(excluded_path_string))][['cascade', path_col, 'value', 'path_len']].groupby(['cascade', 'path_len', path_col],group_keys=True).sum().reset_index()
    else:
        count_table =table[(table['path_len'].isin(path_lengths))][['cascade', path_col, 'value', 'path_len']].groupby(['cascade', 'path_len', path_col],group_keys=True).sum().reset_index()
    count_col = count_table.columns.get_loc('value')
    length_col = count_table.columns.get_loc('path_len')
    cascade_col = count_table.columns.get_loc('cascade')
    def get_avg(row):
        total = total_table[(total_table['path_len'] == row[length_col]) & (total_table['cascade'] == row[cascade_col])].iloc[0]['value']
        return row[count_col]/total
    agg = count_table.copy()
    agg['value'] = agg.apply(lambda x: get_avg(x), axis=1)
    rows = sorted(table['cascade'].unique())
    cols = agg[path_col].unique()
    cols_names = ["avg_"+path_col+"_"+d for d in cols]
    col_2_col_name = {a:b for a, b in zip(cols, cols_names)}
    output = pd.DataFrame(0, columns=cols_names, index=rows)
    for _, row in agg.iterrows():
        output.at[row['cascade'], col_2_col_name[row[path_col]]] = row['value']
    return output

def aggregate_boundary_out_degree(table: pd.DataFrame, bin_size):
    agg = table.groupby(['cascade', 'boundary_outdegree'], group_keys=True).value.sum().groupby([lambda x: x[0], lambda x: x[1]//bin_size]).sum().reset_index()
    rows = sorted(table['cascade'].unique())
    cols = agg['level_1'].unique()
    cols_names = ["boundary_out_degree_"+str(d*bin_size)+"_to_"+str((1+d)*bin_size) for d in cols]
    col_2_col_name = {a:b for a, b in zip(cols, cols_names)}
    output = pd.DataFrame(0, columns=cols_names, index=rows)
    for _, row in agg.iterrows():
        output.at[row['level_0'], col_2_col_name[row['level_1']]] = row['value']
    return output


def aggregate_path_len(table: pd.DataFrame):
    agg = table.groupby(['cascade', 'path_len'], group_keys=True).value.sum().reset_index()
    rows = sorted(table['cascade'].unique())
    cols = agg['path_len'].unique()
    cols_names = ["path_len_"+str(d) for d in agg['path_len'].unique()]
    col_2_col_name = {a:b for a, b in zip(cols, cols_names)}
    output = pd.DataFrame(0, columns=cols_names, index=rows)
    for _, row in agg.iterrows():
        output.at[row['cascade'], col_2_col_name[row['path_len']]] = row['value']
    return output

def aggregate_epicurve(table: pd.DataFrame, bin_size:int):
    agg = table.groupby(['cascade', 'inf_time'], group_keys=True).value.sum().groupby([lambda x: x[0], lambda x: (x[1])//bin_size], group_keys=True).sum().reset_index()
    rows = sorted(table['cascade'].unique())
    cols = agg['level_1'].unique()
    cols_names = ["infections_"+str(d*bin_size)+"_to_"+str((1+d)*bin_size) for d in cols]
    col_2_col_name = {a:b for a, b in zip(cols, cols_names)}
    output = pd.DataFrame(0, columns=cols_names, index=rows)
    for _, row in agg.iterrows():
        output.at[row['level_0'], col_2_col_name[row['level_1']]] = row['value']
    return output

def merge_feature_tables_one_cell(feat_tables:List[pd.DataFrame]):
    """
    Merges a list of feature tables. The tables must have the same index
    """
    # check that indices match
    for table in feat_tables[1:]:
        if table.index.equals(feat_tables[0].index) != True:
            print(feat_tables[0].index, table.index)
            #raise ValueError("Trying to merge feature tables of differing indices")
            print("Warning: Trying to merge feature tables of differing indices")
    result = pd.concat(feat_tables, axis=1).fillna(0)
    return result

def get_label_from_cascade_features(cascade_feats: pd.DataFrame, label_columns):
    """
    Takes a DataFrame with a cell's aggregated features a list of label columns. Each 
    label column is a feature in the cascade_features table. Will return a list of tuples [(label_name, label_value),]
    Each tuple is a label name and its corresponding value.
    """
    labels = []
    for label_column in label_columns:
        labels.append((label_column, cascade_feats.at[label_column, 'value']))
    return labels

def add_labels(feat_table: pd.DataFrame, labels:List[Tuple]):
    labeled_table = feat_table
    for label_name, label_value in labels:
        labeled_table['label_'+label_name] = label_value
    return labeled_table

def merge_multiple_cell_tables(cell_tables:Dict[str, pd.DataFrame]):
    merged_table = pd.concat(
        [table for _, table in cell_tables.items()],
        axis = 0,
        ignore_index=True,
        join = 'outer',
        ).fillna(0)
    return merged_table

def add_extra_features(table: pd.DataFrame, features:dict):
    for col, value in features.items():
        if col in table.columns:
            raise ValueError("Attempting to overwrite a feature already in DF")
        table[col] = value
    return table

def generate_ml_table(input_format: str, input_files: List[str], extra_features:Dict = None, out_degree_bin_size=List[int], epicurve_bin_size=List[int], boundary_out_degree_bin_size=List[int]):
    """
    Reads a list of aggregated propery files and converts them to an ML-ready
    table.
    """
    cells = read_cells(input_format, input_files, num_workers=8)
    cell_tables = {}
    for cell_name, cell_dict in cells.items():
        feat_tables = []
        for i in epicurve_bin_size:
            feat_tables.append(aggregate_epicurve(cell_dict['number_of_nodes'], i)) 
        for i in out_degree_bin_size:
            feat_tables.append(aggregate_out_degree(cell_dict['out_degree'], i)) 
        for i in boundary_out_degree_bin_size:
            feat_tables.append(aggregate_boundary_out_degree(cell_dict['boundary_outdegree'], i)) 
        feat_tables.append(aggregate_labeled_path_averages(cell_dict['en_labeled_paths'], 'en_path', None, [1])) 
        feat_tables.append(aggregate_labeled_path_averages(cell_dict['en_labeled_paths'], 'en_path', None, [2])) 
        feat_tables.append(aggregate_labeled_path_count(cell_dict['en_labeled_paths'], 'en_path', None, [1])) 
        feat_tables.append(aggregate_labeled_path_count(cell_dict['en_labeled_paths'], 'en_path', None, [2])) 
        feat_tables.append(aggregate_labeled_path_averages(cell_dict['s_en_labeled_paths'], 's_en_path', None, [1])) 
        feat_tables.append(aggregate_labeled_path_averages(cell_dict['s_en_labeled_paths'], 's_en_path', None, [2])) 
        feat_tables.append(aggregate_labeled_path_count(cell_dict['s_en_labeled_paths'], 's_en_path', None, [1])) 
        feat_tables.append(aggregate_labeled_path_count(cell_dict['s_en_labeled_paths'], 's_en_path', None, [2])) 

        feat_tables.append(aggregate_path_len(cell_dict['labeled_paths'])) 
        feat_table = merge_feature_tables_one_cell(feat_tables)
        labels = get_label_from_cascade_features(cell_dict['cascade_features'], ['scenario'])
        labeled_table = add_labels(feat_table, labels)
        cell_tables[cell_name] = labeled_table
    merged_table = merge_multiple_cell_tables(cell_tables)
    if extra_features is not None:
        print("Adding extra features:", extra_features)
        merged_table = add_extra_features(merged_table, extra_features)
    print("ML table contains", merged_table.shape[0], "rows and", merged_table.shape[1], "columns")
    print("Sample:")
    print(merged_table.head())
    # For each cell, create the feature tables, merge them, and label them
    # merge all the labeled tables
    return merged_table

def add_scenarios_to_table(all_cells, table_name):
    for name, cell in all_cells.items():
        cell[table_name]['scenario'] = cell['cascade_features'].loc['scenario'].value
    return all_cells

def merge_diff_cell_tables(tables:List[Tuple[str, pd.DataFrame]]):
    """
    Takes a list of tuples, each a string-DataFrame pair. All the DataFrames
    must be of the same columns. Merges them across rows (concatenates downwards) 
    """
    return pd.concat([table for _, table in tables], axis=0)

def map_and_turn_categorical(table, col, mapping, order):
    table[col] = table[col].map(mapping)
    table[col] = pd.Categorical(table[col],
                                categories=order,
                                ordered=True)
    return table

def determine_scenarios(all_cells):
    cascade_features =[cell['cascade_features'] for cell in all_cells.values()]
    scenarios = [table.loc['scenario'].value for table in cascade_features]
    return list(set(scenarios))

def generate_epicurve_figure(node_count_tables, scenario_map, scenario_order, color_order, title, output_folder, vertical_line_x=None):
    merged_node_counts = merge_diff_cell_tables(node_count_tables)
    grouped = merged_node_counts.groupby(['scenario','cascade','inf_time'], group_keys=True).value.sum().reset_index()

    bin_width=5
    grouped_binned = grouped.groupby(['scenario','cascade','inf_time'], group_keys=True).value.sum().groupby([
        lambda x: x[0],
        lambda x: x[1],
        lambda x: x[2]//bin_width
    ]).sum().reset_index()

    grouped_binned = grouped_binned.rename(columns={'level_0':'scenario', 'level_1':'cascade', 'level_2':'inf_time'})
    grouped_binned['inf_time']=grouped_binned['inf_time']*bin_width
    grouped_binned = map_and_turn_categorical(grouped_binned, 'scenario', scenario_map, scenario_order)
    fig, ax = plt.subplots(1,1)
    fig.set_size_inches(10, 4)
    sns.lineplot(ax=ax, data=grouped_binned, x='inf_time', y='value', hue='scenario', palette=color_order)

    handles, labels = ax.get_legend_handles_labels()
    print(handles)
    fig.legend(handles, labels, loc='center right', fontsize=18, bbox_to_anchor=(0.92,-.18), ncol=4)
    ax.get_legend().remove()
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Number of infections")
    ax.set_facecolor('white')
    ax.set_title(title, fontsize=22)
    ax.xaxis.get_label().set_fontsize(20)
    ax.yaxis.get_label().set_fontsize(20)
    for l in ax.xaxis.get_majorticklabels():
        l.set_fontsize(18)

    for l in ax.yaxis.get_majorticklabels():
        l.set_fontsize(18)
    if vertical_line_x is not None:
        plt.axvline(vertical_line_x, color='red')
    plt.grid()
    plt.savefig(os.path.join(output_folder, 'epicurve.pdf'), bbox_inches='tight')

def get_per_cascade_path_averages(tables, path_col, excluded_path_string, path_lengths, extra_columns=[]):
    total_path_counts = [
      (name,
        #table[(table['path_len'].isin(path_lengths)) & ~(table[path_col].str.contains(excluded_path_string))][['cascade', path_col, 'value','scenario', 'path_len']]
        table[(table['path_len'].isin(path_lengths))][['cascade', path_col, 'value','scenario', 'path_len']]
        .copy()
        .groupby(['cascade','path_len'])
        .value
        .sum()
        .reset_index()
      ) for name, table in tables
    ]
    count_tables = [
        (name,
         table[(table['path_len'].isin(path_lengths)) & ~(table[path_col].str.contains(excluded_path_string))][['cascade', path_col, 'value','scenario', 'path_len']+extra_columns]
         .groupby(
             ['scenario','cascade', 'path_len', path_col]+extra_columns
             ,group_keys=True
         )
         .sum()
         .sort_values(by='value', ascending=False)
         .reset_index()
        )
        for name, table in tables
    ]
    avg_path_tables = []

    for (_, total_table), (name, count_table) in zip(total_path_counts, count_tables):
        count_col = count_table.columns.get_loc('value')
        length_col = count_table.columns.get_loc('path_len')
        cascade_col = count_table.columns.get_loc('cascade')
        def get_avg(row):
            total = total_table[(total_table['path_len'] == row[length_col]) & (total_table['cascade'] == row[cascade_col])].iloc[0]['value']
            return row[count_col]/total
        avg_table = count_table.copy()
        avg_table['value'] = avg_table.apply(lambda x: get_avg(x), axis=1)
        avg_path_tables.append((name, avg_table))
    return avg_path_tables

def generate_labeled_path_counts(s_en_label_tables, scenario_map, scenario_order, output_folder, path_col='s_en_path', path_lengths=[1,2], excluded_path_string='ee', filter_by_time=None):
    def add_end_time(table):
        end_time_col = 'end_time'
        ## Filter based on end time
        start_time_col = table.columns.get_loc('start_time')
        diff_time_col = table.columns.get_loc('time_diff')
        new_table = table.copy()
        new_table[end_time_col] = new_table.apply(
            lambda x: float(x[start_time_col])+float(x[diff_time_col]),
            axis=1)
        return new_table

    tables = s_en_label_tables

    filtered_s_en_label_tables = []

    ####### 
    # Add end time and filter according to it
    #######
    if filter_by_time is not None:
        end_time = filter_by_time
        for name, table in tables:
            new_table = add_end_time(table)
            new_table = new_table[new_table['end_time'] <= end_time]
            filtered_s_en_label_tables.append((name, new_table))
    else:
        for name, table in tables:
            filtered_s_en_label_tables.append((name, table))

    #summed_per_cascade_s_en_label_tables = get_per_cascade_path_counts(tables, path_col, excluded_path_string, path_lengths)
    summed_per_cascade_s_en_label_tables = get_per_cascade_path_averages(filtered_s_en_label_tables, path_col, excluded_path_string, path_lengths)

    merged_summed_s_en_table = merge_diff_cell_tables(summed_per_cascade_s_en_label_tables)

    merged_summed_s_en_table_ordered = merged_summed_s_en_table.sort_values(by='scenario', ascending=True)
    merged_summed_s_en_table_ordered['scenario'] = merged_summed_s_en_table_ordered['scenario'].map(scenario_map)

    merged_summed_s_en_table_ordered['scenario'] = pd.Categorical(merged_summed_s_en_table_ordered['scenario'],
                                                                categories=scenario_order,
                                                                ordered=True)
    #merged_summed_s_en_table_ordered_TN_300 = merged_summed_s_en_table_ordered.copy()
    #merged_summed_s_en_table_ordered_TN_70 = merged_summed_s_en_table_ordered.copy()

    fig, axes = plt.subplots(1,2, gridspec_kw={'width_ratios': [1, 5]})
    fig.set_size_inches(15,5)


    e_ee_plot = merged_summed_s_en_table_ordered[merged_summed_s_en_table_ordered['s_en_path'].isin(['E','E-E'])].copy()
    e_ee_plot['s_en_path'] = pd.Categorical(e_ee_plot['s_en_path'],
                                        categories=['E', 'E-E'],
                                        ordered=True)
    e_ee_plot.sort_values(inplace=True, by=['scenario','s_en_path'])
    rest_plot = merged_summed_s_en_table_ordered[~merged_summed_s_en_table_ordered['s_en_path'].isin(['E','E-E'])].copy()
    rest_plot['s_en_path'] = pd.Categorical(rest_plot['s_en_path'],
                                        categories=['N', 'M', 'E-M', 'E-N', 'M-E', 'N-E', 'N-N', 'N-M', 'M-M'],
                                        ordered=True)
    rest_plot.sort_values(inplace=True, by=['scenario','s_en_path'])
    sns.boxplot(
        ax=axes[0], data=e_ee_plot, x='s_en_path', y='value', hue='scenario', showfliers=False,
    palette=['tab:red','tab:orange','tab:green','tab:blue'])
    sns.boxplot(
        ax=axes[1], data=rest_plot, x='s_en_path', y='value', hue='scenario', showfliers=False,
        palette=['tab:red','tab:orange','tab:green','tab:blue']
        #hue_order=
    )
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=4, fontsize=13)
    for ax in axes:
        ax.get_legend().remove()
        ax.set_xlabel("Labeled path")
        ax.set_ylabel("Ratio")
        ax.set_facecolor('white')
        ax.xaxis.get_label().set_fontsize(18)
        ax.yaxis.get_label().set_fontsize(18)
        for l in ax.xaxis.get_majorticklabels():
            l.set_fontsize(16)
        
        for l in ax.yaxis.get_majorticklabels():
            l.set_fontsize(16)

    #sns.catplot(kind = 'box', data=merged_summed_s_en_table_ordered, x='scenario', y='value', col='s_en_path', col_wrap=4, showfliers=False, sharey=False, legend_out=True, sharex=False, palette=['tab:red','tab:orange','tab:green','tab:blue'], height=4)
    plt.savefig(os.path.join(output_folder, 'labeled_path_counts.pdf'), bbox_inches='tight')

def config_ax(ax):
    ax.title.set_size(22) # 16
    #ax.get_xaxis().get_label().set_visible(False)
    ax.xaxis.get_label().set_fontsize(20) # 14
    ax.yaxis.get_label().set_fontsize(20) # 14
    for l in ax.xaxis.get_majorticklabels():
        l.set_fontsize(18) # 12

    for l in ax.yaxis.get_majorticklabels():
        l.set_fontsize(18) # 12
    ax.yaxis.get_offset_text().set_fontsize(18)
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
       
def make_box_border_same_as_face(ax):
    import matplotlib
    box_patches = [patch for patch in ax.patches if type(patch) == matplotlib.patches.PathPatch]
    if len(box_patches) == 0:  # in matplotlib older than 3.5, the boxes are stored in ax2.artists
        box_patches = ax.artists
    num_patches = len(box_patches)
    lines_per_boxplot = len(ax.lines) // num_patches
    for i, patch in enumerate(box_patches):
        # Set the linecolor on the patch to the facecolor, and set the facecolor to None
        col = patch.get_facecolor()
        patch.set_edgecolor(col)
        #patch.set_facecolor('None')

        # Each box has associated Line2D objects (to make the whiskers, fliers, etc.)
        # Loop over them here, and use the same color as above
        for line in ax.lines[i * lines_per_boxplot: (i + 1) * lines_per_boxplot]:
            line.set_color(col)
            line.set_mfc(col)  # facecolor of fliers
            line.set_mec(col)  # edgecolor of fliers

def aggregate_node_count_total(node_count_tables, scenario_map, scenario_order):
    merged_node_counts_total = merge_diff_cell_tables(node_count_tables)
    merged_node_counts_total = map_and_turn_categorical(merged_node_counts_total, 'scenario', scenario_map, scenario_order)
    grouped_node_counts_total = merged_node_counts_total.groupby(['cascade','scenario']).value.sum().reset_index()
    return grouped_node_counts_total

def plot_node_count_total(ax, data, params):
    #sns.stripplot(ax=ax, data=data, x='scenario', y='value', palette=color_order)
    p = sns.boxplot(ax=ax, data=data, x='scenario', y='value', palette=params['color_order'], showfliers=False)
    make_box_border_same_as_face(p)

    try:
        ax.get_legend().remove()
    except:
        pass
    config_ax(ax)
    ax.set_xlabel('Scenario')
    new_labels = [str(x) for x in params['scenario_order']]
    for i in range(4):
        if i%2 == 0:
            pass
            #new_labels[i] = '\n'+new_labels[i]
    ax.set_xticklabels(new_labels, rotation=30)
    ax.set_ylabel('Occurences')    
        
def aggregate_unlabeled_path(label_tables, scenario_map, scenario_order):
    merged_path_table = merge_diff_cell_tables(label_tables)
    merged_path_table['scenario'] = merged_path_table['scenario'].map(scenario_map)
    merged_path_table['scenario'] = pd.Categorical(merged_path_table['scenario'],
                                                                categories=scenario_order,
                                                                ordered=True)
    merged_path_table_grouped = merged_path_table.groupby(['scenario','cascade','path_len'])['value'].sum().reset_index()
    return merged_path_table_grouped

def plot_unlabeled_path(ax, data, params):
    p = sns.boxplot(ax=ax, data=data, x='path_len', y='value', hue='scenario', palette=params['color_order'], showfliers=False)
    make_box_border_same_as_face(p)
    handles, labels = ax.get_legend_handles_labels()
    ax.set_xlabel('Path length')
    ax.set_ylabel('Occurences')
    ax.get_legend().remove()
    config_ax(ax)

def aggregate_outdegree(outdegree_tables, scenario_map, scenario_order):
    merged_outdegree_table = merge_diff_cell_tables(outdegree_tables)
    grouped_merged_outdegree = merged_outdegree_table.groupby(['scenario','cascade','out_degree']).value.sum().reset_index()
    grouped_merged_outdegree = map_and_turn_categorical(grouped_merged_outdegree, 'scenario', scenario_map, scenario_order)
    return grouped_merged_outdegree

def plot_out_degree(ax, data, params):
    x = data[(data['out_degree'] >=params['min'])&(data['out_degree'] <= params['max'])]
    p = sns.boxplot(ax=ax, data=x, x='out_degree', y='value', hue='scenario', palette=params['color_order'], showfliers=False)
    make_box_border_same_as_face(p)
    #p.set_yscale("log")
    #sns.stripplot(ax=ax, data = data, x='out_degree', y='value', hue='scenario', palette=color_order, dodge=True)
    ax.get_legend().remove()
    config_ax(ax)
    ax.set_xlabel('Out degree')
    ax.set_ylabel('Occurences')

def normalize_boundary_edge_count(tables, bin_size):
    normalized = []
    col_name = 'binned_boundary_degree_bins_'+str(bin_size)
    normal_col_name = 'binned_boundary_degree_bins_'+str(bin_size)+'_normalized'
    for name, table in tables:
        max_boundary_degree = table['boundary_outdegree'].max()
        bin_map = {i: i*bin_size for i in range(0, (max_boundary_degree//bin_size)+1)}
        binned_boundary_degree = table.groupby(['scenario', 'cascade', 'boundary_outdegree']) \
        .value \
        .sum() \
        .groupby([lambda x: x[0], lambda x: x[1], lambda x: x[2]//bin_size]) \
        .sum()
        binned_boundary_degree.index.rename(['scenario','cascade',col_name], inplace=True)
        binned_boundary_degree = binned_boundary_degree.reset_index()
        binned_boundary_degree[col_name] = \
         binned_boundary_degree[col_name].map(bin_map)
        total_bound_nodes = binned_boundary_degree.groupby(['scenario','cascade']).value.sum().reset_index()
        casc_col = binned_boundary_degree.columns.get_loc('cascade')
        scen_col = binned_boundary_degree.columns.get_loc('scenario')
        val_col = binned_boundary_degree.columns.get_loc('value')
        def get_norm(row):
            return row[val_col] / total_bound_nodes[
                (total_bound_nodes['cascade'] == row[casc_col]) &
                (total_bound_nodes['scenario'] == row[scen_col])
            ].iloc[0]['value']
        binned_boundary_degree[normal_col_name] = binned_boundary_degree.apply(
            get_norm,
            axis=1)
        normalized.append((name, binned_boundary_degree))
    return normalized, col_name, normal_col_name

def aggregate_boundary_degree(boundary_degree_tables, scenario_map, scenario_order):
    normalized, col_name, normal_col_name = normalize_boundary_edge_count(
    boundary_degree_tables, 
    #node_count_tables, 
    1)

    normalized[0][1]
    min_b_small = 0
    max_b_small = 5
    min_b_large = 360
    max_b_large = 365
    merged_normalized = merge_diff_cell_tables(normalized)
    merged_normalized['scenario'] = merged_normalized['scenario'].map(scenario_map)
    merged_normalized['scenario'] = pd.Categorical(merged_normalized['scenario'],
                                                                categories=scenario_order,
                                                                ordered=True)
    return merged_normalized, col_name, normal_col_name


def plot_boundary_degree(ax, data, params):
    x = data[data[params['col_name']] < params['max']]
    p = sns.boxplot(ax=ax, data=x, x=params['col_name'], y=params['normal_col_name'], hue='scenario', showfliers=False)
    make_box_border_same_as_face(p)
    config_ax(ax)
    ax.set_xlabel('Boundary degree')
    ax.set_ylabel('Occurences')
    ax.get_legend().remove()



def generate_multi_feature_figure(dataframes, features, output_folder, cols=1):
    #pdb.set_trace()
    rows = len(features)
    fig= plt.figure(constrained_layout='tight')
    fig.set_size_inches((10,20))
    subfigs = fig.subfigures(nrows=rows, ncols=1)
    if rows == 1:
        subfigs = [subfigs]
    col_titles = ['Column '+str(i) for i in range(cols)]
    handles, labels = None, None
    for i, subfig in enumerate(subfigs):
        subfig.suptitle(dataframes[features[i]]['name'], fontsize=25) # 20
        axes = subfig.subplots(nrows=1, ncols=cols)
        if cols == 1:
            axes = [axes]
        #if i == 0:
        for ax, col in zip(axes, col_titles):    
            ax.set_title(col)
        for col in range(cols):
            dataframes[features[i]]['function'](axes[col], dataframes[features[i]][col], dataframes[features[i]]['params'])
            if i == 0:
                handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', fontsize=18, ncol=4, bbox_to_anchor=(0.5,-0.07))
    fig.savefig(os.path.join(output_folder,'multi_feature_figure.pdf'), bbox_inches='tight')
