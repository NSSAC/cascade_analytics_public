# Feature extraction
The scripts in this directory take property files in which properties have been aggregated on the cascade level, and perform the following:

* `prop_to_ml_table.py`: produces machine learning-ready CSV files.
* `props_to_figs.py`: produces figures visualizing these properties.

Both files take as input a list of cell property archives. Each archive contains a set of pandas DataFrame tables, and each of these tables contains a certain measure, aggregated at the cacsade level. To elaborate, every row contains the value of a certain measure aggregated over a single cascade in the cell.

## Installation and setting up environment
Python, matplotlib, pandas, Seaborn

## How to run the example?
The bash script `master_usage.sh` in the same directory contains two sample bash commands. The first will produce an ML table and the second will produce example figures for the aggregated property files in `../examples/aggregated_properties/`. To use this script, simply navigate to the current directory and run the bash script:

```bash
bash master_usage.sh
```

Example figures will be printed to the directory `feature_extraction/figures` and the ML table will be saved as `../examples/ml_table/exp5_T70_features.csv`.


## `prop_to_ml_table.py`
Takes a list of cell files and a set of required properties, and additional aggregation parameters, and produces a single table. Each row in the table corresponds to a single cascade in a particular cell. The columns are either features calculated for the cascade, or classification labels pertaining to the cascade. Label columns have the prefix 'label_' in their names. The only label currently being generated is the name of the scenario used to generate the cascade.

Arguments:

* `-c --cel-files` space-seperated list of cell propert files.
* `-o --output` path to the generated ML-ready table.
* `--out-degree-bin-size` space-seperated list of binning coarseness values for out-degree binning. 
* `--epicurve-bin-size` space-seperated list of binning coarseness values of the epicurve binning. 
* `--boundary-out-degree-bin-size` space-seperated list of binning coarseness values of the boundary out-degree. 
* `--extra-features` A dictionary of extra features to be added to the output data set. 
* `--input-format` the storage formats of the tables (JSON or parquet).

Outputs:
Produces a CSV file containing the ML table. The table contains the following colums:
* Epicurve columns: given the values supplied by the user for the parameter `--out-degree-bin-size`, for each value `c` from these values, the number of nodes infected in the time ranges `[0, c), [c, 2c), ...` are counted. Each of these counts will correspond to a column in the output table named `infections_<s>_to_<e>` with `<s>` and `<e>` being the start and end of the time range.
* Boundary out-degree columns: given the values supplied by the user for the parameter `--boundary-out-degree-bin-size`, for each value `c` from these values, the number of nodes with out-degrees `[0, c), [c, 2c), ...` are counted. Each of these counts will correspond to a column in the output table named `out_degree_<s>_to_<e>` with `<s>` and `<e>` being the start and end of the out-degree range.
* Out-degree count columns: given the values supplied by the user for the parameter `--out-degree-bin-size`, for each value `c` from these values, the number of nodes with out-degrees `[0, c), [c, 2c), ...` are counted. Each of these counts will correspond to a column in the output table named `out_degree_<s>_to_<e>` with `<s>` and `<e>` being the start and end of the out-degree range.
* Label count columns: the number of paths of different lengths. Every length provided in the input cell tables will be counted. Columns are named `path_len_<l>` where `<l>` is the path length.
* Labeld path count columns: For each labeled path of lengths 1 and 2, the occurrences of that path in each cascade are counted. Each column is of the name `cnt_s_en_path_<path>` where `<path>` is a dash-delimeted list of edge labels occurring on the corresponding path. 
* Labeld path percentage columns: For each labeled path of lengths 1 and 2, the percentage of said path with respect to all paths of the same length in a cascade are printed. Each column is of the name `avg_s_en_path_<path>` where `<path>` is a dash-delimeted list of edge labels occurring on the corresponding path. 
* Extra features: For each key-value pair in the `--extra-features` dictionary given as input, a new column is added to the resultant table. The name of the column and its value will be the key and value from the key-value pair, respectively.

## `prop_to_fig.py`
Takes a list of cell files and generates figures visualizing some aggregated measures.

Arguments:

* `-c --cel-files` space-seperated list of cell propert files.
* `-o --output-folder` path to the directory in which figures are to be stored (if pat doesn't exist, it is created).
* `--input-format` the storage formats of the tables (JSON or parquet).
* `--scenario-label-color` a JSON object containing a key for each scenario from which cell files are generated. Each scenario key must have two values, `color` and `label, which are the color used for this scenario in the figures and its label, respectively.
* `--scenario-order` space-seperated list of the scenarios in the order in which scenarios they should be displayed in figures.

Produces three figures that will be stored in the directory specified in the `output-folder` parameter:
1. epicurve.pdf: shows the epicurve of all the scenarios in one figure. Each scenario in the data is represented by a shaded area with a solid line cutting through it. The line represents the mean value for all the cascades of the corresponding scenario, and the shadow is the 95% confidence interval. 
2. labeled_path_counts.pdf: box plots showing the ratios of each labeled path in each scenario with respect to all other paths of the same length. Box plots represent statistics over all cascades of the corresponding scenario.
3. multi_feature_figure.pdf: contains four different box plot figures each showing a different measure. All box plot statistics are with respect to all the cascades in the corresponding scenario. The figures are the following:
   1. Total infected nodes: the total number of infections at the end of the simulation.
   2. Unlabeled path counts: number of paths of lengths 1-4.
   3. Out degree: number of nodes with certain out degrees. In this example, we show nodes with outdegrees 1 and 2.
   4. Boundary degree: number of boundary nodes with certain degrees. In this example, we show nodes with degrees 1 and 2.