# TODO: replace with commands created for example network
python props_to_ml_table.py -c \
 /project/biocomplexity/cascade_analytics/2023-KDD/properties/exp5.T30/cell_1_coverage_1_seed_0_T_30.parquet.zip \
 /project/biocomplexity/cascade_analytics/2023-KDD/properties/exp5.T30/cell_2_coverage_1_seed_0_T_30.parquet.zip \
 /project/biocomplexity/cascade_analytics/2023-KDD/properties/exp5.T30/cell_3_coverage_1_seed_0_T_30.parquet.zip \
 /project/biocomplexity/cascade_analytics/2023-KDD/properties/exp5.T30/cell_4_coverage_1_seed_0_T_30.parquet.zip \
 -o exp5.T30.test.csv --input-format parquet --out-degree-bin-size 1 2 --epicurve-bin-size 1 2 \
 --boundary-out-degree-bin-size 2

python props_to_figs.py -c \
 /project/biocomplexity/cascade_analytics/2023-KDD/properties/exp5.T30/cell_1_coverage_1_seed_0_T_30.parquet.zip \
 /project/biocomplexity/cascade_analytics/2023-KDD/properties/exp5.T30/cell_2_coverage_1_seed_0_T_30.parquet.zip \
 /project/biocomplexity/cascade_analytics/2023-KDD/properties/exp5.T30/cell_3_coverage_1_seed_0_T_30.parquet.zip \
 /project/biocomplexity/cascade_analytics/2023-KDD/properties/exp5.T30/cell_4_coverage_1_seed_0_T_30.parquet.zip \
 --output-folder figs --input-format parquet --scenario-label-color \
 '{"cell_1": {"color":"tab:red", "label": "No Vax\\nLow GSD"}, "cell_2": {"color":"tab:orange", "label": "No Vax\\nHigh GSD"},  "cell_3": {"color":"tab:green", "label": "Vax\\nLow GSD"}, "cell_4": {"color":"tab:blue", "label": "Vax\\nHigh GSD"}}' \
 --scenario-order 'No Vax\nLow GSD' 'No Vax\nHigh GSD' 'Vax\nLow GSD' 'Vax\nHigh GSD'
