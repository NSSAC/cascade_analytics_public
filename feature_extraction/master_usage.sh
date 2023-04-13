# TODO: replace with commands created for example network
python props_to_ml_table.py -c \
 ../examples/aggregated_properties/cell_1_coverage_1_seed_0_T_70.parquet.zip \
 ../examples/aggregated_properties/cell_2_coverage_1_seed_0_T_70.parquet.zip \
 ../examples/aggregated_properties/cell_3_coverage_1_seed_0_T_70.parquet.zip \
 ../examples/aggregated_properties/cell_4_coverage_1_seed_0_T_70.parquet.zip \
 -o exp5_T70_features.csv --input-format parquet --out-degree-bin-size 1 2 --epicurve-bin-size 1 2 \
 --boundary-out-degree-bin-size 2

python props_to_figs.py -c \
 ../examples/aggregated_properties/cell_1_coverage_1_seed_0_T_70.parquet.zip \
 ../examples/aggregated_properties/cell_2_coverage_1_seed_0_T_70.parquet.zip \
 ../examples/aggregated_properties/cell_3_coverage_1_seed_0_T_70.parquet.zip \
 ../examples/aggregated_properties/cell_4_coverage_1_seed_0_T_70.parquet.zip \
 --output-folder figures --input-format parquet --scenario-label-color \
 '{"cell_1": {"color":"tab:red", "label": "No Vax\\nLow GSD"}, "cell_2": {"color":"tab:orange", "label": "No Vax\\nHigh GSD"},  "cell_3": {"color":"tab:green", "label": "Vax\\nLow GSD"}, "cell_4": {"color":"tab:blue", "label": "Vax\\nHigh GSD"}}' \
 --scenario-order 'No Vax\\nLow GSD' 'No Vax\\nHigh GSD' 'Vax\\nLow GSD' 'Vax\\nHigh GSD'
