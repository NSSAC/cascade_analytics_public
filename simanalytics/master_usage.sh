#!/bin/bash
# For more details, see "python simprop.py -h"
python ../simanalytics/simprop.py \
    --no_parallel \
    --output_format parquet \
    --study_inputs "{\"edge_features\":
    \"../examples/network/edge_activity_map.feather\", \"network\": \"../examples/network/unique_edges.feather\", \"coverage\": 1, \"random_seed\": 0, \"time_horizon\": 70}" \
    --study KDD \
    -s ../examples/cascades/TestEpiHiper_1.csv ../examples/cascades/TestEpiHiper_2.csv \
    -c '{"scenario": "cell_1", "network": "example"}' \
    --features_for_property '{"number_of_nodes": ["inf_time"], "number_of_edges": ["inf_time"], "labeled_paths": ["inf_time"], "out_degree": ["inf_time"]}' \
    -o out

