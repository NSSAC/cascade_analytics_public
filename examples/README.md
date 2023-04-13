# Examples
This folder contains an example network in ``network/`` and corresponding
simulation outputs in ``cascades/``.

The files are used by ``../simanalytics/master_usage.sh``.

Additionally, it contains aggregated network properties for the VA network at time T=70 and using a coverage of 1. The files are in ``aggregated_properties/``. These files are used by ``../feature_extraction/master_usage.sh``.

Also, it contains a machine learning-ready table generated using the aggregated network property files mentioned in the last paragraph. These files are in ``ml_table/`` and can be regenrated using ``../feature_extraction/master_usage.sh``.