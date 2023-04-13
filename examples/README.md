# Examples
## Network and cascades
This folder contains an example network in ``network/`` and corresponding
simulation outputs in ``cascades/``.

The files are used by ``../simanalytics/master_usage.sh``.

## Network measures and feature sets
This folder contains aggregated network properties for the VA network at time T=70 and using a coverage of 1. The files are in ``aggregated_properties/``. These files are used by ``../feature_extraction/master_usage.sh``.

Also, it contains a table of data points for the ML module generated using the aggregated network property files mentioned above. These files are in ``ml_table/`` and can be regenrated using ``../feature_extraction/master_usage.sh``.
