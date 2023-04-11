# Code for "Identifying Complex Disease Scenarios from Cascade Data"
Contact: Abhijin Adiga (abhijin@virginia.edu)

The experiments were performed using a high-performance computing cluster in Linux environment (https://www.rc.virginia.edu/userinfo/rivanna/overview/). Therefore, the user will encounter some SLURM commands. These can be easily replaced by normal Unix shell statements.

See README.md in each folder for more information.

## Example network
Folder: ``./example_network``

## Generation of network properties for simulation outputs
Folder: ``./simanalytics``

## Feature extraction
Folder: ``./feature_extraction``

## Learning framework for scenario identification
Folder: ``./scenario_identification``

Pending
## Results
The results of network measurements and simulations are in ``results.db``. It has two tables.

* ``mpsn_props`` has network measures for all the generated synthetic networks.
* ``summary`` has simulation results.
