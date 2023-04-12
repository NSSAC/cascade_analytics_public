# Code for "Identifying Complex Disease Scenarios from Cascade Data"
Contact: Abhijin Adiga (abhijin@virginia.edu)

The experiments were performed using a high-performance computing cluster in Linux environment (https://www.rc.virginia.edu/userinfo/rivanna/overview/). Therefore, the user will encounter some SLURM commands. These can be easily replaced by normal Unix shell statements.

The repository is organized in to folders. See README.md in each folder for
more information.

* Example network: ``./example_network``
* Generation of network properties for simulation outputs: ``./simanalytics``
* Wrapper scripts for generating properties for this study: ``./KDD23``
* Feature extraction: ``./feature_extraction``
* Learning framework for scenario identification: ``./scenario_identification``

Pending
## Results
The results of network measurements and simulations are in ``results.db``. It has two tables.

* ``mpsn_props`` has network measures for all the generated synthetic networks.
* ``summary`` has simulation results.

## Directory organization
``scripts/`` contains all code.

``work/`` is where all scripts are executed. No content in this folder is
committed to the repository.

``tests/`` contains tests.

``examples/`` contains example inputs.

## Mode of operation
1. ``cd work``
1. ``. .env.sh`` to setup paths. Now every executable in ``scripts/`` is
   available for execution.

<mark>Do not run scripts in ``scripts/`` folder.</mark>

## Scripts
* ``master_usage.sh`` contains usage.
* ``display_tt.py`` displays dendrograms

## Tests 
pending

# Data
Locations data in ``../data/va_activity_locations.csv`` from
``/project/biocomplexity/us_population_pipeline/population_data/usa_840/2017/ver_1_9_0/va/locations``. Made some modifications
* flipped lat,lon
* added name column

