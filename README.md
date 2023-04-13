# Code for "Identifying Complex Disease Scenarios from Cascade Data"
Contact: Abhijin Adiga (abhijin@virginia.edu)

The experiments were performed using a high-performance computing cluster in Linux environment (https://www.rc.virginia.edu/userinfo/rivanna/overview/). Therefore, the user might encounter some SLURM commands. These can be easily replaced by normal Unix shell statements.

## Organization
The repository is organized in to folders. See README.md in each folder for
more information. Each folder is discussed in the next section.

## What is provided?
### VA network
The network for VA is open to public but takes up lot of space. We have
included them in a separate link: 
https://net.science/files/8026e893-10f6-4f03-be4b-516ea8c208a6/
TN network is not released to the public.

### Computing network measures

Network measures can be computed for a small example.
* Example network and cascades. See ``./examples`` folder.
* Code that generates cascade properties given simulation outputs and contact network. See ``./simanalytics`` folder.

For the TN and VA networks, an HPC cluster was used.

### Generating machine learning features and aggregated visualizations
Machine learning-ready tables, as well as some network property visualizations, can be calculated for real network measure data.
* Real network measures calculated from the VA network. See ``./examples/aggregated_properties``.
* Code that generates ML-tables and visualizations for aggregated property files, along with script to execute code that utilizes the given network measure files. See ``./feature_extraction``. 

### Galen
Modify as necessary
* Code for the learning part. See ``./scenario_identification``.

## What is not provided?

### Simulator
Simulator code for the network diffusion process is not provided. This work
is under review and will be made public after the review process. It has
been used extensively in previous works. More information is provided in
the following references.

Harrison, G., Chen, J., Mortveit, H., Hoops, S., Porebski, P., Xie, D., Wilson, M., Bhattacharya, P., Vullikanti, A., Xiong, L., Marathe, M. (2023). Synthetic Data To Support US-UK Prize Challenge For Developing Privacy Enhancing Methods: Predicting Individual Infection Risk During A Pandemic  [Data set]. doi:10.18130/V3/ZOG1FF

Bhattacharya, Parantapa, et al. "Data-driven scalable pipeline using national agent-based models for real-time pandemic response and decision support." The International Journal of High Performance Computing Applications 37.1 (2023): 4-27.

Bhattacharya, Parantapa, et al. "AI-Driven Agent-Based Models to Study the Role of Vaccine Acceptance in Controlling COVID-19 Spread in the US." 2021 IEEE International Conference on Big Data (Big Data). IEEE, 2021.

