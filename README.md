# Description

This repository contains the code used in the publication "Temporary Title" that investigates the potential value of ultra-slow fluctuations around 0.05Hz in inferior ventricular regions as a marker of light sleep when fMRI-data is available. This work relies on the publicly available 7T resting-state sample from the human connectome project.

# How to run the analyses

**1. Configure your local version of the Repository**

   a. Provide the correct values to the ```DATA_DIR``` and ```SCRIPTS_DIR``` variables in ```utils/variables.py```.
   
   * ```DATA_DIR```: folder where you want to download the HCP dataset and perform all necessary analyses
   * ```SCRIPTS_DIR```: folder where you downloaded this repository
   
   b. provide your XNAT central user and password in ```utils/variables.py```.
   
   * ```XNAT_USER```: provide your username
   * ```XANT_PASSWORD```: provide your password
   
**2. Create the necessary conda environments**

To run this code you will need two separate conda environments:

   a. ```pyxant_env```: This environment is used by notebook ```N00_DownloadDataFromHCPXNAT```. This notebook downloads all the necessary data from XNAT Central using the pyxnat library. Because this library is not compatible with the latest version of other libraries used in this work, we need to create a separate environment just for this first notebook. A ```.yml``` file with the description of this first environment is provided in ```env_files/pyxnat_env.yml```
   
   b. ```hcp7t_fv_sleep_env```: This environment is valid for all other notebooks. A ```.yml``` file with the description of this second environment is provided in ```env_files/hcp7t_fv_sleep_env.yml```

**2. Download data from XNAT**

Notebook ```N00_DownloadDataFromHCPXNAT``` contains the necessary code to downlaod all necessary data files from XNAT Central.

This notebook used the ```pyxant_env``` environment.

Many cells on this notebook will take a long time (over one hour). Be ready for that with a nice cup of coffee.

**3. Basic QA**

Notebook ```N01_QA``` will help us identify resting-state runs with issues such as missing ET data, ET files that do not load correctly, or have an incorrect number of volumes. Those will not be used in further analyses. This notebook will write a dataframe with this information that it is used in subsequent notebooks to load only valid data.


