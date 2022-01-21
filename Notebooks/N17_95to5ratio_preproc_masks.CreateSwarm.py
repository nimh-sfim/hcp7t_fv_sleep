# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.12.0
#   kernelspec:
#     display_name: hcp7t_fv_sleep_env
#     language: python
#     name: hcp7t_fv_sleep_env
# ---

# # Description
#
# This notebook does some basic pre-processing on data in orig space for the purpose of conducting the per-slice 95th-5th percentile analysis.
#
# In particular, for each Drowsy scan of interest, we perform the following steps:
#
# 1. Discard initial 10s of data --> ```${RUN}_orig.discard.nii.gz```
# 2. Time-shift correct the data --> ```${RUN}_orig.tshift.nii.gz```
# 3. Detrend the time-shifted data (mean re-introduced) --> ```${RUN}_orig.detrend.nii.gz```
# 4. Compute MEAN and STDV maps --> ```${RUN}_orig.MEAN.nii.gz``` and ```${RUN}_orig.STDV.nii.gz```
# 5. Create Full Brain mask --> ```${RUN}_orig.mask.FB_auto.nii.gz```
# 6. Create Automated version of the ventricular mask --> ```${RUN}_orig.mask.FV.auto_union.nii.gz```
# 7. Copy automated mask into ```${RUN}_orig.mask.FV.manual.nii.gz``` 
#
# After this, it was necessary to have manual intervention to generate an accurate version of the ventricular mask in original space
#
# ***
#
# #### Load necessary libraries

import os.path as osp
import os
import numpy as np
import subprocess
from utils.variables import Resources_Dir, DATA_DIR

# #### Load list of scans with long EC segments on interest

selected_scans_csv_path = osp.join(Resources_Dir,'EC_lowMot_highPSD_scans.csv')
scan_list               = np.loadtxt(selected_scans_csv_path,dtype=str)
print(scan_list)

# + [markdown] tags=[]
# #### Create Log Dir for swarm jobs

# + tags=[]
# Create log dir for swarm jobs to write their output and error messages
# ======================================================================
if not osp.exists('./N17_95to5ratio_preproc_masks.logs'):
    print('++ INFO: Creating logging dir: N17_95to5ratio_preproc_masks.logs')
    os.mkdir('./N17_95to5ratio_preproc_masks.logs')
# -

# #### Create Swarm File

# Create Swarm file for extracting representative power
# ======================================================
os.system('echo "#swarm -f ./N17_95to5ratio_preproc_masks.SWARM.sh -g 16 -t 16 --partition quick,norm --module afni --logdir ./N17_95to5ratio_preproc_masks.logs" > ./N17_95to5ratio_preproc_masks.SWARM.sh')
for item in scan_list:
    sbj,run = item.split('_',1)
    os.system('echo "export SBJ={sbj} RUN={run}; sh ./N17_95to5ratio_preproc_masks.sh" >> ./N17_95to5ratio_preproc_masks.SWARM.sh'.format(sbj=sbj,run=run))

# ***
#
# # BEFORE GOING TO N18, YOU NEED TO MANUALLY CORRECT THE MASKS
#
# In AFNI correct the ```${RUN}_orig.mask.FV.manual.nii.gz``` so that it only includes voxels of interest in the 4th ventricle
#
