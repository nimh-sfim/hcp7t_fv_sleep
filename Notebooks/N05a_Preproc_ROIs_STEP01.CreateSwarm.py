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
# This notebook does the following operations
#
# 1. Generates a group directory ```ALL``` in the data directory ```DATA_DIR```
#
# 2. Creates a file named ```mPP_Grid.nii.gz``` in ```DATA_DIR/ALL``` that will serve as the reference grid for spatial normalixation steps (e.g., bring anaotmical masks to minimally pre-processed (mPP) fMRI space.
#
# 3. Add Label Table to Schaefer Atlas (this is helpful when looking at the data in AFNI).
#
# 4. Creates a Swarm File so that bash script ```N04_Preprocess_masks_and_ROIs.sh``` can be run in parallel in all subjects, and perform the following operations:
#
#     a. Sets AFNI space field equal to MNI in the brainmask, GM ribbon, and parcellation downloaded from ConnectomeDB. Those are set to ORIG, despite the data being in MNI space.
#
#     b. Create the following masks by combining different parcellation results:
#
# In MNI Space with anatomical grid:
#
#  * ```ROI.FB.nii.gz```: Subject-specific Full brain mask (copy of brainmask_fs.nii.gz)
#  * ```ROI.GM.nii.gz```: Subject-specific GM Cortical ribbon (GM_Ribbon.nii.gz as downloaded from ConnectomeDB also contains masks for WM)
#  * ```ROI.V4.nii.gz```: Subject-specific Forth Ventricle mask
#  * ```ROI.Vl.nii.gz```: Subject-specific Lateral Ventricles mask
#  * ```ROI.WM.nii.gz```: Subject-specific WM mask
#     
# In MNI Space with fMRI grid:
#
#  * ```ROI.automask.nii.gz```: Subject-specific Full brain mask that only includes voxels with data on all rest runs. This is used to constrain all other masks.
#  * ```ROI.FB.mPP.nii.gz```: Subject-specific Full brain mask (copy of brainmask_fs.nii.gz)
#  * ```ROI.GM.mPP.nii.gz```: Subject-specific GM Cortical ribbon (GM_Ribbon.nii.gz as downloaded from ConnectomeDB also contains masks for WM)
#  * ```ROI.V4.mPP.nii.gz```: Subject-specific Forth Ventricle mask
#  * ```ROI.Vl.mPP.nii.gz```: Subject-specific Lateral Ventricles mask
#  * ```ROI.WM.mPP.nii.gz```: Subject-specific WM mask
#   
# In MNI Space with fMRI grid after erosion (1 voxel in Anat grid):
#
#  * ```ROI.FB_e.mPP.nii.gz```: Subject-specific Full brain mask (copy of brainmask_fs.nii.gz)
#  * ```ROI.GM_e.mPP.nii.gz```: Subject-specific GM Cortical ribbon (GM_Ribbon.nii.gz as downloaded from ConnectomeDB also contains masks for WM)
#  * ```ROI.V4_e.mPP.nii.gz```: Subject-specific Forth Ventricle mask
#  * ```ROI.Vl_e.mPP.nii.gz```: Subject-specific Lateral Ventricles mask * ```ROI.WM_e.mPP.nii.gz```: Subject-specific WM mask
#  * ```ROI.compcorr.mPP.nii.gz```: Subject-specific combined ventricular and WM mask for CompCorr
#
# ***
# > **IMPORTANT NOTE:** Parts of this study were conducted using the NIH's High Performance Computing system (https://hpc.nih.gov). The code in this notebook generates a swarm file that permits parallel pre-processing of all runs using that particular system. This code may need to be modified for your particular computational environment.
#
# > **IMPORTANT NOTE 2:** Similarly, this notebook assumes that AFNI (https://afni.nimh.nih.gov/) is avialable and in your PATH.
#
# ***

import os
import os.path as osp
import numpy as np
import pandas as pd
import subprocess
from utils.variables import DATA_DIR, ATLAS_DIR, ATLAS_NAME
from utils.basics import get_7t_subjects

# ***
# ## 1. Create Reference Grid File
#
# All minimally-preprocessed resting-state scans are already in the same space and grid. Therefore, any run from any subject can serve to create a file to be used as a reference (or master) grid in spatial normalization operations. ?Here we decided to use the first run from subject 100610.

# %%time
command = 'module load afni; \
           3dcalc -overwrite -a {data_dir}/100610/rfMRI_REST1_PA/rfMRI_REST1_PA_mPP.nii.gz[0] -expr "a" -prefix {data_dir}/ALL/mPP_Grid.nii.gz'.format(data_dir=DATA_DIR)
output  = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
print(output.strip().decode())

# ***
# ## 2. Add Table Label to Schaefer Atlas File

command = 'module load afni; \
           cd {atlas_dir}; \
           3drefit -space MNI {atlas}_order_FSLMNI152_2mm.nii.gz; \
           @MakeLabelTable -lab_file {atlas}_order.txt 1 0 -labeltable {atlas}_order.niml.lt -dset {atlas}_order_FSLMNI152_2mm.nii.gz;'.format(atlas_dir=ATLAS_DIR, atlas=ATLAS_NAME) 
output  = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
print(output.strip().decode())

# ***
# ## 3. Create Swarm File

# Load list of available subjects
# ===============================
Sbjs = get_7t_subjects() 
print('++ INFO: Number of Subjects: %d' % len(Sbjs))

# Create log dir for swarm jobs to write their output and error messages
# ======================================================================
if not osp.exists('./N05a_Preproc_ROIs_STEP01.logs'):
    print('++ INFO: Creating logging dir: N05a_Preproc_ROIs_STEP01.logs')
    os.mkdir('./N05a_Preproc_ROIs_STEP01.logs')

# Create Swarm file for extracting representative power
# ======================================================
os.system('echo "#swarm -f ./N05a_Preproc_ROIs_STEP01.SWARM.sh -g 16 -t 16 --partition quick,norm --module afni --logdir ./N05a_Preproc_ROIs_STEP01.logs" > ./N05a_Preproc_ROIs_STEP01.SWARM.sh')
for sbj in Sbjs:
    os.system('echo "export SBJ={sbj}; ./N05a_Preproc_ROIs_STEP01.sh" >> ./N05a_Preproc_ROIs_STEP01.SWARM.sh'.format(sbj=sbj))

# ***
# # 5. Submit jobs to the cluster
#
# ```bash
# swarm -f ./N05a_Preproc_ROIs_STEP01.SWARM.sh -g 16 -t 16 --partition quick,norm --module afni --logdir ./N05a_Preproc_ROIs_STEP01.logs
# ```
#
# You can check the status of your jobs with
#
# ```bash
# squeue -u <your_user_name>
# ```
#
# ***
# # 6. Check for missing outputs
#
# Once all your batch jobs have completed, you can run the following code to check that all outputs were created as expected

# %%time
for sbj in Sbjs:
    for file in ['ROI.automask.nii.gz','ROI.FB.nii.gz',       'ROI.GM.nii.gz',       'ROI.V4.nii.gz',       'ROI.Vl.nii.gz',       'ROI.WM.nii.gz', 
                 'ROI.FB.mPP.nii.gz',   'ROI.GM.mPP.nii.gz',   'ROI.V4.mPP.nii.gz',   'ROI.Vl.mPP.nii.gz',   'ROI.WM.mPP.nii.gz',
                 'ROI.FB_e.mPP.nii.gz', 'ROI.GM_e.mPP.nii.gz', 'ROI.V4_e.mPP.nii.gz', 'ROI.Vl_e.mPP.nii.gz', 'ROI.WM_e.mPP.nii.gz']:
        aux_path = osp.join(DATA_DIR,sbj,file)
        if not osp.exists(aux_path):
            print ('++ WARNING: Output missing [%s]' % aux_path)
