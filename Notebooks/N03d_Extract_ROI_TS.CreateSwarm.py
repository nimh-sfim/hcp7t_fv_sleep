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
#     display_name: Vigilance Project
#     language: python
#     name: vigilance
# ---

# # Description
#
# 1. Create average (group-level) T1, T2 and EPI reference image
#
# 2. Create consensus (group-level) 4th Ventricle ROI
#
# 2. Create consensus (group-level) iFV ROI
#
# 3. Create a Swarm file for batch processing of all subjects. Processing steps include:
#
# * Generate new mPP dataset with Signal Percent Units
#
# * Extract representative time-series (based on SPC dataset) for GM, Full Brain (Global Signal), FV (using group ROI), FV (using subject-specific ROI), lateral ventricles, WM.
#
# ## Ouputs
#
# ### Group-level Files
#
# * ```${DATA_DIR}/ALL/ALL_T1w_restore_brain.nii.gz```: average T1 image across all subjects
# * ```${DATA_DIR}/ALL/ALL_T2w_restore_brain.nii.gz```: average T2 image across all subjects
# * ```${DATA_DIR}/ALL/ALL_EPI.nii.gz```: average EPI across all runs
# * ```${DATA_DIR}/ALL/ALL_ROI.V4.mPP.nii.gz```: group-level FV ROI.
# * ```${DATA_DIR}/ALL/ALL_ROI.V4lt.mPP.nii.gz```: group-level iFV ROI.
#
#
# ### Run Specific Outputs
# * ```${DATA_DIR}/${SBJ}/${RUN}/${RUN}_mPP.scale.nii.gz```:     minimally pre-processed dataset in units of signal percent change.
# * ```${DATA_DIR}/${SBJ}/${RUN}/${RUN}_mPP.Signal.V4_grp.1D```: representative timeseries for FV (using group-level ROI)
# * ```${DATA_DIR}/${SBJ}/${RUN}/${RUN}_mPP.Signal.V4_e.1D```:   representative timeseries for FV (using subject-specific ROI)
# * ```${DATA_DIR}/${SBJ}/${RUN}/${RUN}_mPP.Signal.Vl_e.1D```:   representative timeseries for the laterval ventricles eroded (subject-specific)
# * ```${DATA_DIR}/${SBJ}/${RUN}/${RUN}_mPP.Signal.FB.1D```:     global signal (full brain - subject-specific)
# * ```${DATA_DIR}/${SBJ}/${RUN}/${RUN}_mPP.Signal.GM.1D```:     global signal (GM ribbon only - subject-specific)
# * ```${DATA_DIR}/${SBJ}/${RUN}/${RUN}_mPP.Signal.WM_e.1D```:   representative WM signal (subject-specific)
#
# ***

# +
import os
import os.path as osp
import numpy as np
import pandas as pd
import subprocess

from utils.variables import Resources_Dir, DATA_DIR
from utils.basics import get_7t_subjects, get_available_runs
# -

Sbjs     = get_7t_subjects()
Runs     = get_available_runs(when='final', type='all')
print('++ INFO: Number of subjects: %d' % len(Sbjs))
print('++ INFO: Number of runs:     %d' % len(Runs))

# ***
# ## 1. Create folder where to save group-level files

ALL_DIR = osp.join(DATA_DIR,'ALL')
if not osp.exists(ALL_DIR):
    os.mkdir(ALL_DIR)
    print('++ INFO: New folder for group results created')
else:
    print('++ INFO: Group level folder already existed. None were created at this time')

# ***
# # 2. Create Average T1, T2 and EPI images for reference in group folder

# %%time
t1_files = ['../{sbj}/T1w_restore_brain.nii.gz'.format(sbj=sbj) for sbj in Sbjs]
command  = 'module load afni; \
            cd {all_dir}; \
            3dMean -overwrite -prefix ALL_T1w_restore_brain.nii.gz {files};'.format(all_dir=ALL_DIR,files=' '.join(t1_files))
output  = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
print(output.strip().decode())

# %%time
t2_files = ['../{sbj}/T2w_restore_brain.nii.gz'.format(sbj=sbj) for sbj in Sbjs]
command  = 'module load afni; \
            cd {all_dir}; \
            3dMean -overwrite -prefix ALL_T2w_restore_brain.nii.gz {files};'.format(all_dir=ALL_DIR,files=' '.join(t2_files))
output  = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
print(output.strip().decode())

# %%time
epi_files = ['../{sbj}/{run}/{run}_mPP.nii.gz[0]'.format(sbj=item.split('_')[0],run=item.split('_',1)[1]) for item in Runs]
command   = 'module load afni; \
             cd {all_dir}; \
             3dTcat -overwrite -prefix ALL_EPI_firstvols.nii.gz {files}; \
             3dTstat -overwrite -mean -prefix ALL_EPI.nii.gz ALL_EPI_firstvols.nii.gz; \
             rm ALL_EPI_firstvols.nii.gz'.format(all_dir=ALL_DIR, files=' '.join(epi_files))
output  = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
print(output.strip().decode())

# *** 
# ## 3. Create Forth Ventricle (FV) group-level ROI

# %%time
fv_masks = ['../{sbj}/ROI.V4.mPP.nii.gz'.format(sbj=sbj) for sbj in Sbjs]
command  = 'module load afni; \
            cd {all_dir}; \
            3dMean -overwrite -prefix ALL_ROI.V4.mPP_avg.nii.gz {files}; \
            3dcalc -overwrite -a ALL_ROI.V4.mPP_avg.nii.gz -expr "ispositive(a-0.98)" -prefix ALL_ROI.V4.mPP.nii.gz; \
            3dcalc -overwrite -a ALL_ROI.V4.mPP.nii.gz -expr "a*isnegative(k-28)" -prefix ALL_ROI.V4lt.mPP.nii.gz; \
            rm ALL_ROI.V4.mPP_avg.nii.gz'.format(all_dir=ALL_DIR, files=' '.join(fv_masks))
output  = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
print(output.strip().decode())

# ***
# # 3. Create Swarm Scripts for job submission

if not osp.exists('./N03d_Extract_ROI_TS.logs'):
    print('++ INFO: Creating logging dir: ./N03d_Extract_ROI_TS.logs')
    os.mkdir('./N03d_Extract_ROI_TS.logs')
else:
    print('++ INFO: Logging directory already existed')

# Create Swarm file for extracting representative power
# ==========================================================
os.system('echo "#swarm -f ./N03d_Extract_ROI_TS.SWARM.sh -g 64 -t 32 --partition quick,norm --logdir ./N03d_Extract_ROI_TS.logs" > ./N03d_Extract_ROI_TS.SWARM.sh')
for sbj_run in Runs:
    sbj,run  = sbj_run.split('_',1)
    out_dir  = osp.join(DATA_DIR,sbj,run)
    os.system('echo "export SBJ={sbj} RUN={run}; sh ./N03d_Extract_ROI_TS.sh" >> ./N03d_Extract_ROI_TS.SWARM.sh'.format(sbj=sbj, run=run, ddir=DATA_DIR))

# ***
#
# # 4. Check all outputs exists
#
# ## 4.1. Group Files

for group_file in ['ALL_EPI.nii.gz','ALL_T1w_restore_brain.nii.gz','ALL_T2w_restore_brain.nii.gz','ALL_ROI.V4.mPP.nii.gz','ALL_ROI.V4lt.mPP.nii.gz']:
    path = osp.join(DATA_DIR,'ALL',group_file)
    if not osp.exists(path):
        print("++ WARNING: [%s] is missing." % path)

# ## 4.2 Subject-specific Files

# %%time
for item in Runs:
    sbj,run=item.split('_',1)
    for suffix in ['scale.nii.gz', 'Signal.V4_grp.1D', 'Signal.V4_e.1D', 'Signal.Vl_e.1D', 'Signal.FB.1D', 'Signal.GM.1D', 'Signal.WM_e.1D']:
        path = osp.join(DATA_DIR,sbj,run,'{run}_mPP.{suffix}'.format(run=run, suffix=suffix))
        if not osp.exists(path):
            print('++ WARNING: [%s] is missing.' % path)


