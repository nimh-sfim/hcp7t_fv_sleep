# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.9.1
#   kernelspec:
#     display_name: hcp7t_fv_sleep_env
#     language: python
#     name: hcp7t_fv_sleep_env
# ---

# # Description
#
# This notebook execute all additional pre-processing pipelines that take as input the minimally pre-processed data.
#
# This notebook, as previous ones, only sets the swarm file necessary to paralelize this part of the analysis across all runs. The core of the processing code is in ```N06_mPP_postProcessing.sh```. This bash script will perform the following steps for a given run:
#
# 1. Discard initial 10 seconds from fMRI data
#
# 2. Spatially smooth the minimally pre-processed data (FWHM = 4)
#
# 3. Estimate motion DVARS (based on mPP data prior to smoothing)
#
# 4. Run the Basic Pipeline: blur + bandpass (0.01 - 0.1 Hz) + mot regressors + legendre polynomials.
#
# 5. Run the Basic Pipeline without the bandpass. This will be the input to the RapidTide software in subsequent steps
#
# 6. Run the AFNI CompCor Pipeline: blur + bandpass (0.01 - 0.1 Hz) + mot regressors + legendre poly + 3 PCA from lateral ventricles. (Most likely will be discarded in review)
#
# 7. Run the AFNI CompCor+ Pipeline: blur + bandpass (0.01 - 0.1 Hz) + mot regressors + legendre poly + 3 PCA from lateral ventricles + FV signal. (Most likely will be discarded in review)
#
# 8. Run the Bezhadi CompCor Pipeline: blur + bandpass (0.01 - 0.1 Hz) + mot regressors + legendre poly + 5 PCA from WM + CSF mask.
#
# ## Outputs:
#
# * ```${DATA_DIR}/${SBJ}/${RUN}/${RUN}_mPP.blur.nii.gz```: Spatially smoothed fMRI data.
# * ```${DATA_DIR}/${SBJ}/${RUN}/${RUN}_mPP.blur.scale.nii.gz```: Spatially smoothed fMRI data in Signal Percent Change units.
# * ```${DATA_DIR}/${SBJ}/${RUN}/${RUN}_Movement_SRMS.1D```: Motion in terms of DVARs
# * ```${DATA_DIR}/${SBJ}/${RUN}/${RUN}_BASIC.nii.gz```: Output of the Basic pre-processing pipeline
# * ```${DATA_DIR}/${SBJ}/${RUN}/${RUN}_BASICnobpf.nii.gz```: Output of the Basic pre-processing pipeline (no filtering). Only used as input to rapitdite
# * ```${DATA_DIR}/${SBJ}/${RUN}/${RUN}_AFNI_COMPCOR.nii.gz```: Output from AFNI CompCor pre-processing pipeline
# * ```${DATA_DIR}/${SBJ}/${RUN}/${RUN}_AFNI_COMPCORp.nii.gz```: Output from AFNI CompCor+ pre-processing pipeline
# * ```${DATA_DIR}/${SBJ}/${RUN}/${RUN}_Behzadi_COMPCOR.nii.gz```: Output from Behzadi CompCor pre-processing pipeline
#

import pandas as pd
import numpy as np
import os.path as osp
import os
from utils.variables import Resources_Dir, DATA_DIR
from utils.basics import get_available_runs

# ***
# # 1. Load list of runs

Manuscript_Runs = get_available_runs('final')
print('++ INFO: Number of Runs = %d' % len(Manuscript_Runs))

# ***
#
# # 2. Generate Swarm Infrastructure: log directory, SWARM file

# Create Log Directory for swarm jobs
# ===================================
if not osp.exists('./N06_mPP_postProcessing.logs'):
    print('++ INFO: Creating logging dir')
    os.mkdir('./N06_mPP_postProcessing.logs')

# Create Swarm file for extracting representative power
# =====================================================
os.system('echo "#swarm -f ./N06_mPP_postProcessing.SWARM.sh -g 16 -t 16 --partition quick,norm --module afni --logdir ./N06_mPP_postProcessing.logs" > ./N06_mPP_postProcessing.SWARM.sh')
# Add entries regarding periods of eye closure
for item in Manuscript_Runs:
    sbj,run = item.split('_',1) 
    os.system('echo "export SBJ={sbj} RUN={run}; sh ./N06_mPP_postProcessing.sh" >> ./N06_mPP_postProcessing.SWARM.sh'.format(sbj=sbj, run=run))

# ***
#
# # 3. Check all outputs were generated correctly

# %%time
for item in Manuscript_Runs:
    sbj,run = item.split('_',1)
    for suffix in ['_mPP.blur.nii.gz','_mPP.blur.scale.nii.gz','_Movement_SRMS.1D','_BASIC.nii.gz','_BASICnobpf.nii.gz','_AFNI_COMPCOR.nii.gz','_AFNI_COMPCORp.nii.gz','_Behzadi_COMPCOR.nii.gz']:
        path = osp.join(DATA_DIR,sbj,run,'{run}{suffix}'.format(run=run,suffix=suffix))
        if not osp.exists(path):
            print('++ WARNING: Missing output [%s]' % path)
