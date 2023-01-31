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

# + [markdown] tags=[]
# # Description - Differences in FC across scan types
#
# This notebook performs the following operations:
#
# * Create Swarm jobs to generate FC matrices for all runs via AFNI 3dNetCor
# * Check all FC matrices were generated correctly
# * Generates average connectivity matrices per scan type for all different pipelines
# * Generates histograms of FC values across the brain for the different conditions
# * Starts a dashboard where we can explore those FC matrices
# * Prepares files for subsequent statistical analyses using NBS software in MATLAB
# -

# # Import Libraries

# +
import os
import os.path as osp
import numpy as np
import xarray as xr
import pandas as pd
import panel as pn
import holoviews as hv
import hvplot.pandas
import matplotlib.pyplot as plt
from utils.variables import SCRIPTS_DIR, Resources_Dir, DATA_DIR
from utils.basics import get_available_runs

from shutil import rmtree
from random import sample
hv.extension('bokeh')
# -

# # Gather Port Information for launching the dashboard

port_tunnel = int(os.environ['PORT2'])
print('++ INFO: Second Port available: %d' % port_tunnel)

# # Load Run Lists (all, drowsy, awake)

# %%time
Manuscript_Runs = get_available_runs(when='final', type='all')
Awake_Runs      = get_available_runs(when='final', type='awake')
Drowsy_Runs     = get_available_runs(when='final', type='drowsy')
print('++ INFO: Number of Runs: Total = %d | Awake = %d | Drowsy = %d' % (len(Manuscript_Runs), len(Awake_Runs), len(Drowsy_Runs)))

# # Create Swarm Jobs (to compute FC per scan)

if not osp.exists('./N18_FCMatrix_Schaefer200.logs'):
    print('++ INFO: Creating logging dir')
    os.mkdir('./N18_FCMatrix_Schaefer200.logs')

# Create Swarm file for extracting representative power
# ==========================================================
os.system('echo "#swarm -f ./N18_FCMatrix_Schaefer200.SWARM.sh -g 10 -t 5 --partition quick,norm --module afni --logdir ./N18_FCMatrix_Schaefer200.logs" > ./N18_FCMatrix_Schaefer200.SWARM.sh')
for sbj_run in Manuscript_Runs:
    sbj,run  = sbj_run.split('_',1)
    os.system('echo "export SBJ={sbj} RUN={run}; sh ./N18_FCMatrix_Schaefer200.sh" >> ./N18_FCMatrix_Schaefer200.SWARM.sh'.format(sbj=sbj, run=run))

# # Run Swarm Jobs
#
# On a terminal in biowulf, run:
#
# ```bash
# # cd /data/SFIMJGC_HCP7T/hcp7t_fv_sleep/Notebooks
# # rm ./N18_FCMatrix_Schaefer200.logs/*
# swarm -f ./N18_FCMatrix_Schaefer200.SWARM.sh -g 10 -t 5 --partition quick,norm --module afni --logdir ./N18_FCMatrix_Schaefer200.logs
# ```

# + [markdown] tags=[]
# ***
# # Checking for missing files after completion of swarm jobs
# -

i=0
for sbj_run in Manuscript_Runs:
    sbj,run  = sbj_run.split('_',1)
    for suffix in ['Reference','BASIC', 'Behzadi_COMPCOR', 'AFNI_COMPCOR', 'AFNI_COMPCORp']:
        file = osp.join(DATA_DIR,sbj,run,run+'_'+suffix+'.Shaeffer2018_200Parcels_000.netcc')
        if not osp.exists(file):
            i = i + 1
            print('++ WARNING: [%d, %s] is missing.' % (i,file))
