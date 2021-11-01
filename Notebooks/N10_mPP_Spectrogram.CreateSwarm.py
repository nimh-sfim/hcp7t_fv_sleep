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

# # Description - Spectrogram Analysis (Evolution of Spectral Power over time)
#
# This notebook generates swarm files to compute spectrograms for all scans. The primary outputs for each scan are:
#
# * ```${DATA_DIR}/${SBJ}/${RUN}/${RUN}_mPP.Signal.V4_grp.Spectrogram.pkl```: Spectrogram for a given scan
# * ```${DATA_DIR}/${SBJ}/${RUN}/${RUN}_mPP.Signal.V4_grp.Spectrogram_BandLimited.pkl```: Timeseries of average power in the Sleep band for a given scan

import os
import os.path as osp
import numpy as np
import pandas as pd
from utils.variables import Resources_Dir, DATA_DIR
from utils.basics import get_available_runs

# ## Get list of Scans included in the manuscript

Manuscript_Runs = get_available_runs(when='final', type='all')
print('++ INFO: Number of Runs = %d' % len(Manuscript_Runs))

# ## Create Logging folder for Swarm jobs

if not osp.exists('./N10_mPP_Spectrogram.logs'):
    print('++ INFO: Creating logging dir')
    os.mkdir('./N10_mPP_Spectrogram.logs')

# ## Create Swarm File

# Create Swarm file for extracting representative power
# ==========================================================
os.system('echo "#swarm -f ./N10_mPP_Spectrogram.SWARM.sh -g 16 -t 16 --partition quick,norm --time 00:20:00 --logdir ./N10_mPP_Spectrogram.logs" > ./N10_mPP_Spectrogram.SWARM.sh')
for sbj_run in Manuscript_Runs:
    sbj,run  = sbj_run.split('_',1)
    out_dir  = osp.join(DATA_DIR,sbj,run)
    for region in ['V4_grp']:
        os.system('echo "export SBJ={sbj} REGION={reg} RUN={run} DATADIR={ddir}; sh ./N10_mPP_Spectrogram.sh" >> ./N10_mPP_Spectrogram.SWARM.sh'.format(sbj=sbj, run=run, reg=region, ddir=DATA_DIR))

# ## Submit jobs to the cluster
#
# Open a terminal in biowulf and run the following commands:
#
# ```bash
# # cd /data/SFIMJGC_HCP7T/hcp7t_fv_sleep/Notebooks
# # rm /N10_mPP_Spectrogram.logs/*
# swarm -f ./N10_mPP_Spectrogram.SWARM.sh -g 16 -t 16 --partition quick,norm --time 00:20:00 --logdir ./N10_mPP_Spectrogram.logs
# watch -n 30 squeue -u javiergc
# ```

# ***
#
# ### Test all outputs have been generated

num_files=0
suffix='mPP'
region='V4_grp'
for sbj_run in Manuscript_Runs:
    sbj,run  = sbj_run.split('_',1)
    out_file01 = osp.join(DATA_DIR,sbj,run,'{run}_{suffix}.Signal.{region}.Spectrogram.pkl'.format(run=run, region=region, suffix=suffix))
    out_file02 = osp.join(DATA_DIR,sbj,run,'{run}_{suffix}.Signal.{region}.Spectrogram_BandLimited.pkl'.format(run=run, region=region, suffix=suffix))
    for out_file in [out_file01, out_file02]:
        if not osp.exists(out_file):
            print('++ WARNING: File missing [%s]' % out_file)
        else:
            num_files +=1
print('++ INFO: Number of available files = %d' % num_files)
