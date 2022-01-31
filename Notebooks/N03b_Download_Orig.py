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
#     display_name: Non Brain Signals (pyxnat)
#     language: python
#     name: non_brain_signals
# ---

# # Description
#
# This notebook will download from XNAT central the un-preprocessed functional scans for the subset of subjects that we will use to do analyses in original space.
#
# * Initially I thought I would only download such data for a few scans to check the validity of the inflow hypothesis.
#
# * Later I decided I would use HAPPY to extract the cardiac signal from all scans. This means I had to download the original data for the 404 scans used in this study

import pyxnat
import os.path as osp
import os
import numpy as np
from utils.variables import DATA_DIR, Resources_Dir
from utils.basics import get_available_runs

# #### Load list of scans

scan_list = get_available_runs(when='final',type='all')
print('++ INFO: Number of scans = %d' % len(scan_list))

# + active=""
# selected_scans_csv_path = osp.join(Resources_Dir,'EC_lowMot_highPSD_scans.csv')
# scan_list               = np.loadtxt(selected_scans_csv_path,dtype=str)
# print(scan_list)
# -

# #### Connect to XNAT CENTRAL

cbd = pyxnat.Interface('https://db.humanconnectome.org','','')
XNAT_PROJECT = 'HCP_1200'

hcp1200 = cbd.select.project(XNAT_PROJECT)

# #### Download Data

# + tags=[]
# %%time
for scan in scan_list:
    sbj,run = scan.split('_',1)
    _,run_id, run_ap = run.split('_')
    pkg              = run + '_unproc'
    dest_path        = osp.join(DATA_DIR,sbj,run,run+'_orig.nii.gz')
    print('++ INFO: Downloading from XNAT %s' % (dest_path))
    xnat_path = hcp1200.subject(str(sbj)).experiment(str(sbj)+'_7T').resource(pkg).file(sbj+'_7T_'+run+'.nii.gz')
    xnat_path.get(dest_path)
