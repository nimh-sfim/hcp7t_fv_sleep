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

# # Data Quality Assurance - Part 1
#
# This notebook will perform the following steps:
#
# 1. Load a list of subjects of interest (i.e., those with at least one resting-state scan at 7T)
# 2. Load motion estimates and compute Framewise Displacement (saves FD to disk on each run folder)
# 3. Attempt loading of ET files for each run (and mark those that are defective)
# 4. Construct a dataframe with the following information per run: correct number of TRs, correct spatial resolution, correct number of volumes, ET available, ET can be loaded
#
# A summary of this QA is saved to disk in ${RESOURCES_DIR}/QA_Part1_Results.pkl
# ***

# +
# %%time
from utils.basics          import get_7t_subjects, load_motion_info
from utils.variables       import RUNS, DATA_DIR, ProjectFiles_DF_Path, QA1_Results_DF_Path
from utils.ParseEyeLinkAsc import ParseEyeLinkAsc

import numpy as np
import pandas as pd
import os.path as osp
import nibabel as nib

VERBOSE=False
# -

# *** 
# ## 1. Check the Dataframe with information about available files

ProjectFiles_DF = pd.read_pickle(ProjectFiles_DF_Path)
print('++ INFO: Shape of Project Files_DF is %s' % str(ProjectFiles_DF.shape))

print('++ INFO: Number of Runs with ET(asc) file available: %d Runs' % (ProjectFiles_DF.shape[0] - ProjectFiles_DF['ET_ASC'].isna().sum()))
print('++ INFO: Number of Runs with ET(csv) file available: %d Runs' % (ProjectFiles_DF.shape[0] - ProjectFiles_DF['ET_CSV'].isna().sum()))

# ***
# ## 2. Load List of Subjects of interest

# Load List of Subjects with at least one resting-state scan
sbjs = get_7t_subjects()
print('++ Number of available subjects: %d' % len(sbjs))

# ***
# ## 4. Load Motion Information and Compute FrameWise Displacement
# This will generate a file per run with the traces of framewise displacepment for that particular run

# %%time
# Load Motion Information for all subjects available and create FD data frame for each run
mot_df = load_motion_info(sbjs, write_FD=True, fillnan=False, verbose=VERBOSE)

print('++ INFO: Shape of mot_df is %s' % str(mot_df.shape))
mot_df.head()

# ***
# ## 5. Check the Integrity of Eye Tracker Data Files & See if FD is low
#
# Unfortunately, not all eye tracking data files can be loaded properly. 
#
# During this initial QA, we will test whether or not a given ET file (e.g., that of one run) can be properly loaded or not
#
# In addition we will also store the previously computed Mean and Max Framewise Displacement

# +
# %%time
# Create Eamty DataFrame with the following columns:
# * Sbj = Subject ID
# * Run = Run ID
# * Dir Avail     = Does the directory for this run exists on our system?
# * Mot Avail     = Is the motion file for this run available on our system?
# * ET Avail      = Are both ET files for this run available on our system?
# * ET_OK         = Are we able to load (e.g., file is uncorrupted) the main ET File
df = pd.DataFrame(columns=['Sbj','Run','Dir Avail','Mot Avail','ET Avail', 'ET_OK'])

# For all subjects
for s,sbj in enumerate(sbjs):
    # For all possible runs
    for run in RUNS:
        # Create the path to this run directory (should it exists)
        drun_path = osp.join(DATA_DIR,str(sbj),run)
        if osp.exists(drun_path):
            # Create the path to the motion file (should it exists)
            mot_path  = osp.join(drun_path,'{run}_Movement_Regressors.txt'.format(run=run))
            # Create the path to the 
            et_asc_path   = osp.join(drun_path,'{run}_eyetrack.asc'.format(run=run))
            et_csv_path   = osp.join(drun_path,'{run}_eyetrack_summary.csv'.format(run=run))
            # Try loading the ET file without causing any type of exception
            if osp.exists(et_asc_path):
                try:
                    dfTrial,dfMsg,dfFix,dfSacc,dfBlink,dfSamples = ParseEyeLinkAsc(et_asc_path)
                    et_ok = True
                except: # If there was any issue (e.g., an exception), then set et_ok to False
                    et_ok = False
            # Update the dataframe with the information about this run
            df        = df.append({'Sbj':sbj,
                               'Run':run,
                               'Dir Avail':osp.exists(drun_path),
                               'Mot Avail':osp.exists(mot_path),
                               'ET Avail':osp.exists(et_asc_path ) & osp.exists(et_csv_path),
                               'ET_OK': et_ok}, 
                               ignore_index=True)
            if VERBOSE:
                print('INFO: Just finsished with subject {sbj} run {run}'.format(sbj=sbj, run=run))
        else: 
            print('WARNING: Subject {sbj} run {run} does not exists'.format(sbj=sbj, run=run))
df = df.infer_objects()
# -

# ***
# ## 6. Check the spatial resolution and length of the scans

run_list = [str(row['Sbj'])+'_'+row['Run'] for r,row in df.iterrows() ]

# %%time
df['Spatial Resolution OK'] = None
df['Nacq OK']               = None
df['TR OK']                 = None
print('++ INFO: Number of items to iter [%d]' % len(run_list))
print(' + ',end='')
for i,item in enumerate(run_list):
    sbj,run   = item.split('_',1)
    file_path = osp.join(DATA_DIR,sbj,run,run+'_mPP.nii.gz')
    if np.mod(i,50)==0:
        print('%i..' % i, end='')
    if not osp.exists(file_path):
        df.loc[((df['Sbj']==sbj) & (df['Run']==run),'Spatial Resolution OK')] = False
        df.loc[((df['Sbj']==sbj) & (df['Run']==run),'Nacq OK')] = False
        df.loc[((df['Sbj']==sbj) & (df['Run']==run),'TR OK')] = False
    else:
        file_img = nib.load(file_path)
        [dx, dy, dz, tr] = file_img.header.get_zooms()
        
        if np.isclose(dx,1.60) & np.isclose(dx,1.60) & np.isclose(dz,1.60):
            df.loc[((df['Sbj']==sbj) & (df['Run']==run),'Spatial Resolution OK')] = True
        else:
            df.loc[((df['Sbj']==sbj) & (df['Run']==run),'Spatial Resolution OK')] = False
            
        if np.isclose(tr,1.0):
            df.loc[((df['Sbj']==sbj) & (df['Run']==run),'TR OK')] = True
        else:
            df.loc[((df['Sbj']==sbj) & (df['Run']==run),'TR OK')] = False
        
        if file_img.shape[3] == 900:
            df.loc[((df['Sbj']==sbj) & (df['Run']==run),'Nacq OK')] = True
        else:
            df.loc[((df['Sbj']==sbj) & (df['Run']==run),'Nacq OK')] = False
print('')
df.head()

print("++ INFO: Number of Runs with directory available:            %d" % df[df['Dir Avail']==True].shape[0])
print("++ INFO: Number of Runs with ET available:                   %d" % df[df['ET Avail']==True].shape[0])
print("++ INFO: Number of Runs with ET OK:                          %d" % df[df['ET_OK']==True].shape[0])
print("++ INFO: Number of Runs with correct spatial resolution:     %d" % df[df['Spatial Resolution OK']==True].shape[0])
print("++ INFO: Number of Runs with correct number of acquisitions: %d" % df[df['Nacq OK']==True].shape[0])
print("++ INFO: Number of Runs with expected TR:                    %d" % df[df['TR OK']==True].shape[0])
print("++ ===============================================================")
print("++ INFO: Number of Runs with all controls OK:                %d" % df[(df['Dir Avail']==True) & 
                                                                             (df['ET Avail']==True) & 
                                                                             (df['ET_OK']==True) & 
                                                                             (df['Spatial Resolution OK']==True) &
                                                                             (df['Nacq OK']==True) &
                                                                             (df['TR OK']==True)].shape[0])

# ***
# ## Save the summary of this first QA part to disk

df.to_pickle(QA1_Results_DF_Path)

print('++ INFO: Number of runs missing ET files = %d RUNS' % (df[df['ET Avail']==False].shape[0]))
print('++ INFO: Number of runs with ET files available but unreadable = %d RUNS' % (df[df['ET_OK']==False].shape[0]))
