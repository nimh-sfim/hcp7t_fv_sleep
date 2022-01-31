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

# Forth Ventricle Signal (NeuroImage 2021) - Data Download from XNAT
# ==================================================================
#
# This notebook downloads the 7T resting state sample form the Human Connectome Project dataset using the pyxnat library.
#
# Prior to using this notebook, you need to manually download the Behavioral Info file for the 7T dataset directly from the ConnectomeDB website.
#
# The notebook expects the downloaded file to be located at: <PRJ_DIR>/Resources/7T_SubjectList.csv. 
#
# Project Setup
# ---------------
#
# Prior to running any script, it is recommended that you correctly setup the following variables to match you environment:
#
# * **SCRIPTS_DIR**: It should point to the folder containing all the scripts for this project (e.g., '/data/SFIMJGC_HCP7T/hcp7t_fv_sleep')
# * **DATA_DIR**: It should point to the folder where the HCP data will be downloaded (e.g., '/data/SFIMJGC_HCP7T/HCP7T')
#
# What does this notebook do
# ----------------------------
#
# * Creates one folder per subject inside DATA_DIR (184 folders)
# * Attempts to **download the T1 map (T1w_restore_brain.nii.gz)** in MNI space for each individual subject and place it in DATA_DIR/SBJ_DIR
# * Attempts to **download the T2 map (T2w_restore_brain.nii.gz)** in MNI space for each individual subject and place it in DATA_DIR/SBJ_DIR
# * Attempts to **downlaod the full brain mask (brainmask_fs.nii.gz)** in MNI space at the resolution of the T1 and place it in DATA_DIR/SBJ_DIR
# * Attempts to **download the GM ribbon mask (GM_Ribbon.nii.gz)** in MNI space at the resolution of the T1 and place it in DATA_DIR/SBJ_DIR
# * Attempts to **download the Freesurfer parcellation (aparc.a2009s+aseg.nii.gz)** in MNI space at the resolution of the T1 and place in DATA_DIR/SBJ_DIR
# * Gather information regarding what rest runs are available for each subject. If one is available, a corresponding run folder is created in DATA_DIR/SBJ_DIR
#
# In addition, for each existing resting-state run the following extra files are downloaded and placed in the appropriate run folder (e.g., rfMRI_REST1_PA):
#
# * Attempts to **download the minimally pre-processed data** for this run (e.g., rfMRI_REST1_PA_mPP.nii.gz).
# * Attempts to **download the raw / unpreprocessed data** for this run (e.g., rfMRI_REST_PA_orig.nii.gz).
# * Attempts to **download the EPI reference scan** for this run (e.g., rfMRI_REST1_PA_SBRef.nii.gz).
# * Attempts to **download the motion parameters** for this run (e.g., rfMRI_REST1_PA_Movement_Regressors.txt).
# * Attempts to **download the first derivative of the motion parameters** for this run (e.g., rfMRI_REST1_PA_Movement_Regressors_dt.txt).
# * Attempts to **download both eye tracking files** for this run (e.g., rfMRI_REST1_PA_eyetrack.asc,  rfMRI_REST1_PA_eyetrack_summary.csv).
#
# Finally, the notebook creates an additional pkl file with a dataframe listing the final location of all the files that were downloaded from XNAT. When a file was missing from
# the database (e.g., an eye tracker file not present in ConnectomeDB) the corresponding cell contains a NaN. The path to this pickle file is given by **ProjectFiles_DF_Path**, which can be found in utils/variables.py
#
#
# > **Importnat Results:** There are 184 subjects in this dataset. Although target was to acquire 4 resting runs per subject, this was not always achieved. Similarly, although the original target was to have concurrent ET in all runs, that was also not achieved.
#
# > **NOTE**: This notebook must be run with the Non Brain Signals (pyxnat) environment. This is the only environment that contains a working version of pyxnat. Becuase that is uncomatible with the latest versions of many of the other libraries used in this study, we have this separate environment just for this download notebook
#
# > **NOTE**: One subject is completely missing from the final dataset becuase we later realized that the parcellation for such subject had failed. The problematic subject is: 178647
#
# * Original Target:                     184 * 4 = 736 rest scans
# * Existing Rest Scans:                 723 
# * Scans after removing subject 178647: 719
#
# ***

import pyxnat
import pandas as pd
import numpy  as np
import os.path as osp
import glob
import os
import shutil
import fnmatch
from utils.variables import DATA_DIR, SCRIPTS_DIR, SbjList_Orig_Path, ProjectFiles_DF_Path, Avail_fMRI_Runs_Info_DF_Path
from utils.variables import XNAT_USER, XNAT_PASSWORD

cbd = pyxnat.Interface('https://db.humanconnectome.org',XNAT_USER,XNAT_PASSWORD)
XNAT_PROJECT = 'HCP_1200'

# ## 1. Get the list of subjects with at least one rest run on the 7T
#
# I downloaded the CSV file from the connectomeDB website. This one contains one entry per subject with a lot of different variables. One such variable is the number of resting-state scans conducted at 7T. We use that variable to select subjects with at least one 7T resting-state scan.

print('++ INFO: Gatheting subject list from: %s' % SbjList_Orig_Path)
SbjInfo = pd.read_csv(SbjList_Orig_Path)
SbjInfo.head()

SbjList_7T = list(SbjInfo[SbjInfo['7T_RS-fMRI_Count']>0]['Subject'].values.astype(str))
print('++ Number of Subjects with 7T resting-state scans is: %d' % len(SbjList_7T))
print('++ Number of 7T resting-state scans is: %d' % SbjInfo['7T_RS-fMRI_Count'].sum())

# ***
# ## 2. Create subject directories in DATA_DIR folder

for sbj in SbjList_7T:
    aux_path = osp.join(DATA_DIR,sbj)
    if not osp.exists(aux_path):
        os.mkdir(aux_path)
    else:
        print('++WARNING: Directory already existed. Not created [%s]' %sbj)

# ***
# ## 3. Download the MNI T1w, T2w, brainmask, fs_parcellation and GM ribbon mask to the subject directory
#
# Those files are per-subject, not per run... so they can be downloaded before we have more detailed information about what rest scans actually exists in XNAT.

# %%time
hcp1200 = cbd.select.project(XNAT_PROJECT)
for i,sbj in enumerate(SbjList_7T):
    print('--> Downloading anatomical data for sbj [%d, %s]' %(i,str(sbj)))
    sbj_dir     = osp.join(DATA_DIR,str(sbj))
    xnat_t1w  = hcp1200.subject(str(sbj)).experiment(str(sbj)+'_3T').resource('Structural_preproc').file('MNINonLinear/T1w_restore_brain.nii.gz')
    xnat_t2w  = hcp1200.subject(str(sbj)).experiment(str(sbj)+'_3T').resource('Structural_preproc').file('MNINonLinear/T2w_restore_brain.nii.gz')
    xnat_bm   = hcp1200.subject(str(sbj)).experiment(str(sbj)+'_3T').resource('Structural_preproc').file('MNINonLinear/brainmask_fs.nii.gz')
    xnat_rois = hcp1200.subject(str(sbj)).experiment(str(sbj)+'_3T').resource('Structural_preproc').file('MNINonLinear/aparc.a2009s+aseg.nii.gz')
    xnat_rib  = hcp1200.subject(str(sbj)).experiment(str(sbj)+'_3T').resource('Structural_preproc').file('MNINonLinear/ribbon.nii.gz')
    t1w_path  = osp.join(sbj_dir,'T1w_restore_brain.nii.gz')
    t2w_path  = osp.join(sbj_dir,'T2w_restore_brain.nii.gz')
    bm_path   = osp.join(sbj_dir,'brainmask_fs.nii.gz')
    rois_path = osp.join(sbj_dir,'aparc.a2009s+aseg.nii.gz')
    rib_path  = osp.join(sbj_dir,'GM_Ribbon.nii.gz')

    if not osp.exists(t1w_path):
        xnat_t1w.get(t1w_path)
    else:
        print('++WARNING: T1 file already existed. Not created [%s]' % sbj)
    if not osp.exists(t2w_path):
        xnat_t2w.get(t2w_path)
    else:
        print('++WARNING: T2 file already existed. Not created [%s]' % sbj)
    if not osp.exists(bm_path):
        xnat_bm.get(bm_path)
    else:
        print('++WARNING: Brain mask file already existed. Not created [%s]' % sbj)
    if not osp.exists(rois_path):
        xnat_rois.get(rois_path)
    else:
        print('++WARNING: Parcellation file already existed. Not created [%s]' % sbj)
    if not osp.exists(rib_path):
        xnat_rib.get(rib_path)
    else:
        print('++WARNING: GM Ribbon file already existed. Not created [%s]' % sbj)


# ***
# ## 4. Create Run Directories only for those runs that actually were acquired
#
# This is very time-intensive cell that traverses the XNAT hierarchy to gather information about what runs actually exists for each subject. For each run, it creates an entry in a dataframe with the subject ID, run ID, and path where data associated with that run will be saved in biowulf

# %%time
# Get list of available runs per subject (Quite time consuming step)
# ==================================================================
df = pd.DataFrame(columns=['Sbj','Run','Path'])
hcp1200 = cbd.select.project(XNAT_PROJECT)
for sbj in SbjList_7T:
    print('--> Subject %s' % str(sbj))
    sbj_dir    = osp.join(DATA_DIR,str(sbj))    
    expt       = hcp1200.subject(str(sbj)).experiment(str(sbj)+'_7T')
    resources  = [res.label() for res in expt.resources()]
    avial_pkgs = fnmatch.filter(resources,'rfMRI_REST?_??_unproc')
    print(' -> Aavailable runs: %s' % str(avial_pkgs))
    for pkg in avial_pkgs:
        run_name = pkg.replace('_unproc','')
        run_dir  = osp.join(DATA_DIR,sbj,run_name)
        if not osp.exists(run_dir):
            print(' > Creating directory [%s]' % run_dir)
            os.mkdir(run_dir)
        else:
            print('++WARNING: Run directory already existed. Not created [%s]' %sbj)
        df = df.append({'Sbj':sbj,'Run':run_name,'Path':run_dir}, ignore_index=True)

df.to_pickle(Avail_fMRI_Runs_Info_DF_Path)
print('++ INFO: Information about existing resting state runs saved to [%s]' % Avail_fMRI_Runs_Info_DF_Path)

# Only run if needed (e.g., you did not run all the previous cells and want to check a previously saved version of this dataframe)
df = pd.read_pickle(Avail_fMRI_Runs_Info_DF_Path)

print('Number of Runs with at least fMRI data in ConnectomeDB: %d runs' % df.shape[0])
df.head()

# ***
# ## 5. Download minimally pre-processed data, motion and ref scan

orig_path1200 = cbd.select.project(XNAT_PROJECT)
for index,data in df.iterrows():
    sbj = data['Sbj']
    run = data['Run']
    _,run_id, run_ap = run.split('_')
    pkg = run + '_preproc'
    xnat_mPP   = hcp1200.subject(str(sbj)).experiment(str(sbj)+'_7T').resource(pkg).file('MNINonLinear/Results/{dir}/{dir}.nii.gz'.format(dir='rfMRI_'+run_id+'_7T_'+run_ap))
    mPP_path   = osp.join(DATA_DIR,sbj,run,run+'_'+'mPP.nii.gz')
    xnat_SBref = hcp1200.subject(str(sbj)).experiment(str(sbj)+'_7T').resource(pkg).file('MNINonLinear/Results/{dir}/{dir}_SBRef.nii.gz'.format(dir='rfMRI_'+run_id+'_7T_'+run_ap))
    SBref_path = osp.join(DATA_DIR,sbj,run,run+'_'+'SBRef.nii.gz')
    xnat_motdt = hcp1200.subject(str(sbj)).experiment(str(sbj)+'_7T').resource(pkg).file('MNINonLinear/Results/{dir}/Movement_Regressors_dt.txt'.format(dir='rfMRI_'+run_id+'_7T_'+run_ap))
    motdt_path = osp.join(DATA_DIR,sbj,run,run+'_'+'Movement_Regressors_dt.txt')
    xnat_mot   = hcp1200.subject(str(sbj)).experiment(str(sbj)+'_7T').resource(pkg).file('MNINonLinear/Results/{dir}/Movement_Regressors.txt'.format(dir='rfMRI_'+run_id+'_7T_'+run_ap))
    mot_path   = osp.join(DATA_DIR,sbj,run,run+'_'+'Movement_Regressors.txt')
    
    for xnat,local in zip([xnat_mPP,xnat_SBref,xnat_motdt,xnat_mot],
                          [mPP_path,SBref_path,motdt_path,mot_path]):
        if not osp.exists(local):
            print('++ INFO [%d]: Downloading from XNAT %s' % (index,local))
            xnat.get(local)
        else:
            print('++ WARNING [%d]: File already in destination, no need to download or copy [%s]' % (index,local))
    print('++ --------')

# ***
# ## 6. Download Eye Tracker Data (two files per run)

for index,data in df.iterrows():
    sbj = data['Sbj']
    run = data['Run']
    _,run_id, run_ap = run.split('_')
    pkg = run + '_unproc'
    for file_suffix in ['eyetrack_summary.csv','eyetrack.asc']:
        xnat_et  = hcp1200.subject(str(sbj)).experiment(str(sbj)+'_7T').resource(pkg).file('LINKED_DATA/EYETRACKER/{sbj}_7T_{run}_{suff}'.format(sbj=sbj,run=run_id, suff=file_suffix))
        if xnat_et.exists():
            et_path = osp.join(DATA_DIR,sbj,run,run+'_'+file_suffix)
            xnat_et.get(et_path)
        else:
            print(' + WARNING --> File missing: %s' % xnat_et)

# ***
# # 7. Ensure all folders were created
#
# 1. Check that there is one directory for each subject
# 2. Check that there is one directory per available resting-state scan

XNAT_Subjects = list(SbjInfo['Subject'].values.astype(str))                     # list of all subject numbers
SbjDirs = glob.glob(DATA_DIR+'/??????')                                         # path to subject directories
if len(SbjDirs) == len(XNAT_Subjects):                                          # chekcs if there is same number of subjects as number of subject directories
    print('++ INFO: Number of Subject Directories = %d --> ALL SUBJECT DIRECTORIES EXIST.' % len(SbjDirs) )
else: 
    print('++ WARNING: Number of Subject Directories = %d --> Some directories are missing.' % len(SbjDirs) )

NXAT_NumberOfRestRuns = SbjInfo['7T_RS-fMRI_Count'].sum()                       # sum of totoal number of fMRI resting state runs
RunDirs               = glob.glob(DATA_DIR+'/??????/rfMRI_REST?_??')            # path to run directory 
if len(RunDirs) == NXAT_NumberOfRestRuns:                                       # check if there is the same number of totoal runs as there are run directories
    print('++ INFO: Number of Subject Directories = %d --> ALL RUN DIRECTORIES EXIST.' % len(RunDirs) )
else: 
    print('++ WARNING: Number of Subject Directories = %d --> Some run directories are missing.' % len(RunDirs) )

# ***
# # 8. Remove problematic subjects
#
# Subject 178647 has an incorrect forth ventricle mask in the Freesurfer automatic parcellation. For this reason this subject was removed from all analyses. To ensure this data is not used, the next cell checks if a directory for this subject exists, and if so it removes it from the DATA_DIR folder.
#
# <img src='./images/178647_Parcellation.jpg'>
#
# Following removal of this subject, the size of our starting dataset should be 719 resting-state scans distributed across 183 subjects.

# Remove this subject from the dataframe that contains information about downloaded resting-state scans
df = df[df['Sbj'].str.contains('178647')==False]
print("++ INFO: Final shape of df dataframe: %s" % str(df.shape))
# Remove the folder from the DATA_DIR folder
bad_sbj_folder = osp.join(DATA_DIR,'178647')
if osp.exists(bad_sbj_folder):
    shutil.rmtree(bad_sbj_folder)
    print("++ INFO: Folder for subject 178647 was removed")

# ***
# # 9. Create the final output of this notebook, a DF with the paths to all the files that have been downloaded

# +
# %%time
# data frame of run with path to each file created above
Project_Files_DF = pd.DataFrame(columns=['Sbj','Run',
                                         'T1 (MNI)','T2 (MNI)',
                                         'Brainmask (MNI)','GM Ribbon (MNI)',
                                         'FS Parcels (MNI)',
                                         'Rest_mPP (MNI)',
                                         'SBref (Orig)',
                                         'ET_ASC','ET_CSV',
                                         'Motion'])
#                                index=range(df.shape[0]))

for index,data in df.iterrows():
    sbj         = data['Sbj'] # subject number
    run         = data['Run'] # run number
    # path to all files created above
    sbj_dir     = osp.join(DATA_DIR,str(sbj))
    t1w_path    = osp.join(sbj_dir,'T1w_restore_brain.nii.gz')
    t2w_path    = osp.join(sbj_dir,'T2w_restore_brain.nii.gz')
    bm_path     = osp.join(sbj_dir,'brainmask_fs.nii.gz')
    rois_path   = osp.join(sbj_dir,'aparc.a2009s+aseg.nii.gz')
    rib_path    = osp.join(sbj_dir,'GM_Ribbon.nii.gz')
    et_asc_path = osp.join(DATA_DIR,sbj,run,run+'_eyetrack_summary.csv')
    et_csv_path = osp.join(DATA_DIR,sbj,run,run+'_eyetrack.asc')
    mot_path    = osp.join(DATA_DIR,sbj,run,run+'_'+'Movement_Regressors.txt')
    sbref_path  = osp.join(DATA_DIR,sbj,run,run+'_'+'SBRef.nii.gz')
    unproc_path = osp.join(DATA_DIR,sbj,run,run+'.nii.gz')
    fix_path    = osp.join(DATA_DIR,sbj,run,run+'_FIX.nii.gz')
    mPP_path    = osp.join(DATA_DIR,sbj,run,run+'_mPP.nii.gz')
    Project_Files_DF.loc[index,'Sbj'] = sbj
    Project_Files_DF.loc[index,'Run'] = run
    # append each path with the coresponding column for each run if path doesnt exixt put NaN
    for (label, aux_path) in zip(['T1 (MNI)','T2 (MNI)','Brainmask (MNI)', 'GM Ribbon (MNI)', 'FS Parcels (MNI)', 'Rest_mPP (MNI)', 'SBref (Orig)', 'ET_ASC',    'ET_CSV',    'Motion'],
                                 [t1w_path,  t2w_path,   bm_path,          rib_path,          rois_path,          mPP_path,         sbref_path,     et_asc_path, et_csv_path, mot_path]):
        Project_Files_DF.loc[index,label]  = (lambda path: path if osp.exists(path) else np.nan)(aux_path)

Project_Files_DF.head()    
# -

print("++ INFO: Final number of subjects: %d" % len(Project_Files_DF['Sbj'].unique()))
print('++ INFO: Number of missing files per file_type')
Project_Files_DF.isna().sum() # count for each column (i.e. file) how many empyt values there are (i.e. the file does not exist)

# Save information to disk
ProjectFiles_DF_Path = osp.join(SCRIPTS_DIR,'Resources','7T_ProjectFiles.pkl') # create path for data frame created in above cell
Project_Files_DF.to_pickle(ProjectFiles_DF_Path) # save data frame as pickle file
print("++ INFO: Information about downloaded files available at [%s]" % ProjectFiles_DF_Path)

# ***
# ***
# # ADDITIONAL CODE TO DOWNLOAD OTHER FILES (NOT USED IN THIS PAPER)
#
# ### Download or Copy the un-processed datasets (NOT USED IN THE PAPER)

hcp1200 = cbd.select.project(XNAT_PROJECT)
for index,data in df.iterrows():
    sbj = data['Sbj']
    run = data['Run']
    _,run_id, run_ap = run.split('_')
    pkg = run + '_unproc'
    prev_path = osp.join(DATA_DIR,sbj,'rfMRI_'+run_id+'_7T_'+run_ap,sbj+'_7T_rfMRI_'+run_id+'_'+run_ap+'.nii.gz')
    dest_path = osp.join(DATA_DIR,sbj,run,run+'.nii.gz')
    if not osp.exists(dest_path):
        if osp.exists(prev_path):
            print('++ INFO [%d]: Moving file from different local path %s' % (index,prev_path))
            os.system('mv {orig} {dest}'.format(orig=prev_path,dest=dest_path))
        else:
            print('++ INFO [%d]: Downloading from XNAT %s' % (index,dest_path))
            xnat_path = hcp1200.subject(str(sbj)).experiment(str(sbj)+'_7T').resource(pkg).file(sbj+'_7T_'+run+'.nii.gz')
            xnat_path.get(dest_path)
    else:
        print('++ WARNING [%d]: File already in destination, no need to download or copy [%s]' % (index,dest_path))

# ### Download or copy fix-clean data (NOT USED IN THE PAPER)

# Download or copy FIX rsfMRI runs
# ================================
LOCAL_DIR = '/data/SFIMJGC_HCP7T/PRJ_IndividualDifferences/PrcsData/'
hcp1200 = cbd.select.project(XNAT_PROJECT)
for index,data in df.iterrows():
    sbj = data['Sbj']
    run = data['Run']
    _,run_id, run_ap = run.split('_')
    pkg = run + '_FIX'
    prev_path = osp.join(LOCAL_DIR,sbj,'D00_OriginalData','rfMRI_'+run_id+'_7T_'+run_ap+'_hp2000_clean.nii.gz')
    dest_path = osp.join(DATA_DIR,sbj,run+'_mPP',run+'_FIX.nii.gz')
    if not osp.exists(dest_path):
        if osp.exists(prev_path):
            print('++ INFO [%d]: Moving file from different local path %s' % (index,prev_path))
            os.system('mv {orig} {dest}'.format(orig=prev_path,dest=dest_path))
        else:
            print('++ INFO [%d]: Downloading from XNAT %s' % (index,dest_path))
            xnat_path = hcp1200.subject(str(sbj)).experiment(str(sbj)+'_7T').resource(pkg).file('rfMRI_'+run_id+'_7T_'+run_ap+'/rfMRI_'+run_id+'_7T_'+run_ap+'_hp2000.nii.gz')
            xnat_path.get(dest_path)
    else:
        print('++ WARNING [%d]: File already in destination, no need to download or copy [%s]' % (index,dest_path))
