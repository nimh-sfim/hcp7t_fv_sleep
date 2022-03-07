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

# + [markdown] tags=[]
# ***
#
# # Statistical Analyses for FC using NBS
#
# To find statistical differences in FC between scan types, we will rely on the NBS software (https://www.nitrc.org/projects/nbs). 
#
# Results from those analyses will then be plotted as 3D network graphs using BrainNetViewer (https://www.nitrc.org/projects/bnv/)
#
# NBS needs FC matrices in a very particular format (and organized in a particular way). In addition, it needs desigh matrices describing which matrices belong to each group. The following cells will generate all necessary files
#
# ### Generate Design Matrices for NBS 
#
# Here we generate 3 matrices, one per scan gropings. As mentioned above, we only report resutls for scan groups based on ET data. The other two resutls were for explorative purposes.
# -

# Create the design matrices
# ==========================
ET_DMatrix  = np.vstack([np.tile(np.array([0,1]),(len(ET_Groups['Awake']),1)),np.tile(np.array([1,0]),(len(ET_Groups['Drowsy']),1))])
np.savetxt(osp.join(Resources_Dir,'NBS_ET_DesingMatrix.txt'),ET_DMatrix,delimiter=' ',fmt='%d')
GS_DMatrix  = np.vstack([np.tile(np.array([0,1]),(len(GS_Groups['Awake']),1)),np.tile(np.array([1,0]),(len(GS_Groups['Drowsy']),1))])
np.savetxt(osp.join(Resources_Dir,'NBS_GS_DesingMatrix.txt'),GS_DMatrix,delimiter=' ',fmt='%d')
PSD_DMatrix = np.vstack([np.tile(np.array([0,1]),(len(PSD_Groups['Awake']),1)),np.tile(np.array([1,0]),(len(PSD_Groups['Drowsy']),1))])
np.savetxt(osp.join(Resources_Dir,'NBS_PSD_DesingMatrix.txt'),PSD_DMatrix,delimiter=' ',fmt='%d')

# ### Make copies of FC matrices on Resources folder that follow NBS requirements for data organization
#
# Here we will:
#
# * Generate individual folders for each pre-processing pipeline and group selection method --> We will have a total of 12 folders (4 pre-processing pipelines X 3 scan groupings)
#
# * Inside each folder, we will make copies of the connectivity matrices. Those copied will be named simply as subject???.txt --> This allows automatic loading of files in NBS

# %%time
# Create files with Z-scored connectivity matrices for their use in NBS
# =====================================================================
for suffix in ['BASIC', 'Behzadi_COMPCOR', 'AFNI_COMPCOR', 'AFNI_COMPCORp']:
    cc_matrix_xr = xr.DataArray(dims=['Run','ROI_x','ROI_y'], coords={'Run':Manuscript_Runs,'ROI_x':roi_info_df['ROI_Name'],'ROI_y':roi_info_df['ROI_Name']})
    # Load all the connectivity matrices for all subjects
    for i,item in enumerate(Manuscript_Runs):
        sbj,run  = item.split('_',1)
        path     = osp.join(DATA_DIR,sbj,run,'{run}_{suffix}.Shaeffer2018_200Parcels_000.netcc'.format(run=run,suffix=suffix))
        aux_cc_r = pd.read_csv(path,sep='\t',comment='#', header=1)
        # Convert R to Fisher Z-score as we will be computing the mean in the next cell
        aux_cc_Z = aux_cc_r.apply(np.arctanh)
        np.fill_diagonal(aux_cc_Z.values,1)
        cc_matrix_xr.loc[item,:,:] = aux_cc_Z
    # Save files to disk
    for sbj_lists,target_dir_prefix in zip([ET_Groups,    PSD_Groups,    GS_Groups],
                                           ['NBS_ET_Data','NBS_PSD_Data','NBS_GS_Data']):
        target_dir = osp.join(Resources_Dir,target_dir_prefix+'_'+suffix)
        if osp.exists(target_dir):
            rmtree(target_dir)
        os.mkdir(target_dir)
        print("++ INFO: Working on %s" % target_dir)
        for r,item in enumerate(sbj_lists['Awake']):
            sbj,run   = item.split('_',1)
            dest_path = osp.join(Resources_Dir,target_dir,'subject{id}.txt'.format(id=str(r+1).zfill(3)))
            np.savetxt(dest_path,cc_matrix_xr.loc[item,:,:],delimiter=' ',fmt='%f')
        for s,item in enumerate(sbj_lists['Drowsy']):
            sbj,run   = item.split('_',1)
            dest_path = osp.join(Resources_Dir,target_dir,'subject{id}.txt'.format(id=str(r+1+s+1).zfill(3)))    
            np.savetxt(dest_path,cc_matrix_xr.loc[item,:,:],delimiter=' ',fmt='%f')
        del r,s,item
    del cc_matrix_xr

# ### Run Statistical Analyses in MATLAB / NBS
#
# These analyses were conducted using NBS v1.2 on MATLAB 2019a. 
#
# To run the analysis, do the following:
#
# 1. Connect to a spersist node via NoMachine or VNC
#
# 2. Open a terminal and enter the project folder
#
# ```bash
# # cd /data/SFIMJGC_HCP7T/hcp7T_fv_sleep/
# ```
#
# 3. Load the matlab module and start matlab
#
# ```bash
# module load matlab/2019a
# matlab
# ```
#
# 4. Add NBS to the MATLAB path.
#
# ```matlab
# addpath(genpath('/data/SFIMJGC_HCP7T/hcp7t_fv_sleep_extraSW/NBS1.2/'))
# ```
#
# 5. Start NBS
#
# ```matlab
# NBS
# ```
#
# 6. Configure NBS for a particular scenario
#
#     See detailed instructions for each case on the following cells
#     
# 7. Save the results to disk using the File --> Save Current

# #### ET-based Groups, Behzadi_COMPCOR, Awake > Drowsy
#
# * Design Matrix: ```/data/SFIMJGC_HCP7T/hcp7t_fv_sleep/Resources/NBS_ET_DesingMatrix.txt```
# * Constrast: [-1,1]
# * Statistical Test: T-test
# * Threshold: 3.1
# * Connectivity Matrices: ```/data/SFIMJGC_HCP7T/hcp7t_fv_sleep/Resources/NBS_ET_Data_Behzadi/COMPCOR/subject0001.txt```
# * Node Coordinates: ```/data/SFIMJGC_HCP7T/hcp7t_fv_sleep/Resources/NBS_Node_Coordinates.txt```
# * Node LAbesl: ```/data/SFIMJGC_HCP7T/hcp7t_fv_sleep/Resources/NBS_Node_Labels.txt```
# * Exchange Blocks: EMPTY
# * Permutations: 5000
# * Significance: 0.05
# * Method: Network-Based Statistics
# * Component Size: Extent
#
# Once analyses are completed, please save as ```/data/SFIMJGC_HCP7T/hcp7t_fv_sleep/Resources/NBS_ET_Results/NBS_ET_Behzadi_COMPCOR_AgtD.mat```
# ![](./images/NBS_ET_Bezhadi_COMCOR_AgtD.configuration.png)

# #### ET-based Groups, Behzadi_COMPCOR, Drowsy > Awake
#
# * Design Matrix: ```/data/SFIMJGC_HCP7T/hcp7t_fv_sleep/Resources/NBS_ET_DesingMatrix.txt```
# * Constrast: [1,-1]
# * Statistical Test: T-test
# * Threshold: 3.1
# * Connectivity Matrices: ```/data/SFIMJGC_HCP7T/hcp7t_fv_sleep/Resources/NBS_ET_Data_Behzadi/COMPCOR/subject0001.txt```
# * Node Coordinates: LEAVE EMPTY TO AVOID MEMORY ISSUES DURING FINAL PLOTTING
# * Node LAbesl: ```/data/SFIMJGC_HCP7T/hcp7t_fv_sleep/Resources/NBS_Node_Labels.txt```
# * Exchange Blocks: EMPTY
# * Permutations: 5000
# * Significance: 0.05
# * Method: Network-Based Statistics
# * Component Size: Extent
#
# Once analyses are completed, please save as ```/data/SFIMJGC_HCP7T/hcp7t_fv_sleep/Resources/NBS_ET_Results/NBS_ET_Behzadi_COMPCOR_DgtA.mat```
# ![](./images/NBS_ET_Bezhadi_COMCOR_DgtA.configuration.png)

# Similarly, equivalent results can be generated for the other pre-processing scenarios, by selecteding the corresponding folder in the "Connectivity Matrices" field. 

# ***
# # Draw Statistically Significant Differences in Connectivity using BrainNetViewer
#
# ### Install BrainNetViewer
#
# For this step, we will use the BrainNetViewer (https://www.nitrc.org/projects/bnv/) software that runs on MATLAB. 
#
# 1. Connect to biowulf spersist node using NoMachine or VNC
#
# 2. Download the MATLAB version of BrainNetViewer (Version 1.7 Release 20191031) into /data/SFIMJGC_HCP7T/hcp7t_fv_sleep_extraSW/
#
# 3. Unzip the downloaded file in /data/SFIMJGC_HCP7T/hcp7t_fv_sleep_extraSW/BrainNetViewer
#
# ```bash
# # cd /data/SFIMJGC_HCP7T/hcp7t_fv_sleep_extraSW/
# # mkdir BrainNetViewer
# # mv ~/Downloads/BrainNetViewer_20191031.zip /data/SFIMJGC_HCP7T/hcp7t_fv_sleep_extraSW/BrainNetViewer
# # cd /data/SFIMJGC_HCP7T/hcp7t_fv_sleep_extraSW/BrainNetViewer
# unzip BrainNetViewer_20191031.zip
# # rm BrainNetViewer_20191031.zip
# ```
#
# 4. Start MATLAB
#
# ```bash
# # cd /data/SFIMJGC_HCP7T/hcp7t_fv_sleep
# module load matlab/2019a
# matlab
# ```
#
# 5. Add the path to BrainNetViewer in MATLAB's command window
#
# ```matlab
# addpath('/data/SFIMJGC_HCP7T/hcp7t_fv_sleep_extraSW/BrainNetViewer')
# ```
#
# 6. Start BrainNetViewer
#
# ```matlab
# BrainNet
# ```
#

# ### Convert NBS Ouputs to BrainNetViewer format
#
# We have generated a small MATLAB script that will take as input the saved results from a given NBS analysis and will write out an ```.edge``` file to be loaded into BrainNetViewer.
#
# This script will also print to the screen the number of signficantly different connections for each scenario. We use that information when composing the manuscript figure.
#
# 1. Start MATLAB
#
# ```bash
# # cd /data/SFIMJGC_HCP7T/hcp7t_fv_sleep
# module load matlab/2019a
# matlab
# ```
#
# 2. Enter the Notbook folder on the MATLAB console
#
# ```matlab
# # cd /data/SFIMJGC_HCP7T/hcp7t_fv_sleep/Notebooks
# ```
#
# 3. Run ```N12_FCMatrix_NBS2BrainViewer.m``` on the MATLAB console
#
# ```matlab
# N12_FCMatrix_NBS2BrainViewer.m
# ```

# ### Plot Results
#
# 1. Start MATLAB
#
# ```bash
# # cd /data/SFIMJGC_HCP7T/hcp7t_fv_sleep
# module load matlab/2019a
# matlab
# ```
#
# 2. Add BrainNetViewer to the path
#
# ```matlab
# addpath('/data/SFIMJGC_HCP7T/hcp7t_fv_sleep_extraSW/BrainNetViewer')
# ```
#
# 3. Start BrainNetViewer
#
# ```matlab
# BrainNet
# ```
#
# 4. Select "File --> Load File"
#
#     * Surface File: ```/data/SFIMJGC_HCP7T/hcp7t_fv_sleep_extraSW/BrainNetViewer/Data/SurfTemplate/BrainMesh_ICBM152.nv```
#     * Data File (nodes): ```/data/SFIMJGC_HCP7T/hcp7t_fv_sleep/Resources/BrainNet_Nodes.node```
#     * Data File (edges): ```/data/SFIMJGC_HCP7T/hcp7t_fv_sleep/Resources/NBS_ET_Results/NBS_ET_Behzadi_COMPCOR_AgtD.edge``` (This one will vary depending on which results you want to plot)
#     * Mapping: Leave Empty
#     
# 5. Press OK
#
# 6. On the BrainNet_option dialog that just opened, click Load and select ```/data/SFIMJGC_HCP7T/hcp7t_fv_sleep/Resources/NBS_ET_Results/BrainNet_Options_SchaeferColors.mat```
#
# 7. Click Apply
#
# This should result in a figure similar to this (depending on which data you are loading)
#
# ![](./images/NBS_SampleResult.png)
