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
from utils.variables import SCRIPTS_DIR, Resources_Dir, DATA_DIR, Resources_NBS_Dir
from utils.basics import get_available_runs

from shutil import rmtree
from random import sample
hv.extension('bokeh')
# -

# # Load Run Lists (all, drowsy, awake)

# +
# %%time
Manuscript_Runs,Awake_Runs,Drowsy_Runs = {},{},{}
scan_HR_info             = pd.read_csv(osp.join(Resources_Dir,'HR_scaninfo.csv'), index_col=0)
scan_HR_info             = scan_HR_info[(scan_HR_info['HR_aliased']< 0.03) | (scan_HR_info['HR_aliased']> 0.07)]
Manuscript_Runs['noHRa'] = list(scan_HR_info.index)
Awake_Runs['noHRa']      = list(scan_HR_info[scan_HR_info['Scan Type']=='Awake'].index)
Drowsy_Runs['noHRa']     = list(scan_HR_info[scan_HR_info['Scan Type']=='Drowsy'].index)

Manuscript_Runs['all'] = get_available_runs(when='final', type='all')
Awake_Runs['all']      = get_available_runs(when='final', type='awake')
Drowsy_Runs['all']     = get_available_runs(when='final', type='drowsy')
print('++ INFO [All]:   Number of Runs: Total = %d | Awake = %d | Drowsy = %d' % (len(Manuscript_Runs['all']), len(Awake_Runs['all']), len(Drowsy_Runs['all'])))
print('++ INFO [noHRa]: Number of Runs: Total = %d | Awake = %d | Drowsy = %d' % (len(Manuscript_Runs['noHRa']), len(Awake_Runs['noHRa']), len(Drowsy_Runs['noHRa'])))
# -

# ***
# # Load Run Group Information
#
# In the manuscript we will only report results for run groupings based on Eye Tracker data. For exploration purposes, we also generated FC matrices when scans have been separated according to their ranking based on PSDsleep or GS. Those alternative groupings were generated in Notebook N11

# +
ET_Groups  = {'noHRa':{'Awake':Awake_Runs['noHRa'],'Drowsy':Drowsy_Runs['noHRa']},
              'all':  {'Awake':Awake_Runs['all'],'Drowsy':Drowsy_Runs['all']}}
              
GS_Groups  = {'noHRa':{'Drowsy':np.loadtxt(osp.join(Resources_Dir,'Run_List_GS_Top100.V4lt_grp.noHRa.txt'),dtype=str),
                       'Awake':np.loadtxt(osp.join(Resources_Dir,'Run_List_GS_Bot100.V4lt_grp.noHRa.txt'),dtype=str)},
              'all':  {'Drowsy':np.loadtxt(osp.join(Resources_Dir,'Run_List_GS_Top100.V4lt_grp.all.txt'),dtype=str),
                       'Awake':np.loadtxt(osp.join(Resources_Dir,'Run_List_GS_Bot100.V4lt_grp.all.txt'),dtype=str)}}
              
PSD_Groups = {'noHRa':{'Drowsy':np.loadtxt(osp.join(Resources_Dir,'Run_List_PSD_Top100.V4lt_grp.noHRa.txt'),dtype=str),
                       'Awake':np.loadtxt(osp.join(Resources_Dir,'Run_List_PSD_Bot100.V4lt_grp.noHRa.txt'),dtype=str)},
              'all':  {'Drowsy':np.loadtxt(osp.join(Resources_Dir,'Run_List_PSD_Top100.V4lt_grp.all.txt'),dtype=str),
                       'Awake':np.loadtxt(osp.join(Resources_Dir,'Run_List_PSD_Bot100.V4lt_grp.all.txt'),dtype=str)}}

#PSD_Groups = {'Drowsy': [s for s in PSD_Groups['Drowsy'] if s in ET_Groups['Drowsy']],
#              'Awake': [s for s in PSD_Groups['Awake'] if s in ET_Groups['Awake']]}

print('++ [All] ET-Based Grouping     : Awake = %d, Drowsy = %d' %(len(ET_Groups['all']['Awake']),len(ET_Groups['all']['Drowsy'])))
print('++ [noHRa] ET-Based Grouping   : Awake = %d, Drowsy = %d' %(len(ET_Groups['noHRa']['Awake']),len(ET_Groups['noHRa']['Drowsy'])))
print('++')
print('++ [All] GS-Based Grouping     : Awake = %d, Drowsy = %d' %(len(GS_Groups['all']['Awake']),len(GS_Groups['all']['Drowsy'])))
print('++ [noHRa] GS-Based Grouping   : Awake = %d, Drowsy = %d' %(len(GS_Groups['noHRa']['Awake']),len(GS_Groups['noHRa']['Drowsy'])))
print('++')
print('++ [All] PSD-Based Grouping    : Awake = %d, Drowsy = %d' %(len(PSD_Groups['all']['Awake']),len(PSD_Groups['all']['Drowsy'])))
print('++ [noHRa] PSD-Based Grouping  : Awake = %d, Drowsy = %d' %(len(PSD_Groups['noHRa']['Awake']),len(PSD_Groups['noHRa']['Drowsy'])))
# -

# # Generate FC Matrices
#
# ### Load ROI Names, and create labels and locations for matrix display

# +
# Load Info in the label table file created in N04b
roi_info_path = '/data/SFIMJGC_HCP7T/HCP7T/ALL/Schaefer2018_200Parcels_7Networks_order_mPP.GM_Ribbon_order.txt'
roi_info_df   = pd.read_csv(roi_info_path,header=None,index_col=0, sep='\t')
# We drop the last column, as I have no idea what it means and seems to contains zeros
roi_info_df.drop([5],axis=1, inplace=True)
# Add an extra column with the index (comes handy when plotting later)
roi_info_df.reset_index(inplace=True)
# We rename the columns
roi_info_df.columns = ['ROI_ID','ROI_Name','R','G','B']
# We add a new column with informaiton about network membership for each ROI
roi_info_df['Network']      = [item.split('_')[2] for item in roi_info_df['ROI_Name']]
roi_info_df['Network_Hemi'] = [item.split('_')[1] for item in roi_info_df['ROI_Name']]
roi_info_df['Node_ID']      = [item.split('_',1)[1] for item in roi_info_df['ROI_Name']]
# Transform the colors original provided by Shaefer (as a tuple) into HEX code that HVPLOT likes
cols = []
for row in roi_info_df.itertuples():
    cols.append('#%02x%02x%02x' % (row.R, row.G, row.B))
roi_info_df['RGB']=cols
# Remove the original color columns (as those are redundant and not useful)
roi_info_df.drop(['R','G','B'],axis=1,inplace=True)

print('++ INFO: Number of ROIs according to label file = %d' % roi_info_df.shape[0])
roi_info_df.head()
# -

Nrois = roi_info_df.shape[0]
print('++ INFO: Number of ROIs: %d' % Nrois)

# ## Create Design Matrix for NBS

# Create the design matrices
# ==========================
for scan_selection in ['all','noHRa']:
    ET_DMatrix      = np.vstack([np.tile(np.array([0,1]),(len(ET_Groups[scan_selection]['Awake']),1)),np.tile(np.array([1,0]),(len(ET_Groups[scan_selection]['Drowsy']),1))])
    ET_DMatrix_path = osp.join(Resources_NBS_Dir,'NBS_ET_DesingMatrix.{ss}.txt'.format(ss=scan_selection))
    np.savetxt(ET_DMatrix_path,ET_DMatrix,delimiter=' ',fmt='%d')
    print('++ INFO: File saved [%s]:' % ET_DMatrix_path)
    
    GS_DMatrix      = np.vstack([np.tile(np.array([0,1]),(len(GS_Groups[scan_selection]['Awake']),1)),np.tile(np.array([1,0]),(len(GS_Groups[scan_selection]['Drowsy']),1))])
    GS_DMatrix_path = osp.join(Resources_NBS_Dir,'NBS_GS_DesingMatrix.{ss}.txt'.format(ss=scan_selection))
    np.savetxt(GS_DMatrix_path,GS_DMatrix,delimiter=' ',fmt='%d')
    print('++ INFO: File saved [%s]:' % GS_DMatrix_path)
    
    PSD_DMatrix      = np.vstack([np.tile(np.array([0,1]),(len(PSD_Groups[scan_selection]['Awake']),1)),np.tile(np.array([1,0]),(len(PSD_Groups[scan_selection]['Drowsy']),1))])
    PSD_DMatrix_path =  osp.join(Resources_NBS_Dir,'NBS_PSD_DesingMatrix.{ss}.txt'.format(ss=scan_selection))
    np.savetxt(PSD_DMatrix_path,PSD_DMatrix,delimiter=' ',fmt='%d')
    print('++ INFO: File saved [%s]:' % PSD_DMatrix_path)

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
for suffix in ['Reference', 'GSR','BASIC', 'BASICpp', 'Behzadi_COMPCOR', 'Behzadi_COMPCORpp']:
    for scan_selection in ['all','noHRa']:
        cc_matrix_xr = xr.DataArray(dims=['Run','ROI_x','ROI_y'], coords={'Run':Manuscript_Runs[scan_selection],'ROI_x':roi_info_df['ROI_Name'],'ROI_y':roi_info_df['ROI_Name']})
        # Load all the connectivity matrices for all subjects
        for i,item in enumerate(Manuscript_Runs[scan_selection]):
            sbj,run  = item.split('_',1)
            path     = osp.join(DATA_DIR,sbj,run,'{run}_{suffix}.Shaeffer2018_200Parcels_000.netcc'.format(run=run,suffix=suffix))
            aux_cc_r = pd.read_csv(path,sep='\t',comment='#', header=1)
            # Convert R to Fisher Z-score as we will be computing the mean in the next cell
            aux_cc_Z = aux_cc_r.apply(np.arctanh)
            np.fill_diagonal(aux_cc_Z.values,1)
            cc_matrix_xr.loc[item,:,:] = aux_cc_Z
        # Save files to disk
        for sbj_lists,target_dir_prefix in zip([ET_Groups[scan_selection],    PSD_Groups[scan_selection],    GS_Groups[scan_selection]],
                                           ['NBS_ET_Data','NBS_PSD_Data','NBS_GS_Data']):
            target_dir = osp.join(Resources_NBS_Dir,target_dir_prefix+'_'+suffix+'_'+scan_selection)
            if osp.exists(target_dir):
                rmtree(target_dir)
            os.mkdir(target_dir)
            print("++ INFO: Working on %s" % target_dir)
            for r,item in enumerate(sbj_lists['Awake']):
                sbj,run   = item.split('_',1)
                dest_path = osp.join(Resources_NBS_Dir,target_dir,'subject{id}.txt'.format(id=str(r+1).zfill(3)))
                np.savetxt(dest_path,cc_matrix_xr.loc[item,:,:],delimiter=' ',fmt='%f')
            for s,item in enumerate(sbj_lists['Drowsy']):
                sbj,run   = item.split('_',1)
                dest_path = osp.join(Resources_NBS_Dir,target_dir,'subject{id}.txt'.format(id=str(r+1+s+1).zfill(3)))    
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
