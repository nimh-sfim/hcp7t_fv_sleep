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
# This notebook completes the preliminary analyses looking to whether or not singals in the 4th ventricle follow an inflow-like profile of decreasing intensity across succesive slides. These analyses were conducted on a subset of subjects, as FV ROIs are not available in original space. This notebook performs the following operations.
#
# 1. Select a subset of scans on which we will focus our attention. We will select scans with low motion marked as drowsy.
# 2. Load the manually obtained location of the obex, dorsomedial recess and aqueduct of Sylvius.
# 3. Do a first pass at generating the ROIs in orig space in a pseduo-automatic manner. Those will need a lot of manual correction, but it provides us with some priors for drawing the ROIs.
#
# After manual correction of ROIs is performed, the notebook continues with the next steps:
#
# 4. Create the mean across all 20 scans (backgound in Figure 2.A and B)
# 5. Extract ROI-wise and slice-wise represenative timeseries for the FV
# 6. Compute slice timing informaiton (Figure 2.C)
# 7. Compute slice-wise 95th to 5ht percentile ratio (Figure 2.D - F)

# +
import os
import pandas as pd
import numpy  as np
import hvplot.pandas
import matplotlib.pyplot as plt
import os.path as osp
import holoviews as hv
import seaborn as sns
hv.extension('bokeh')
from scipy.stats import pearsonr
from utils.variables import DATA_DIR, Resources_Dir, PercentWins_EC_Awake, PercentWins_EC_Sleep
from utils.basics import get_available_runs
import panel as pn
import uuid
import random
from scipy.stats import ttest_ind, mannwhitneyu

# Extra imports for the 95th to 5th
from matplotlib.cm import get_cmap
from utils.gui import generate_cmap_hex
import matplotlib
from utils.basics import load_segments
import xarray as xr
import subprocess
import csv 
sns.set(font_scale=2)
# -
Nacq = 890

# ## 1. Load list of runs that make it to the main analyses

Manuscript_Runs = get_available_runs(when='final',type='all')
Awake_Runs      = get_available_runs(when='final',type='awake')
Drowsy_Runs     = get_available_runs(when='final',type='drowsy')
print('++ INFO: Number of Runs is %d' % len(Manuscript_Runs))


# ***
# ## 2. Load Framewise Displacement

# %%time 
# Load FD traces per run
fd = pd.DataFrame(index=np.arange(Nacq), columns=Manuscript_Runs)
for item in Manuscript_Runs:
    sbj,run = item.split('_',1)
    fd_path = osp.join(DATA_DIR,sbj,run,run+'_Movement_FD.txt')
    fd[item]= pd.read_csv(fd_path, sep='\t', index_col=0)
# Compute Mean FD per Run   
mean_fd = pd.DataFrame(fd.mean(), columns=['Mean_FD'])
# Add column with run type
mean_fd['Scan Type'] = 'N/A'
mean_fd.loc[(Awake_Runs,'Scan Type')]  = 'Awake'
mean_fd.loc[(Drowsy_Runs,'Scan Type')] = 'Drowsy'
mean_fd = mean_fd.reset_index(drop=True)

# ***
# ## 3. Select Scans based on ET and motion criteria

drowsy_mean_fd = fd[Drowsy_Runs].mean()
selected_scans_df = drowsy_mean_fd[drowsy_mean_fd<.1].sort_values()
print('++ INFO: Number of scans selected for preliminary inflow-profile analyses: %d scans' % selected_scans_df.shape[0])
selected_scans_df.head()

# ***
# ## 4. Record the location (slice number) for FV macroanatomical structures
#
# We used AFNI to visually explore the selected 30 scans and annotated the slice numbers for slices cutting through three key macro-anatomical landmarks for the definition of the FV, namely the Obex, the Dorsomedial Recess and the Aquduct of Sylvius. These annotations were saved into a csv file.

# +
macro_fv_path = osp.join(Resources_Dir,'hcp7t_inflow_scans_fv_macro.csv')
macro_fv_df   = pd.read_csv(macro_fv_path)
obex_location = int(np.floor(macro_fv_df['Obex'].median()))
dmr_location  = int(np.floor(macro_fv_df['Dorsomedial Recess'].median()))
aqs_location  = int(np.floor(macro_fv_df['Aqueduct'].median()))

print('++ INFO: Obex median location:         %s' % obex_location)
print('++ INFO: Dorsomedial recess location:  %s' % dmr_location)
print('++ INFO: Aqueduct of Sylvius location: %s' % aqs_location)
# -

macro_fv_df.hvplot.hist(normed=True).opts(xrotation=90, invert_xaxis=True, fontsize={'xticks':18,'legend':18, 'yticks':18, 'ylabel':18}, xlabel='', ylabel='Scan Count [Density]', legend_position='top', toolbar=None) * \
macro_fv_df.hvplot.kde().opts(xrotation=90, invert_xaxis=True) * \
hv.VLine(obex_location).opts(line_width=2, line_dash='dashed') * \
hv.VLine(dmr_location).opts(line_width=2, line_dash='dashed') * \
hv.VLine(aqs_location).opts(line_width=2, line_dash='dashed')

# Based on the observation of an outlier (in terms of the location of the FV), we drop scan ```782561_rfMRI_REST2_AP``` from the list of 30 selected scans.

selected_scans_df.drop('782561_rfMRI_REST2_AP', inplace=True)
selected_scans_list = list(selected_scans_df.index)

# Save the list of selected scans to disk ```Resources/Inflow_Profile_Selected_Scans.csv``` so that we can access it from other notebooks

selected_scans_path = osp.join(Resources_Dir,'Inflow_Profile_Selected_Scans.csv')
pd.Series(selected_scans_list).to_csv(selected_scans_path, index=False, header=False)

# ***
#
# ## 5. Automated generation of FV mask in orig space
#
# Although the masks will be ultimately generated by hand, we used this semi-automated first step to guide the drawing of the FV ROIs. In particular we look for voxels that 3dSeg identifies as CSF and which have a higher than normal standard deviation over time.

# + tags=[]
# Create log dir for swarm jobs to write their output and error messages
# ======================================================================
if not osp.exists('./N03c_Inflow_Profile_Analysis.logs'):
    print('++ INFO: Creating logging dir: N03c_Inflow_Profile_Analysis.logs')
    os.mkdir('./N03c_Inflow_Profile_Analysis.logs')
# -

# #### Create Swarm File

# Create Swarm file for extracting representative power
# ======================================================
os.system('echo "#swarm -f ./N03c_Inflow_Profile_Analysis.SCRIPT01_SWARM.sh -g 16 -t 16 --partition quick,norm --module afni --logdir ./N03c_Inflow_Profile_Analysis.logs" > ./N03c_Inflow_Profile_Analysis.SCRIPT01_SWARM.sh')
for item in selected_scans_list:
    sbj,run = item.split('_',1)
    os.system('echo "export SBJ={sbj} RUN={run}; sh ./N03c_Inflow_Profile_Analysis_SCRIPT01_AutomaticROICreation.sh" >> ./N03c_Inflow_Profile_Analysis.SCRIPT01_SWARM.sh'.format(sbj=sbj,run=run))

# Next we need to run the swarm jobs as follows:
#
# ```bash
# # cd /data/SFIMJGC_HCP7T/hcp7t_fv_sleep/Notebooks
# swarm -f ./N03c_Select_Scans_Inflow_Profile_Analysis.SWARM.sh -g 16 -t 16 --partition quick,norm --module afni --logdir ./N03c_Select_Scans_Inflow_Profile_Analysis.logs
#
# watch -n 30 squeue -u javiergc
# ```

# ***
#
# ## 5. Manual correction
#
# The automatically generated masks will contain many high stdev voxels outside the FV. For that reason it is necessary to manually correct these masks and ensure they are confined to the target structure of this study (i.e., the FV). The file ```<run_ID>_orig.mask.FV.mask.nii.gz``` were manually corrected with the AFNI software. 

# ***
# ## 6 Generate Mean Image across the MEAN of all selected scans (Figure 2.A)

import nibabel as nib

# + tags=[]
# %%time
# Get a list with all the MEAN Files
selected_scans_MEAN_files  = [osp.join(DATA_DIR,scan.split('_',1)[0],scan.split('_',1)[1],scan.split('_',1)[1]+'_orig.MEAN.nii.gz') for scan in selected_scans_list]
N_selected_scans           = len(selected_scans_MEAN_files)

# Compute Mean across MEAN Files (underlay)
mean_data = np.zeros((130,130,85))
for i, path in enumerate(selected_scans_MEAN_files):
    img = nib.load(path)
    img_data = img.get_fdata()
    mean_data = mean_data + img_data
mean_data = mean_data / N_selected_scans

mean_out_path = osp.join(DATA_DIR,'ALL','Inflow_SelectedScans_MEAN_Avg.nii.gz')
mean_img      = nib.Nifti1Image(mean_data,np.eye(4))
nib.save(mean_img,mean_out_path )
print('++ INFO: Average of MEANs will be saved in %s' % mean_out_path)

# + tags=[]
# %%time
# Get a list with all the MEAN Files
selected_scans_MASK_files  = [osp.join(DATA_DIR,scan.split('_',1)[0],scan.split('_',1)[1],scan.split('_',1)[1]+'_orig.mask.FV.manual.dil01.nii.gz') for scan in selected_scans_list]
N_selected_scans           = len(selected_scans_MASK_files)

# Compute Mean across MEAN Files (underlay)
mean_data = np.zeros((130,130,85))
for i, path in enumerate(selected_scans_MASK_files):
    img = nib.load(path)
    img_data = img.get_fdata()
    mean_data = mean_data + img_data
mean_data = mean_data / N_selected_scans

mean_out_path = osp.join(DATA_DIR,'ALL','Inflow_SelectedScans_FV.manual.dil01_Avg.nii.gz')
mean_img      = nib.Nifti1Image(mean_data,np.eye(4))
nib.save(mean_img,mean_out_path )
print('++ INFO: Average of MEANs will be saved in %s' % mean_out_path)

# + active=""
# %%time
# for in_file in selected_scans_MASK_files:
#     sbj      = in_file.split('/')[4]
#     run      = in_file.split('/')[5]
#     out_file = osp.join(DATA_DIR,'ALL','{sbj}_{run}_orig.mask.FV.manual.dil01.nii.gz'.format(sbj=sbj,run=run))
#     img      = nib.load(in_file)
#     img_data = img.get_fdata()
#     new_img  = nib.Nifti1Image(img_data,np.eye(4))
#     nib.save(new_img,out_file)
# -

# ***
#
# ## 7. Extract TS for whole ROI and slice-by-slice
#
# The following jobs will compute one representative time-series for the whole FV ROI, at thre different pre-processing steps:
#
# * After discarding the first 10 volumes of data: ```${RUN}_orig.discard.FV.mean.csv```
# * After slice-time correction: ```${RUN}_orig.tshift.FV.mean.csv```
# * After linear and quadratic detrend: ```${RUN}_orig.detrend.FV.mean.csv```
#
# It will also generate slice-by-slice representative time-series for the slices in the FV ROI for each scan. This also happens at the three pre-processing steps listed above:
#
# * Discarding: ```${RUN}_orig.discard.FV.k.csv```
# * Slice-time correction: ```${RUN}_orig.tshift.FV.k.csv```
# * Detrending: ```${RUN}_orig.detrend.FV.k.csv```

# Create Swarm file for extracting representative power
# ======================================================
os.system('echo "#swarm -f ./N03c_Inflow_Profile_Analysis.SCRIPT02_SWARM.sh -g 16 -t 16 --partition quick,norm --module afni --logdir ./N03c_Inflow_Profile_Analysis.logs" > ./N03c_Inflow_Profile_Analysis.SCRIPT02_SWARM.sh')
for item in selected_scans_list:
    sbj,run = item.split('_',1)
    os.system('echo "export SBJ={sbj} RUN={run}; sh ./N03c_Inflow_Profile_Analysis_SCRIPT02_PostManualCorrection.sh" >> ./N03c_Inflow_Profile_Analysis.SCRIPT02_SWARM.sh'.format(sbj=sbj,run=run))

# To run the jobs:
#
# ```bash
# # cd /data/SFIMJGC_HCP7T/hcp7t_fv_sleep/Notebooks
#
# swarm -f ./N03c_Inflow_Profile_Analysis.SCRIPT02_SWARM.sh -g 16 -t 16 --partition quick,norm --module afni --logdir ./N03c_Inflow_Profile_Analysis.logs
# ```

# + [markdown] tags=[]
# ***
# ## 6. Compute/Plot Slice Timing Profiles
# -

# Information we have about how the data was acquired
# ===================================================
N_slices        = 85
TR              = 1000
MB_factor       = 5
Increment_slice = 2
N_shots         = int(N_slices / MB_factor)
Intershot_timing = TR / N_shots
print("++ INFO: Number of Shots: %d" % N_shots)

shot_cmap      = generate_cmap_hex('Blues',N_shots)
shot_cmap_dict = {i:c for i,c in enumerate(shot_cmap)}

get_cmap('Blues',N_shots)

# DataFrame with inferred slice timing
# ====================================
slice_timing_pd               = pd.DataFrame(columns=['Slice Number','Shot','Slice Time'],index=np.arange(N_slices) )
shots                         = [list(np.repeat(s,MB_factor)) for s in np.arange(N_shots)]
shots                         = [item for sublist in shots for item in sublist]
slice_timing_pd['Shot']       = shots
slice_timing_pd['Slice Time'] = slice_timing_pd['Shot'] * Intershot_timing

slice_order = []
for inc in np.arange(Increment_slice):
    for shot in np.arange(N_shots):
        slices_per_shot = np.arange(inc+Increment_slice*shot,N_slices,N_shots)
        if len(slices_per_shot) == MB_factor:
            slice_order = slice_order + list(slices_per_shot)
slice_timing_pd['Slice Number'] = slice_order

slice_timing_pd.to_csv(osp.join(Resources_Dir,'HCP7T_SliceTimingTable.csv'))

# %matplotlib inline
sns.reset_orig()
matplotlib.rc('xtick', labelsize=16) 
matplotlib.rc('ytick', labelsize=16) 
fig, axs = plt.subplots(1,1,figsize=(18,7))
slice_timing_pd.plot.scatter(x='Slice Number',y='Slice Time', ax=axs, s=100, marker='s',color=[shot_cmap_dict[c] for c in slice_timing_pd['Shot']], edgecolor='k')
axs.set_xlabel('Slice Number', fontsize=16)
axs.set_ylabel('Slice Timing [ms]', fontsize=16)
for r,row in slice_timing_pd.iterrows():
    axs.annotate('S'+str(int(row['Slice Number'])).zfill(2),(row['Slice Number']-3,row['Slice Time']-12), fontsize=12)
for (bot_sl,top_sl) in [(1,15),(0,32),(17,49),(34,66),(51,83),(68,84)]:
    x0 = bot_sl
    x1 = top_sl
    y0 = slice_timing_pd[slice_timing_pd['Slice Number']==bot_sl]['Slice Time'].values[0]
    y1 = slice_timing_pd[slice_timing_pd['Slice Number']==top_sl]['Slice Time'].values[0]
    plt.annotate(s='', xy=(x1+3,y1), xytext=(x0+3,y0), arrowprops=dict(facecolor='black', lw=5, arrowstyle='->'))#dict(arrowstyle='->',width=3));
#axs.arrow(1+1,  slice_timing_pd[slice_timing_pd['Slice Number']==1]['Slice Time'].values[0],
#          15+1, slice_timing_pd[slice_timing_pd['Slice Number']==15]['Slice Time'].values[0]-slice_timing_pd[slice_timing_pd['Slice Number']==1]['Slice Time'].values[0],width=1)

# ***
# ## 7. Load Information about eye closure segments
#
# First, we load information about all the eyes closed segments in the 29 selected scans

ec_segments = load_segments('EC',selected_scans_list,min_dur=60)
ec_segments.set_index('Run', inplace=True, drop=True)

# Next, we extract the segment IDs for those segments

selected_segments_list = [s['Segment_UUID'] for _,s in ec_segments.iterrows() ]
selected_subjects_list = list(set([item.split('_',1)[0] for item in selected_scans_list]))

# Here are some basic counts regarding how the selected segments are distributed across different scans and subjects. This is important to keep in mind becuase more than one segment can occur per scan, and more than one scan can be associated with the same subject. We want to look at data from several subjects. This cell allows us to investigate if that was the case.

print('++ INFO: Number of selected subjects: %d' % len(selected_subjects_list))
print('++ INFO: Number of selected scans: %d' % len(selected_scans_list))
print('++ INFO: Number of selected segments: %d' % len(selected_segments_list))

# ***
# ## 8. Generate Figure 2.C and 2.D
#
# The next cell performs the following operations for scan segment:
#
# 1. Loads the slice-wise time-series for the associated scan
# 2. Limits the time-series to the EC segment of interest
# 3. Compute the 95h / 5th ratio

# As mentioned in the manuscript, we will restrict our analyses to slices 17 to 28 as those seem to be the ones sitting over the FV. The graph below allow us to check that we have data comming from most scans for these slices

per_slice_TS       = xr.DataArray(dims=['Segment','Slice','Time'], coords={'Segment':selected_segments_list,'Slice':np.arange(0,85),'Time':np.arange(0,890)})
per_slice_Qratio   = pd.DataFrame(index=np.arange(0,85), columns=selected_segments_list)
for s,s_row in ec_segments.iterrows():
    # Extract Segment onset, offset, ID and scan to which it belongs
    aux_onset  = int(s_row['Onset'])
    aux_offset = int(s_row['Offset'])
    aux_segID  = s_row['Segment_UUID']
    aux_scanID = s_row.name
    aux_sbj, aux_run = aux_scanID.split('_',1)
    # Load the slice-wise time-series (after discarding initial 10 volumes) for the appropriate scan.
    path       = osp.join(DATA_DIR,aux_sbj,aux_run,'{run}_orig.discard.FV.dil01.k.csv'.format(run=aux_run))
    aux        = pd.read_csv(path, sep='\t')
    aux        = aux.drop(['File','Sub-brick'], axis=1)
    aux.index.name = 'Time'
    aux.columns    = [int(c.split('_',1)[1]) for c in aux.columns ]
    # Remove any time points outside the EC period
    aux            = aux.loc[aux_onset:aux_offset]
    # Compute the 95th to 5th percentile ratio
    per_slice_TS.loc[aux_segID,aux.columns,aux.index] = aux.T
    per_slice_Qratio.loc[:,aux_segID] = aux.quantile(0.95,axis=0) / aux.quantile(0.05,axis=0)

# * Keep only the slices that we have identified as being inside the FV

per_slice_Qratio = per_slice_Qratio[17:29]
slice_timing_pd  = slice_timing_pd[slice_timing_pd['Slice Number'].isin(np.arange(17,29))].sort_values(by='Slice Number').set_index('Slice Number')

# * Compute averate ratio values across all segments

aux_mean = per_slice_Qratio.mean(axis=1)
aux_stdv = per_slice_Qratio.std(axis=1)
aux_ster = aux_stdv / np.sqrt(per_slice_Qratio.shape[1])

per_slice_Qratio_STATS = pd.concat([aux_mean,aux_stdv,aux_ster,slice_timing_pd['Shot']],axis=1)
per_slice_Qratio_STATS.columns    = ['Mean','Stdv','StErr','Shot']
per_slice_Qratio_STATS.index.name = 'Slice Number'

# * Plot results

inflow_directions = {0:[1,3,5,7,9,11,13,15],
                     1:[0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32],
                     2:[17,19,21,23,25,27,29,31,33,35,37]}

# +
# %matplotlib inline
import matplotlib 

matplotlib.rc('xtick', labelsize=16) 
matplotlib.rc('ytick', labelsize=16) 
fig,axs = plt.subplots(3,1,figsize=(10,20))

# All Slices Together
per_slice_Qratio_STATS.reset_index().plot.bar(x='Slice Number',y='Mean', 
                                                     ax=axs[0], width=.9 , 
                                                     color=[shot_cmap_dict[c] for c in per_slice_Qratio_STATS['Shot']],
                                                     edgecolor='k',
                                                     yerr='StErr', capsize=4)
axs[0].plot([7,7],[1,1.5],c='red',lw=2,linestyle='dashed')
axs[0].set_ylabel('95th to 5th Ratio', fontsize=16)
axs[0].set_xlabel('Slice Number', fontsize=16)
axs[0].set_ylim([1,1.5])
axs[0].get_legend().remove()

# Per Trajectory
for inflow_dir in [1,2]:
    aux = per_slice_Qratio_STATS.copy()
    for i in per_slice_Qratio_STATS.index:
        if not(i in inflow_directions[inflow_dir]):
            aux.loc[i,'Mean']  = np.NaN
            aux.loc[i,'Stdv']  = np.NaN
            aux.loc[i,'StErr'] = np.NaN
    
    # Plot
    aux['Slice Number'] = aux.index
    aux.plot.bar(x='Slice Number',y='Mean', ax=axs[inflow_dir], width=.9 , 
                 color=[shot_cmap_dict[c] for c in aux['Shot']],
                 edgecolor='k',yerr='StErr', capsize=4)
    axs[inflow_dir].plot([7,7],[1,1.5],c='red',lw=2,linestyle='dashed')
    axs[inflow_dir].set_ylabel('95th to 5th Ratio', fontsize=16)
    axs[inflow_dir].set_xlabel('Slice Number', fontsize=16)
    axs[inflow_dir].get_legend().remove()
    axs[inflow_dir].set_ylim([1,1.5])
    
plt.yticks(rotation = 90);
# -

# ***
# ***
# # END OF NOTEBOOK
# ***
# ***
