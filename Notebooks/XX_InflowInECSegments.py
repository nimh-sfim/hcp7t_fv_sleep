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

# # Select Scans based on max Motion and max PSDsleep

from utils.basics import load_motion_FD, get_available_runs, load_segments, load_PSD, get_time_index, get_window_index
from utils.variables import DATA_DIR, Resources_Dir
import subprocess
import os.path as osp
import os
import csv
import pandas as pd
import numpy as np
import xarray as xr
import hvplot.pandas
import hvplot.xarray
import holoviews as hv
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.colors import rgb2hex


def generate_sleep_psd(f,t,Sxx,SleepBand_BotFreq=0.03,SleepBand_TopFreq=0.07):
    # Put Spectrogram into pandas dataframe (tidy-data form)
    df = pd.DataFrame(Sxx.T, columns=f, index=t)
    df = pd.DataFrame(df.stack()).reset_index()
    df.columns=['TR','Freq','PSD']
    
    # Extract average timeseries of power at different bands
    sleep_band_power     = df[(df['Freq'] < SleepBand_TopFreq) & (df['Freq'] > SleepBand_BotFreq)].groupby('TR').mean()['PSD'].values
    non_sleep_band_power = df[(df['Freq'] > SleepBand_TopFreq) | (df['Freq'] < SleepBand_BotFreq)].groupby('TR').mean()['PSD'].values
    control_band_power   = df[(df['Freq'] > 0.1)               & (df['Freq'] < 0.2)].groupby('TR').mean()['PSD'].values
    total_power          = df.groupby('TR').mean()['PSD'].values
    
    # Create output dataframe, which will contain both the average power traces computed above,
    # as well as the ratio of power in sleep band to all other "control" conditions
    df3                   = pd.DataFrame(columns=['sleep','non_sleep','total','control','ratio_w_total','ratio_w_non_sleep','ratio_w_control'])
    df3['sleep']          = sleep_band_power
    df3['non_sleep']      = non_sleep_band_power
    df3['total']          = total_power
    df3['control']        = control_band_power

    df3['ratio_w_total']     = sleep_band_power/total_power
    df3['ratio_w_non_sleep'] = sleep_band_power/non_sleep_band_power
    df3['ratio_w_control']   = sleep_band_power/control_band_power
    # Set the time index for the output dataframe
    df3.index = t
    return df3


def generate_cmap_hex(cmap_name,num_colors):
    cmap     = get_cmap(cmap_name,num_colors)
    cmap_hex = []
    for i in range(cmap.N):
        rgba = cmap(i)
        cmap_hex.append(rgb2hex(rgba))
    return cmap_hex


Nacqs = 890
TR    = 1    # seconds
win_dur = 60 # seconds

# #### Create Time- and Window-based indexes so we can plot things concurrently

time_index = get_time_index(Nacqs,TR)
win_index  = get_window_index(Nacqs,TR, win_dur)

# #### Load list of scans classified as drowsy

Drowsy_Runs = get_available_runs(when='final', type='drowsy')

# #### Load Framewise Displacement (motion) traces for the drowsy scans

motFD       = load_motion_FD(Drowsy_Runs, index=time_index)
motFD.index = motFD.index.total_seconds()

# #### Load PSDsleep for the 4th Ventricle (whole-roi) for Drowsy scans

PSDsleep       = load_PSD(Drowsy_Runs,band='sleep', index=win_index)
PSDsleep.index = PSDsleep.index.total_seconds()

# #### Load information about long periods of Eye Closure ( > 60 seconds) for all drowsy scans

EC_segments = load_segments('EC',Drowsy_Runs,60)

# #### Plot Motion vs. PSDsleep for all selected periods of Eye Closure

for r,row in EC_segments.iterrows():
    aux_onset, aux_offset, aux_run = int(row['Onset']), int(row['Offset']), row['Run']
    aux = motFD.loc[aux_onset:aux_offset,aux_run]
    EC_segments.loc[r,'FD_mean'] = aux.mean()
    EC_segments.loc[r,'FD_max'] = aux.max()
    aux = PSDsleep.loc[aux_onset:aux_offset,aux_run]
    EC_segments.loc[r,'PSDs_mean'] = aux.mean()
    EC_segments.loc[r,'PSDs_max'] = aux.max()

EC_segments.hvplot.scatter(x='FD_max',y='PSDs_max', hover_cols=['Run','Onset','Offset']) * \
hv.VLine(0.3).opts(line_width=1, line_dash='dashed', line_color='k') * \
hv.HLine(50).opts(line_width=1, line_dash='dashed', line_color='k')

# #### Select a subset of segments with low motion (FD<=0.3) and high PSD (Max[PSDsleep]>=50)

selected_segments = EC_segments[(EC_segments['FD_max']<=0.3) & (EC_segments['PSDs_max']>=50)]
selected_scans    = list(selected_segments['Run'].unique())
selected_subjects = list(set([s.split('_',1)[0] for s in selected_scans]))
print('++ ===============================================================')
print('++ INFO: Number of selected EC segments: %d' % len(selected_segments))
print('++ INFO: Number of runs containing the selected segments: %d' % len(selected_scans))
print('++ INFO: Number of subjects containing the selected segments: %d' % len(selected_subjects))

selected_scans_csv_path = osp.join(Resources_Dir,'XX_EC_selected_scans.csv')
np.savetxt(selected_scans_csv_path,selected_scans, fmt='%s')

selected_segments_csv_path = osp.join(Resources_Dir,'XX_EC_selected_segments.csv')
selected_segments.to_csv(selected_segments_csv_path)

print(selected_scans)

# ***
# # Download the original data (done somewhere else as it needs a different conda environment for pyxnat)

selected_scans = ['115017_rfMRI_REST1_PA', '115017_rfMRI_REST3_PA', '115825_rfMRI_REST1_PA', '134627_rfMRI_REST1_PA', '134627_rfMRI_REST3_PA', 
                  '146432_rfMRI_REST1_PA', '146432_rfMRI_REST2_AP', '175237_rfMRI_REST2_AP', '581450_rfMRI_REST3_PA', '581450_rfMRI_REST4_AP', 
                  '782561_rfMRI_REST2_AP', '899885_rfMRI_REST3_PA', '943862_rfMRI_REST1_PA', '943862_rfMRI_REST3_PA', '966975_rfMRI_REST1_PA', 
                  '966975_rfMRI_REST3_PA', '966975_rfMRI_REST4_AP']

import pyxnat
import os.path as osp
import os
import subprocess
from utils.variables import DATA_DIR

cbd = pyxnat.Interface('https://db.humanconnectome.org','javiergcas','Sp1n#Ech0')
XNAT_PROJECT = 'HCP_1200'

hcp1200 = cbd.select.project(XNAT_PROJECT)

# %%time
for scan in selected_scans:
    sbj,run = scan.split('_',1)
    _,run_id, run_ap = run.split('_')
    pkg              = run + '_unproc'
    dest_path        = osp.join(DATA_DIR,sbj,run,run+'_orig.nii.gz')
    print('++ INFO: Downloading from XNAT %s' % (dest_path))
    xnat_path = hcp1200.subject(str(sbj)).experiment(str(sbj)+'_7T').resource(pkg).file(sbj+'_7T_'+run+'.nii.gz')
    xnat_path.get(dest_path)

# ***
# # Create automatic version of the FV ROI in original space

# + tags=[]
# %%time
for scan in selected_scans:
    print(scan)
    sbj,run = scan.split('_',1)
    work_folder = osp.join(DATA_DIR,sbj,run)
    command="module load afni; \
             cd {wd}; \
             pwd; \
             3dTstat -overwrite -mean -prefix {run}_orig.MEAN.nii.gz {run}_orig.nii.gz; \
             if [ -d Segsy ]; then rmdir Segsy; fi; \
             3dSeg -anat {run}_orig.MEAN.nii.gz -mask AUTO -classes 'CSF ; GM ; WM' -bias_classes 'GM ; WM' -bias_fwhm 25 -mixfrac UNI -main_N 5 -blur_meth BFT; \
             3dcalc -overwrite -a Segsy/Classes+orig. -expr 'equals(a,3)*within(k,15,30)*within(i,55,70)*within(j,30,50)' -prefix {run}_orig.FVmask.auto.nii.gz; \
             3dClusterize -overwrite -nosum -1Dformat -inset {run}_orig.FVmask.auto.nii.gz \
                          -idat 0 -ithr 0 -NN 2 -clust_nvox 50 -bisided -0.0001 0.0001 \
                          -pref_map {run}_orig.FVmask.auto.clust.nii.gz".format(wd=work_folder,run=run)
    output                = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
    print(output.strip().decode())
# -

# ## Code for extension of the masks

# + tags=[]
# %%time
for scan in ['966975_rfMRI_REST3_PA']:
    print(scan)
    sbj,run = scan.split('_',1)
    work_folder = osp.join(DATA_DIR,sbj,run)
    command="module load afni; \
             cd {wd}; \
             pwd; \
             3dcalc -overwrite -a Segsy/Classes+orig. -expr 'equals(a,3)*within(k,0,40)*within(i,55,70)*within(j,30,50)' -prefix {run}_orig.FVmask2.auto.nii.gz; \
             3dClusterize -overwrite -nosum -1Dformat -inset {run}_orig.FVmask2.auto.nii.gz \
                          -idat 0 -ithr 0 -NN 2 -clust_nvox 50 -bisided -0.0001 0.0001 \
                          -pref_map {run}_orig.FVmask2.auto.clust.nii.gz; \
             3dcalc -overwrite -a {run}_orig.FVmask.nii.gz -b {run}_orig.FVmask2.auto.clust.nii.gz \
                    -expr 'k*((equals(k,18)*step(b))+a+(equals(k,17)*step(b))+(equals(k,16)*step(b)))' \
                    -prefix {run}_orig.FVmask2.nii.gz".format(wd=work_folder,run=run)
    output                = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
    print(output.strip().decode())
# -

# # Manually correct the ROIs to create the final version with 11 slices

# # Time-shift correction, remove initial 10 volumes and extract representative FV timeseries per slice

selected_scans = ['115017_rfMRI_REST1_PA', '115017_rfMRI_REST3_PA', '115825_rfMRI_REST1_PA', '134627_rfMRI_REST1_PA', '134627_rfMRI_REST3_PA', 
                  '146432_rfMRI_REST1_PA', '146432_rfMRI_REST2_AP', '175237_rfMRI_REST2_AP', '581450_rfMRI_REST3_PA', '581450_rfMRI_REST4_AP', 
                  '782561_rfMRI_REST2_AP', '899885_rfMRI_REST3_PA', '943862_rfMRI_REST1_PA', '943862_rfMRI_REST3_PA', '966975_rfMRI_REST1_PA', 
                  '966975_rfMRI_REST3_PA', '966975_rfMRI_REST4_AP']

# +
import pandas as pd
import numpy  as np
import os
import os.path as osp

from utils.variables import DATA_DIR
# -

N_slices        = 85
MB_factor       = 5
Increment_Slice = 2
N_shots   = int(N_slices / MB_factor)
print('++ INFO: Number of shots = %d' % N_shots)
print(' + Working with an odd number of shots')

pd.set_option('display.max_rows', 85)
slice_timing_info = pd.read_csv('/data/SFIMJGC_HCP7T/hcp7t_fv_sleep/Resources/HCP7T_SliceTimingInformation.csv')
slice_timing_info

slice_timing_byslice = slice_timing_info.sort_values(by='Slice Number')
slice_timing_byslice

slice_timing_byslice.hvplot(x='Slice Number', y='Slice Timing') * slice_timing_byslice.hvplot.scatter(x='Slice Number', y='Slice Timing')

slice_timing_inms = slice_timing_info.sort_values(by='Slice Number')['Slice Timing'].values
slice_timing_inms

np.savetxt('/data/SFIMJGC_HCP7T/HCP7T/ALL/ALL_orig.SliceTimings.info',slice_timing_inms, delimiter=' ', fmt='%0.2f', newline=' ')

# #### Apply the slice timing correction based on the timings we just extracted

if not osp.exists('./XX_InflowInECSegments.logs'):
    print('++ INFO: Creating logging dir')
    os.mkdir('./XX_InflowInECSegments.logs')

# Create Swarm file for extracting representative power
# ==========================================================
os.system('echo "#swarm -f ./XX_InflowInECSegments.SWARM.sh -g 32 -t 32 --partition quick,norm --module afni --logdir ./XX_InflowInECSegments.logs" > ./XX_InflowInECSegments.SWARM.sh')
for sbj_run in selected_scans:
    sbj,run  = sbj_run.split('_',1)
    out_dir  = osp.join(DATA_DIR,sbj,run)
    os.system('echo "export SBJ={sbj} RUN={run}; sh ./XX_InflowInECSegments.sh" >> ./XX_InflowInECSegments.SWARM.sh'.format(sbj=sbj, run=run))

# + [markdown] tags=[]
# ***
# # Plot Percentile across slices (Full FV or just frontal part)
# -

from utils.basics import load_motion_FD, get_available_runs, load_segments, load_PSD, get_time_index, get_window_index
from utils.variables import DATA_DIR
import subprocess
import os.path as osp
import os
import pandas as pd
import numpy as np
import xarray as xr
import hvplot.pandas
import hvplot.xarray
import holoviews as hv
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.colors import rgb2hex


def generate_sleep_psd(f,t,Sxx,SleepBand_BotFreq=0.03,SleepBand_TopFreq=0.07):
    # Put Spectrogram into pandas dataframe (tidy-data form)
    df = pd.DataFrame(Sxx.T, columns=f, index=t)
    df = pd.DataFrame(df.stack()).reset_index()
    df.columns=['TR','Freq','PSD']
    
    # Extract average timeseries of power at different bands
    sleep_band_power     = df[(df['Freq'] < SleepBand_TopFreq) & (df['Freq'] > SleepBand_BotFreq)].groupby('TR').mean()['PSD'].values
    non_sleep_band_power = df[(df['Freq'] > SleepBand_TopFreq) | (df['Freq'] < SleepBand_BotFreq)].groupby('TR').mean()['PSD'].values
    control_band_power   = df[(df['Freq'] > 0.1)               & (df['Freq'] < 0.2)].groupby('TR').mean()['PSD'].values
    total_power          = df.groupby('TR').mean()['PSD'].values
    
    # Create output dataframe, which will contain both the average power traces computed above,
    # as well as the ratio of power in sleep band to all other "control" conditions
    df3                   = pd.DataFrame(columns=['sleep','non_sleep','total','control','ratio_w_total','ratio_w_non_sleep','ratio_w_control'])
    df3['sleep']          = sleep_band_power
    df3['non_sleep']      = non_sleep_band_power
    df3['total']          = total_power
    df3['control']        = control_band_power

    df3['ratio_w_total']     = sleep_band_power/total_power
    df3['ratio_w_non_sleep'] = sleep_band_power/non_sleep_band_power
    df3['ratio_w_control']   = sleep_band_power/control_band_power
    # Set the time index for the output dataframe
    df3.index = t
    return df3


def generate_cmap_hex(cmap_name,num_colors):
    cmap     = get_cmap(cmap_name,num_colors)
    cmap_hex = []
    for i in range(cmap.N):
        rgba = cmap(i)
        cmap_hex.append(rgb2hex(rgba))
    return cmap_hex


Nacqs = 890
TR    = 1    # seconds
win_dur = 60 # seconds

time_index = get_time_index(Nacqs,TR)
win_index  = get_window_index(Nacqs,TR, win_dur)

Drowsy_Runs = get_available_runs(when='final', type='drowsy')

motFD       = load_motion_FD(Drowsy_Runs, index=time_index)
motFD.index = motFD.index.total_seconds()

PSDsleep       = load_PSD(Drowsy_Runs,band='sleep', index=win_index)
PSDsleep.index = PSDsleep.index.total_seconds()

EC_segments = load_segments('EC',Drowsy_Runs,60)

for r,row in EC_segments.iterrows():
    aux_onset, aux_offset, aux_run = int(row['Onset']), int(row['Offset']), row['Run']
    aux = motFD.loc[aux_onset:aux_offset,aux_run]
    EC_segments.loc[r,'FD_mean'] = aux.mean()
    EC_segments.loc[r,'FD_max'] = aux.max()
    aux = PSDsleep.loc[aux_onset:aux_offset,aux_run]
    EC_segments.loc[r,'PSDs_mean'] = aux.mean()
    EC_segments.loc[r,'PSDs_max'] = aux.max()

EC_segments.hvplot.scatter(x='FD_max',y='PSDs_max', hover_cols=['Run','Onset','Offset']) * \
hv.VLine(0.3).opts(line_width=1, line_dash='dashed', line_color='k') * \
hv.HLine(50).opts(line_width=1, line_dash='dashed', line_color='k')

selected_segments = EC_segments[(EC_segments['FD_max']<=0.3) & (EC_segments['PSDs_max']>=50)]
selected_scans    = list(selected_segments['Run'].unique())
selected_subjects = list(set([s.split('_',1)[0] for s in selected_scans]))
print('++ ===============================================================')
print('++ INFO: Number of selected EC segments: %d' % len(selected_segments))
print('++ INFO: Number of runs containing the selected segments: %d' % len(selected_scans))
print('++ INFO: Number of subjects containing the selected segments: %d' % len(selected_subjects))

# #### Compute 95to5 Ratio (slice-by-slice | Full region)

slice_index    = ['SL'+str(i+1).zfill(2) for i in np.arange(11)]
selected_segments_ids = selected_segments['Segment_UUID']
segment_TSs = {}
print(slice_index)

# %%time
# Full FV ROI
roiTS_shift_95to5ratio_FV = pd.DataFrame(columns=slice_index, index=selected_segments_ids)
for segmentID in selected_segments_ids:
    row     = selected_segments[selected_segments['Segment_UUID']==segmentID]
    scanID  = row['Run'].values[0]
    sbj,run = scanID.split('_',1)
    onset   = int(row['Onset'].values[0])
    offset  = int(row['Offset'].values[0]) 
    path    = osp.join('/data/SFIMJGC_HCP7T/HCP7T',sbj,run,'{run}_orig.tshift.FVmask.bySlice.csv'.format(run=run))
    if osp.exists(path):
        aux     = pd.read_csv(path, sep='\t').drop(['File','Sub-brick'],axis=1)
        aux     = aux[onset:offset]
        aux.columns = slice_index
        segment_TSs[('Full',str(segmentID))] = aux
        roiTS_shift_95to5ratio_FV.loc[str(segmentID)] = aux.quantile(0.95)/aux.quantile(0.05)
    else:
        print('++ WARNING: %s is missing' % path)
roiTS_shift_95to5ratio_FV_summary = pd.concat([roiTS_shift_95to5ratio_FV.mean(), roiTS_shift_95to5ratio_FV.std(), roiTS_shift_95to5ratio_FV.std()/np.sqrt(len(selected_scans))], axis=1)
roiTS_shift_95to5ratio_FV_summary.columns=['Mean','StDev','StErr']

# %%time
# Tube ROI
roiTS_shift_95to5ratio_TUBE = pd.DataFrame(columns=slice_index, index=selected_segments_ids)
for segmentID in selected_segments_ids:
    row     = selected_segments[selected_segments['Segment_UUID']==segmentID]
    scanID  = row['Run'].values[0]
    sbj,run = scanID.split('_',1)
    onset   = int(row['Onset'].values[0])
    offset  = int(row['Offset'].values[0]) 
    path    = osp.join('/data/SFIMJGC_HCP7T/HCP7T',sbj,run,'{run}_orig.tshift.FVmask.tube.bySlice.csv'.format(run=run))
    if osp.exists(path):
        aux     = pd.read_csv(path, sep='\t').drop(['File','Sub-brick'],axis=1)
        aux     = aux[onset:offset]
        aux.columns = slice_index
        segment_TSs[('Ventral',str(segmentID))] = aux
        roiTS_shift_95to5ratio_TUBE.loc[str(segmentID)] = aux.quantile(0.95)/aux.quantile(0.05)
    else:
        print('++ WARNING: %s is missing' % path)
roiTS_shift_95to5ratio_TUBE_summary = pd.concat([roiTS_shift_95to5ratio_TUBE.mean(), roiTS_shift_95to5ratio_TUBE.std(), roiTS_shift_95to5ratio_TUBE.std()/np.sqrt(len(selected_scans))], axis=1)
roiTS_shift_95to5ratio_TUBE_summary.columns=['Mean','StDev','StErr']

# %matplotlib inline
import matplotlib 
matplotlib.rc('xtick', labelsize=18) 
matplotlib.rc('ytick', labelsize=18) 
fig,axs=plt.subplots(1,2,figsize=(30,6))
for i,(df,title) in enumerate(zip([roiTS_shift_95to5ratio_FV_summary, roiTS_shift_95to5ratio_TUBE_summary],['4th Ventricle ROI (full)','4th Ventricle ROI (Ventral Portion)'])):
    df.plot(kind='bar', y='Mean', yerr='StErr', color=generate_cmap_hex('viridis',11), ax=axs[i])
    df.plot(y='Mean', ax=axs[i], c='k', lw=2)
    df.plot(y='Mean', ax=axs[i], c='k', lw=2)
    axs[i].set_ylim(1,1.4)
    axs[i].set_ylabel('95th vs 5th percentile ratio', fontsize=18)
    axs[i].set_xlabel('Slice Number', fontsize=18)
    axs[i].get_legend().remove()
    axs[i].set_title(title, fontsize=22)

# ***
# # Spectrogram-based Analysis of Power across slices

from scipy.signal import spectrogram, get_window
WIN_LENGTH  = 60
WIN_OVERLAP = 59  #59
NFFT        = 128 #64
SCALING     = 'density'
DETREND     = 'constant'
FS          = 1

roiTS_shift_PSDsleep = pd.DataFrame(columns=slice_index, index=selected_segments_ids)
for segmentID in selected_segments_ids:
    row     = selected_segments[selected_segments['Segment_UUID']==segmentID]
    scanID  = row['Run'].values[0]
    sbj,run = scanID.split('_',1)
    onset   = int(row['Onset'].values[0])
    offset  = int(row['Offset'].values[0]) 
    path        = osp.join(DATA_DIR,sbj,run,'{RUN}_orig.tshift.FVmask.bySlice.csv'.format(RUN=run))
    aux         = pd.read_csv(path, sep='\t').drop(['File','Sub-brick'],axis=1)
    aux.columns = slice_index
    
    for slice_id in slice_index:
        # Compute Spectrogram
        # ===================
        f,t,Sxx        = spectrogram(aux[slice_id],FS,window=get_window(('tukey',0.25),WIN_LENGTH), noverlap=WIN_OVERLAP, scaling=SCALING, nfft=NFFT, detrend=DETREND, mode='psd')
        
        # Compute Average Power/Time in Sleep Band
        # ========================================
        band_lim_spect_df = generate_sleep_psd(f,t,Sxx)
        
        # Save mean PSDsleep into summary dataframe
        # =========================================
        roiTS_shift_PSDsleep.loc[segmentID,slice_id] = band_lim_spect_df.loc[onset:offset,'control'].mean()

roiTS_shift_PSDsleep_summary = pd.concat([roiTS_shift_PSDsleep.mean(), roiTS_shift_PSDsleep.std(), roiTS_shift_PSDsleep.std()/np.sqrt(len(selected_scans))], axis=1)
roiTS_shift_PSDsleep_summary.columns=['Mean','StDev','StErr']

# %matplotlib inline
import matplotlib
matplotlib.rc('xtick', labelsize=18) 
matplotlib.rc('ytick', labelsize=18) 

# +

fig,axs=plt.subplots(1,1,figsize=(15,6))
roiTS_shift_PSDsleep_summary.plot(kind='bar', y='Mean', yerr='StErr', color=generate_cmap_hex('viridis',11), ax=axs)
roiTS_shift_PSDsleep_summary.plot(y='Mean', ax=axs, c='k', lw=2)
roiTS_shift_PSDsleep_summary.plot(y='Mean', ax=axs, c='k', lw=2)
#axs.set_ylim(1,1.4)
axs.set_ylabel('PSDcontrol', fontsize=18)
axs.set_xlabel('Slice Number', fontsize=18)
axs.get_legend().remove()
# -

# ***
# # GUI - See original Time series

import panel as pn

sel_segment   = pn.widgets.Select(name='Segment ID', options=[str(s) for s in selected_segments_ids])
sel_roi       = pn.widgets.Select(name='ROI', options=['Full','Ventral'])
sel_norm_mode = pn.widgets.Select(name='Normalization', options=['None','minus Mean','Divided by Mean','SPC'])


@pn.depends(sel_segment, sel_norm_mode, sel_roi)
def plot_sliceTS(segment, norm_mode, roi):
    aux = segment_TSs[(roi,segment)]
    if norm_mode == 'minus Mean':
        aux = aux - aux.mean()
    if norm_mode == 'Divided by Mean':
        aux = aux / aux.mean()
    if norm_mode == 'SPC':
        aux = 100*((aux - aux.mean()) / aux.mean())
    x_min = aux.index[0]
    x_max = x_min + 180
    return aux[['SL03','SL06','SL09']].hvplot(width=750, color=['#833995','#00aeed','#38b44c'], xlim=(x_min,x_max)).opts(legend_position='top')


pn.Column(pn.Row(sel_segment, sel_norm_mode, sel_roi), plot_sliceTS)

# # What/How to consider slice timing

slice_timing_path = '/data/SFIMJGC_HCP7T/HCP7T/ALL/ALL_orig.SliceTimings.info'
slice_timing      = pd.DataFrame(np.loadtxt(slice_timing_path))

roi_Ks = pd.DataFrame(index=slice_index,columns=selected_scans)
for scanID in selected_scans:
    sbj,run = scanID.split('_',1)
    roi_rank_path = osp.join(DATA_DIR,sbj,run,'{run}_orig.FVmask.rankmap.1D'.format(run=run))
    roi_rank         = pd.DataFrame(np.loadtxt(roi_rank_path, dtype=int))
    roi_rank.columns = ['Slice','K']
    roi_rank         = roi_rank.drop(0)
    roi_rank.index   = slice_index
    roi_Ks.loc[:,scanID] = roi_rank['K']

roi_Ks.loc['SL01'].value_counts()

roi_Ks

roi_Ks.columns[roi_Ks.loc['SL01']==18]

SS18     = selected_segments[selected_segments['Run'].isin(roi_Ks.columns[roi_Ks.loc['SL01']==18])]
SS18_ids = list(SS18['Segment_UUID'].values)

# %%time
# Full FV ROI
roiTS_shift_95to5ratio_FV = pd.DataFrame(columns=slice_index, index=SS18_ids)
for segmentID in SS18_ids:
    row     = selected_segments[selected_segments['Segment_UUID']==segmentID]
    scanID  = row['Run'].values[0]
    sbj,run = scanID.split('_',1)
    onset   = int(row['Onset'].values[0])
    offset  = int(row['Offset'].values[0]) 
    path    = osp.join('/data/SFIMJGC_HCP7T/HCP7T',sbj,run,'{run}_orig.tshift.FVmask.bySlice.csv'.format(run=run))
    if osp.exists(path):
        aux     = pd.read_csv(path, sep='\t').drop(['File','Sub-brick'],axis=1)
        aux     = aux[onset:offset]
        aux.columns = slice_index
        roiTS_shift_95to5ratio_FV.loc[segmentID] = aux.quantile(0.95)/aux.quantile(0.05)
    else:
        print('++ WARNING: %s is missing' % path)
roiTS_shift_95to5ratio_FV_summary = pd.concat([roiTS_shift_95to5ratio_FV.mean(), roiTS_shift_95to5ratio_FV.std(), roiTS_shift_95to5ratio_FV.std()/np.sqrt(len(selected_scans))], axis=1)
roiTS_shift_95to5ratio_FV_summary.columns=['Mean','StDev','StErr']

roiTS_shift_95to5ratio_FV

# %%time
# Tube ROI
roiTS_shift_95to5ratio_TUBE = pd.DataFrame(columns=slice_index, index=SS18_ids)
for segmentID in SS18_ids:
    row     = selected_segments[selected_segments['Segment_UUID']==segmentID]
    scanID  = row['Run'].values[0]
    sbj,run = scanID.split('_',1)
    onset   = int(row['Onset'].values[0])
    offset  = int(row['Offset'].values[0]) 
    path    = osp.join('/data/SFIMJGC_HCP7T/HCP7T',sbj,run,'{run}_orig.tshift.FVmask.tube.bySlice.csv'.format(run=run))
    if osp.exists(path):
        aux     = pd.read_csv(path, sep='\t').drop(['File','Sub-brick'],axis=1)
        aux     = aux[onset:offset]
        aux.columns = slice_index
        roiTS_shift_95to5ratio_TUBE.loc[segmentID] = aux.quantile(0.95)/aux.quantile(0.05)
    else:
        print('++ WARNING: %s is missing' % path)
roiTS_shift_95to5ratio_TUBE_summary = pd.concat([roiTS_shift_95to5ratio_TUBE.mean(), roiTS_shift_95to5ratio_TUBE.std(), roiTS_shift_95to5ratio_TUBE.std()/np.sqrt(len(selected_scans))], axis=1)
roiTS_shift_95to5ratio_TUBE_summary.columns=['Mean','StDev','StErr']

roiTS_shift_95to5ratio_TUBE

# %matplotlib inline
import matplotlib 
matplotlib.rc('xtick', labelsize=18) 
matplotlib.rc('ytick', labelsize=18) 
fig,axs=plt.subplots(1,2,figsize=(30,6))
for i,(df,title) in enumerate(zip([roiTS_shift_95to5ratio_FV_summary, roiTS_shift_95to5ratio_TUBE_summary],['4th Ventricle ROI (full)','4th Ventricle ROI (Ventral Portion)'])):
    df.plot(kind='bar', y='Mean', yerr='StErr', color=generate_cmap_hex('viridis',11), ax=axs[i])
    df.plot(y='Mean', ax=axs[i], c='k', lw=2)
    df.plot(y='Mean', ax=axs[i], c='k', lw=2)
    axs[i].set_ylim(1,1.4)
    axs[i].set_ylabel('95th vs 5th percentile ratio', fontsize=18)
    axs[i].set_xlabel('Slice Number', fontsize=18)
    axs[i].get_legend().remove()
    axs[i].set_title(title, fontsize=22)

a = slice_timing.copy()
a.columns = ['Time']
a = a[18:29]
a['Slice'] = slice_index
a

a.sort_values(by='Time')

slice_sorted_by_time = list(a.sort_values(by='Time')['Slice'].values)
print(slice_sorted_by_time)

# %matplotlib inline
import matplotlib 
matplotlib.rc('xtick', labelsize=18) 
matplotlib.rc('ytick', labelsize=18) 
fig,axs=plt.subplots(1,2,figsize=(30,6))
for i,(df,title) in enumerate(zip([roiTS_shift_95to5ratio_FV_summary.loc[slice_sorted_by_time], roiTS_shift_95to5ratio_TUBE_summary.loc[slice_sorted_by_time]],['4th Ventricle ROI (full)','4th Ventricle ROI (Ventral Portion)'])):
    df.plot(kind='bar', y='Mean', yerr='StErr', color=generate_cmap_hex('viridis',11), ax=axs[i])
    df.plot(y='Mean', ax=axs[i], c='k', lw=2)
    df.plot(y='Mean', ax=axs[i], c='k', lw=2)
    axs[i].set_ylim(1,1.4)
    axs[i].set_ylabel('95th vs 5th percentile ratio', fontsize=18)
    axs[i].set_xlabel('Slice Number', fontsize=18)
    axs[i].get_legend().remove()
    axs[i].set_title(title, fontsize=22)

roiTS_shift_95to5ratio_FV_summary.loc[slice_sorted_by_time]





slice_timing.hvplot() * slice_timing.hvplot.scatter()

slice_timing_i


