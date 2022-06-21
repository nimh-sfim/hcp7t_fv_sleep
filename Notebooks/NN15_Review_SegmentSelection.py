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
# This notebook is used to find long EC segments from Drowsy scans that have:
#
# * Low Motion: as defined by having a maximum FD below 0.4mm.
# * High PSD sleep: as defined by having a mean PSDsleep in the top half percentile.
#
# We write to disk informaion about this scans and segments so that we can download original data for them and do subsequent analysis in original space.

import pandas as pd
import numpy as np
import hvplot.pandas
import holoviews as hv
import os.path as osp
from utils.basics import get_time_index, get_window_index, get_available_runs, load_motion_FD, load_PSD, load_segments
from utils.variables import Resources_Dir

Nacqs       = 890
TR          = 1    # seconds
win_dur     = 60   # seconds

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

# + [markdown] tags=[]
# #### Plot Motion vs. PSDsleep for all selected periods of Eye Closure
# -

for r,row in EC_segments.iterrows():
    aux_onset, aux_offset, aux_run = int(row['Onset']), int(row['Offset']), row['Run']
    aux = motFD.loc[aux_onset:aux_offset,aux_run]
    EC_segments.loc[r,'FD_mean'] = aux.mean()
    EC_segments.loc[r,'FD_max'] = aux.max()
    aux = PSDsleep.loc[aux_onset:aux_offset,aux_run]
    EC_segments.loc[r,'PSDs_mean'] = aux.mean()
    EC_segments.loc[r,'PSDs_max'] = aux.max()

# #### Set Thresholds for Motion and PSD

thr_FDmax   = 0.4
thr_PSDmean = EC_segments['PSDs_mean'].quantile(0.50)

selected_segments = EC_segments[(EC_segments['FD_max']<=thr_FDmax) & (EC_segments['PSDs_mean']>=thr_PSDmean)]
selected_scans    = list(selected_segments['Run'].unique())
selected_subjects = list(set([s.split('_',1)[0] for s in selected_scans]))
print('++ ===============================================================')
print('++ INFO: Number of selected EC segments: %d' % len(selected_segments))
print('++ INFO: Number of runs containing the selected segments: %d' % len(selected_scans))
print('++ INFO: Number of subjects containing the selected segments: %d' % len(selected_subjects))

EC_segments.hvplot.scatter(x='FD_max',y='PSDs_mean', hover_cols=['Run','Onset','Offset'], xlabel='Maximum FD in segment [mm]', ylabel='Mean PSDsleep in segment', c='gray').opts(fontsize={'xlabel':12, 'ylabel':12, 'xticks':12, 'yticks':12}) * \
selected_segments.hvplot.scatter(x='FD_max',y='PSDs_mean',c='g') * \
hv.VLine(thr_FDmax).opts(line_width=3, line_dash='dashed', line_color='gray') * \
hv.HLine(thr_PSDmean).opts(line_width=3, line_dash='dashed', line_color='gray')

# #### Write lists of selected scans, segments and subjects to disk

selected_scans_csv_path = osp.join(Resources_Dir,'EC_lowMot_highPSD_scans.csv')
np.savetxt(selected_scans_csv_path,selected_scans, fmt='%s')

selected_segments_csv_path = osp.join(Resources_Dir,'EC_lowMot_highPSD_segments.csv')
selected_segments.to_csv(selected_segments_csv_path)

# #### Show lists of selected segments

print('++ INFO: Selected scans:')
print(selected_scans)

print('++ INFO: Selected segments:')
selected_segments.sort_values(by='Duration')


