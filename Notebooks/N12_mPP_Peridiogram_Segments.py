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
# # Description - Do Spectral Power Analysis at the Segment Level (Welch Method)
#
# This notebook will do the following operations:
#
# * Compute the spectrograms for each identified Eyes Open (EO) and Eyes Closed (EC) segment
# * Looks for statistical differences in power spectra across both types of scan segments
# * Generate Panel B of figure 5
#
# > Becuase of how fast these operations are, this time we do not rely on swarm jobs, but do all the computations as part of this notebook
#
# Primary outputs from this notebook include:
#
# * ```Resources/ET_Peridiograms_perSegments_EC.pkl```: Peridiograms for all EC segments lasting more than 60 seconds.
# * ```Resources/ET_Peridiograms_perSegments_EO.pkl```: Peridiograms for all EO segments lasting more than 60 seconds.
# * Panel B of Figure 5
# -

# ***
# # Import Libraries

# +
import pandas as pd
import numpy as np
import os.path as osp
from utils.variables import Resources_Dir, DATA_DIR
from utils.basics import load_segments
from scipy.signal import get_window, welch
from scipy.stats  import kruskal

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
# -

# ## Configuration of Scan Level PSD Analysis

REGION      = 'V4lt_grp'      # Work with Group Level FV mask
WIN_LENGTH  = 60            # Window Length for Welch Method (in seconds)
WIN_OVERLAP = 45            # Window Overlap for Welch Method (in seconds)
NFFT        = 128           # FFT Length for Welch Method (in number of samples)
SCALING     = 'density'     # Scaling method
DETREND     = 'constant'    # Type of detrending
FS          = 1             # 1/TR (in Hertzs)
NACQ        = 890           # Number of time-points
ONLY_NOTaliased_HR = False

# # Loading and selecting Scan segments of interest

Scan_Segments = {}
if ONLY_NOTaliased_HR:
    segment_HR_info = pd.read_csv(osp.join(Resources_Dir,'HR_segmentinfo.csv'), index_col=0)
    Scan_Segments['EC'] = segment_HR_info[((segment_HR_info['HR aliased']<0.03) | (segment_HR_info['HR aliased']>0.07)) & (segment_HR_info['Type']=='EC')]
    Scan_Segments['EO'] = segment_HR_info[((segment_HR_info['HR aliased']<0.03) | (segment_HR_info['HR aliased']>0.07)) & (segment_HR_info['Type']=='EO')]
else:
    Scan_Segments['EC'] = load_segments('EC',min_dur=60)
    Scan_Segments['EO'] = load_segments('EO',min_dur=60)
num_EC = Scan_Segments['EC'].shape[0]
num_EO = Scan_Segments['EO'].shape[0]

print('++ Number of EC segments with duration > 60 seconds is %d' % num_EC)
print('++ Number of EO segments with duration > 60 seconds is %d' % num_EO)

# ***
# ## Compute Peridiograms (via Welch method) on a segment-per-segment basis

# %%time
EC_col_names = [tuple(x) for r,x in Scan_Segments['EC'][['Run','Segment_UUID']].iterrows()]
EO_col_names = [tuple(x) for r,x in Scan_Segments['EO'][['Run','Segment_UUID']].iterrows()]
## Load periods of EC and compute the peridiogram for each fo them
# ================================================================
Scan_Segments_Peridiograms = {'EC':pd.DataFrame(columns=EC_col_names),'EO':pd.DataFrame(columns=EO_col_names)}
for segment_type in ['EC','EO']:
    # For each segment
    for i, segment in Scan_Segments[segment_type].iterrows():
        # Load the timeseries from the 4th ventricle
        sbj,run      = (segment['Run']).split('_',1)
        roi_ts_path  = osp.join(DATA_DIR,sbj,run,'{run}_mPP.Signal.{region}.1D'.format(run=run, region=REGION))
        roi_ts       = pd.read_csv(roi_ts_path, header=None)
        # Ensure the timeseries are complete
        assert roi_ts.shape[0] == NACQ, "++ ERROR: {file_path} did not have {NACQ} datapoints.".format(file_path=roi_ts_path, NACQ=NACQ)
        # Select samples within the segment of interest
        roi_ts.columns = [REGION]
        data           = roi_ts[int(segment['Onset']):int(segment['Offset'])]
        # Compute the peridiogram
        wf, wc         = welch(data[REGION], fs=FS, window=get_window(('tukey',0.25),WIN_LENGTH), noverlap=WIN_OVERLAP, scaling=SCALING, detrend=DETREND, nfft=NFFT)
        Scan_Segments_Peridiograms[segment_type][(segment[('Run')],segment[('Segment_UUID')])] = wc
    Scan_Segments_Peridiograms[segment_type].index=wf
    Scan_Segments_Peridiograms[segment_type].index.rename('Frequency',inplace=True)

Scan_Segments_Peridiograms['EC'].head()

# ## Save Peridiograms to Disk

if not ONLY_NOTaliased_HR:
    print("++ Writing peridiograms to disk")
    Scan_Segments_Peridiograms['EC'].to_pickle(osp.join(Resources_Dir,'ET_Peridiograms_perSegments_EC.{region}.pkl'.format(region=REGION)))
    Scan_Segments_Peridiograms['EO'].to_pickle(osp.join(Resources_Dir,'ET_Peridiograms_perSegments_EO.{region}.pkl'.format(region=REGION)))

# ***
#
# # Prepare Real Data to be drawn with Seaborn

sns.set(font_scale=2)
df_EC                 = Scan_Segments_Peridiograms['EC'].stack().reset_index()
df_EC.columns         = ['Frequency','Run','PSD (a.u./Hz)']
df_EC['Segment Type'] = 'Eyes Closed'
df_EO                 = Scan_Segments_Peridiograms['EO'].stack().reset_index()
df_EO.columns         = ['Frequency','Run','PSD (a.u./Hz)']
df_EO['Segment Type'] = 'Eyes Open'
df_todraw             = pd.concat([df_EO,df_EC])

# ### Check for statistical difference in average PSD at each frequency interval

welch_freqs = Scan_Segments_Peridiograms['EC'].index.tolist()
kw_tests    = {'KW':[],'p':[],'Bonf_sign':[]}
for f in welch_freqs:
    EO_data  = Scan_Segments_Peridiograms['EO'].loc[f]
    EC_data  = Scan_Segments_Peridiograms['EC'].loc[f]
    kw,p     = kruskal(EO_data,EC_data)
    kw_tests['KW'].append(kw)
    kw_tests['p'].append(p)
kw_tests['Bonf_sign'] = [80 if p<0.05/len(welch_freqs) else np.nan for p in kw_tests['p']]

# ***
#
# ## Prepare Randomize Version of Data (Sanity check not used in publication)

available_segments = df_todraw['Run'].unique().tolist()

from random import sample
list01 = sample(available_segments,num_EO)
list02 = [r for r in available_segments if r not in list01]
print('++ INFO: Number of scans assigned to list 01 is %d' % len(list01))
print('++ INFO: Number of scans assigned to list 02 is %d' % len(list02))

df_all_segments  = pd.concat([Scan_Segments_Peridiograms['EC'],Scan_Segments_Peridiograms['EO']],axis=1)
random_kw_tests = {'KW':[],'p':[],'Bonf_sign':[]}
for f in welch_freqs:
    list01_data = df_all_segments.loc[(0.0000,df_all_segments.columns.intersection(list01))]
    list02_data = df_all_segments.loc[(0.0000,df_all_segments.columns.intersection(list02))]
    kw,p = kruskal(list01_data,list02_data)
    random_kw_tests['KW'].append(kw)
    random_kw_tests['p'].append(p)
    random_kw_tests['Bonf_sign'] = [80 if p<0.05/df_all_segments.shape[0] else np.nan for p in random_kw_tests['p']]

df_random_todraw  = pd.concat([Scan_Segments_Peridiograms['EC'],Scan_Segments_Peridiograms['EO']],axis=1)
df_random_todraw  = df_random_todraw.stack().reset_index()
df_random_todraw.columns  = ['Frequency','Run','PSD (a.u./Hz)']
df_random_todraw['Segment Type'] = 'N/A'
df_random_todraw.loc[(df_random_todraw['Run'].isin(list01),'Segment Type')] = 'Random Type 01'
df_random_todraw.loc[(df_random_todraw['Run'].isin(list02),'Segment Type')] = 'Random Type 02'

# ***
# ## Plotting Results

sns.set(font_scale=1.5)
sns.set_style("whitegrid",{"xtick.major.size": 0.1,
    "xtick.minor.size": 0.05,'grid.linestyle': '--'})
fig, axs   = plt.subplots(1,2,figsize=(20,5))
sns.lineplot(data=df_todraw, 
             x='Frequency', 
             hue='Segment Type', hue_order=['Eyes Closed', 'Eyes Open'],
             y='PSD (a.u./Hz)', estimator=np.mean, n_boot=100, ax=axs[0])
axs[0].set_title('Power Spectral Density (Segment Level)')
axs[0].legend(ncol=1, loc='upper right')
axs[0].plot(welch_freqs,kw_tests['Bonf_sign'],'k*',lw=5)
axs[0].set_ylim([0,90])
axs[0].xaxis.set_major_locator(ticker.MultipleLocator(0.05))
sns.lineplot(data=df_random_todraw, 
             x='Frequency', 
             hue='Segment Type', hue_order=['Random Type 01', 'Random Type 02'],
             y='PSD (a.u./Hz)', estimator=np.mean, n_boot=100, ax=axs[1])
axs[1].set_title('Power Spectral Density (Segment Level - Randomized Labels)')
axs[1].legend(ncol=1, loc='upper right')
axs[1].plot(welch_freqs,random_kw_tests['Bonf_sign'],'k*',lw=5)
axs[1].set_ylim([0,90])
axs[1].xaxis.set_major_locator(ticker.MultipleLocator(0.05))

# Print the frequencies for which we detected statistical differences

for i,f in enumerate(df_all_segments.index):
    if kw_tests['p'][i]<0.05/df_all_segments.shape[0]:
        print(f,end=',')

fig

# ### 6. Save Figures to disk

# + tags=[]
fig.savefig('./figures/Fig05_PanelB.png')
# -
sns.set(font_scale=1.5)
sns.set_style("whitegrid",{"xtick.major.size": 0.1,
    "xtick.minor.size": 0.05,'grid.linestyle': '--'})
fig, axs   = plt.subplots(1,2,figsize=(20,5))
sns.lineplot(data=df_todraw, 
             x='Frequency', 
             hue='Segment Type', hue_order=['Eyes Closed', 'Eyes Open'],
             y='PSD (a.u./Hz)', estimator=np.mean, n_boot=100, ax=axs[0])
axs[0].set_title('Power Spectral Density (Segment Level)')
axs[0].legend(ncol=1, loc='upper right')
axs[0].plot(welch_freqs,kw_tests['Bonf_sign'],'k*',lw=5)
axs[0].set_ylim([0,90])
axs[0].xaxis.set_major_locator(ticker.MultipleLocator(0.05))
axs[0].set(xscale="log")
sns.lineplot(data=df_random_todraw, 
             x='Frequency', 
             hue='Segment Type', hue_order=['Random Type 01', 'Random Type 02'],
             y='PSD (a.u./Hz)', estimator=np.mean, n_boot=100, ax=axs[1])
axs[1].set_title('Power Spectral Density (Segment Level - Randomized Labels)')
axs[1].legend(ncol=1, loc='upper right')
axs[1].plot(welch_freqs,random_kw_tests['Bonf_sign'],'k*',lw=5)
axs[1].set_ylim([0,90])
axs[1].xaxis.set_major_locator(ticker.MultipleLocator(0.05))
axs[1].set(xscale="log")
fig



