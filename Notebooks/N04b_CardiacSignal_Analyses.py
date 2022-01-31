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
# This notebook will use the outputs from the HAPPY software to perform the following operations:
#
# 1. Extract the fundamental HR per scan
# 2. Compute the aliased HR frequency per scan
# 3. Save scan-wise HR into to ```Resources/HR_scaninfo.csv```
# 4. Check for statistical differences in HR across scan types (i.e., drowsy vs. awake)
# 5. Check for statistical differences in aliased HR across scan types (i.e., drowsy vs. awake)
#
# ### Import Libraries

# +
import pandas as pd
import numpy  as np
import hvplot.pandas
import holoviews as hv
import matplotlib.pyplot as plt
import seaborn as sns
import os.path as osp
import json

from utils.basics import get_available_runs, load_segments, aliased_freq
from utils.variables import DATA_DIR, Resources_Dir

from scipy.signal import get_window, spectrogram, welch
from scipy.stats import kruskal, wilcoxon, ttest_ind, mannwhitneyu
from scipy.signal import find_peaks

from bokeh.models.formatters import DatetimeTickFormatter
formatter = DatetimeTickFormatter(minutes = ['%Mmin:%Ssec'])
# -

# ### Load Scan Lists

Drowsy_scans = get_available_runs(when='final',type='drowsy')
Awake_scans  = get_available_runs(when='final',type='awake')
All_scans    = get_available_runs(when='final',type='all')
print('++ INFO: Total number of scans:  %d' % len(All_scans))
print('++ INFO: Number of Drowsy scans: %d' % len(Drowsy_scans))
print('++ INFO: Number of Awake scans:  %d' % len(Awake_scans))

# ### Load HAPPY-estimated Cardiac Traces (25hz) into a Dataframe

# Create Time Delta Index
cardiac_25hz_df_index = pd.timedelta_range(start='0 s', end='900 s', periods=((900*25)-1))
cardiac_25hz_df_index.shape

# %%time
cardiac_25hz_df = pd.DataFrame(columns=All_scans,index=cardiac_25hz_df_index)
for scanID in All_scans:
    aux_sbj, aux_run = scanID.split('_',1)
    # Load Cardiac Trace
    aux_data_path = osp.join(DATA_DIR, aux_sbj, aux_run, '{run}_orig.happy'.format(run=aux_run),'{run}_orig.happy_desc-stdrescardfromfmri_timeseries.tsv'.format(run=aux_run))
    aux_json_path = osp.join(DATA_DIR, aux_sbj, aux_run, '{run}_orig.happy'.format(run=aux_run),'{run}_orig.happy_desc-stdrescardfromfmri_timeseries.json'.format(run=aux_run))
    # Load Json File
    with open(aux_json_path) as json_file:
        aux_json = json.load(json_file)
    # Load Data
    aux_data         = pd.read_csv(aux_data_path,sep='\t', header=None)
    aux_data.columns = aux_json['Columns']
    # Load Sampling Frequency
    aux_fs = aux_json['SamplingFrequency']
    #Add to DF
    cardiac_25hz_df[scanID] = aux_data['cardiacfromfmri_dlfiltered_25.0Hz'].values

# ***
# ## Scan-wise Analyses
#
# #### Estimate Spectrogram for each Trace, extract average HR, and aliased freq

fs_card = 25 # Hz (Frequency in standarized HAPPY outputs)
fs_fmri = 1  # Hz (TR in fMRI data)

cardiac_25hz_df[0:255]

# +
# %%time
cardiac_welch_df = pd.DataFrame(columns=All_scans)
cardiac_hr_df = pd.DataFrame(index=All_scans, columns=['HR','HR aliased','Scan Type'])
for scanID in All_scans:
    #wf, wc      = welch(cardiac_25hz_df[scanID], fs=fs_card, window=get_window(('tukey',0.25),256), noverlap=128, scaling='density', detrend='constant', nfft=1024)
    wf, wc      = welch(cardiac_25hz_df[scanID], fs=fs_card, window=get_window(('tukey',0.25),750), noverlap=375, scaling='density', detrend='constant', nfft=1024)
    
    cardiac_welch_df[scanID] = wc
    # Extract Average HR
    cardiac_hr_df.loc[scanID,'HR'] = wf[cardiac_welch_df[scanID].idxmax()]
    # Compute alisaed HR
    cardiac_hr_df.loc[scanID,'HR_aliased'] = aliased_freq(fs_fmri,cardiac_hr_df.loc[scanID,'HR'] )
    #n                                      = round(cardiac_hr_df.loc[scanID,'HR'] / float(fs_fmri))
    #cardiac_hr_df.loc[scanID,'HR aliased'] = abs(fs_fmri * n - cardiac_hr_df.loc[scanID,'HR'])
    # Add Scan Type
    if scanID in Awake_scans:
        cardiac_hr_df.loc[scanID,'Scan Type'] = 'Awake'
    if scanID in Drowsy_scans:
        cardiac_hr_df.loc[scanID,'Scan Type'] = 'Drowsy'
    
cardiac_welch_df.index      = wf
cardiac_welch_df.index.name = 'Frequency [Hz]' 
# -

# #### Representative Figures with one scan (to explain method)

sample_scan = All_scans[30]
aux_sbj, aux_run = scanID.split('_',1)
# Load Cardiac Trace
aux_data_path = osp.join(DATA_DIR, aux_sbj, aux_run, '{run}_orig.happy'.format(run=aux_run),'{run}_orig.happy_desc-stdrescardfromfmri_timeseries.tsv'.format(run=aux_run))
aux_json_path = osp.join(DATA_DIR, aux_sbj, aux_run, '{run}_orig.happy'.format(run=aux_run),'{run}_orig.happy_desc-stdrescardfromfmri_timeseries.json'.format(run=aux_run))
# Load Json File
with open(aux_json_path) as json_file:
    aux_json = json.load(json_file)
# Load Data
aux_data         = pd.read_csv(aux_data_path,sep='\t', header=None)
aux_data.columns = aux_json['Columns']
# Udpate Index
aux_data.index  =  cardiac_25hz_df_index
aux_data.index.name = 'Time'

aux_data['cardiacfromfmri_dlfiltered_25.0Hz'].hvplot(c='k',width=1500,xformatter=formatter) + \
(cardiac_welch_df[sample_scan].hvplot(c='k', ylabel='Power Spectrum') * \
hv.Text(8,4,'HR = {hr:.2f} Hz --> HR_aliased = {hra:.2f} Hz'.format(hr=cardiac_hr_df.loc[sample_scan,'HR'],hra=cardiac_hr_df.loc[sample_scan,'HR_aliased'])))

#
# #### Distribution of scan-level frequencies
#
# * Plot distributon of original HRs for all scans

hv.Rectangles([(50/60,0,80/60,10)]).opts(alpha=0.5, color='r') * \
cardiac_hr_df.reset_index(drop=True).infer_objects().hvplot.hist(y='HR', color='gray', bins=30, normed=True, title='(A) Distribution of Scan-Level Heart Rate',ylim=(0,5), xlim=(0,2)) * \
cardiac_hr_df.reset_index(drop=True).infer_objects().hvplot.kde( y='HR', color='gray', xlabel='Heart Rate [Hz]', ylabel='Density', fontsize={'xticks':18,'yticks':18,'ylabel':18,'xlabel':18, 'title':18}).opts(toolbar=None)

# * Plot distribution of aliased HRs for all scans

hv.Rectangles([(0.03,0,0.07,10)]).opts(alpha=0.5, color='r') * \
cardiac_hr_df.reset_index(drop=True).infer_objects().hvplot.hist(y='HR_aliased', color='gray', bins=30, normed=True, title='(B) Distribution of Scan-Level Aliased Heart Rate', ylim=(0,10), xlim=(-.1,.8)) * \
cardiac_hr_df.reset_index(drop=True).infer_objects().hvplot.kde(y='HR_aliased', color='gray', xlabel='Aliased Heart Rate [Hz]', ylabel='Density', fontsize={'xticks':18,'yticks':18,'ylabel':18,'xlabel':18, 'title':18}, xlim=(-.1,.8)).opts(toolbar=None)

# * Plot group differences in aliased HR across scan types 

cardiac_hr_df.reset_index(drop=True).infer_objects().hvplot.box(y='HR_aliased', by='Scan Type', title='(C) Aliased HR segregated by Scan Type',
                                                                fontsize={'xticks':18,'yticks':18,'ylabel':18,'xlabel':18, 'title':18}, ylabel='Aliased HR [Hz]', color='Scan Type', cmap=['orange','lightblue'], legend=False).opts(toolbar=None)

# * Test for statistical differences in aliased HR at the group level

print('++ INFO: Statistical Tests for differences in HRa across scan types')
print('++ ================================================================')
awake_hrs  = cardiac_hr_df[cardiac_hr_df['Scan Type']=='Awake']['HR_aliased']
drowsy_hrs = cardiac_hr_df[cardiac_hr_df['Scan Type']=='Drowsy']['HR_aliased']
tt_s, tt_p = ttest_ind(    awake_hrs, drowsy_hrs, alternative='two-sided')
mw_s, mw_p = mannwhitneyu( awake_hrs, drowsy_hrs, alternative='two-sided')
kk_s, kk_p = kruskal(      awake_hrs, drowsy_hrs)
print('   T-Test                   [HRa EO different than EC] T    = %2.2f | p=%0.5f' % (tt_s, tt_p))
print('   Mann-Whitney U Rank Test [HRa EO different than EC] Stat = %2.2f | p=%0.5f' % (mw_s, mw_p))
print('   Kruskas-Wallis H Test    [HRa EO different than EC] Stat = %2.2f | p=%0.5f' % (kk_s, kk_p))

# * Save scan-wise HR info to disk

path = osp.join(Resources_Dir,'HR_scaninfo.csv')
cardiac_hr_df.to_csv(path)
print('++ INFO: Save scan-wise HR info to disk [%s]' % path)
cardiac_hr_df.head()

# + [markdown] tags=[] jp-MarkdownHeadingCollapsed=true
# ***
# # Segment-wise Analyses
#
# ### Load segments longer than 60s (EO & EC)
# -

segments = load_segments('all',min_dur=60)

segments.columns = ['Run', 'Segment Type', 'Segment_Index', 'Segment_UUID', 'Onset', 'Offset', 'Duration', 'Scan_Type']

# ### Compute HR and aliased-HR per segment

# +
# %%time
hr_list, hr_alias_list = [],[]
for r,row in segments.iterrows():
    scanID    = row['Run']
    segID     = row['Segment_UUID']
    onset     = pd.Timedelta(int(row['Onset']), unit='s')
    offset    = pd.Timedelta(int(row['Offset']), unit='s')
    time_mask = (cardiac_25hz_df.index >=onset) & (cardiac_25hz_df.index <=offset)
    ts        = cardiac_25hz_df.loc[time_mask,scanID]
    wf, wc      = welch(ts, fs=fs_card, window=get_window(('tukey',0.25),750), noverlap=375, scaling='density', detrend='constant', nfft=1024)
    hr        = wf[wc.argmax()]
    # Compute aliased HR
    hr_alias  = aliased_freq(fs_fmri,hr)
    hr_list.append(hr)
    hr_alias_list.append(hr_alias)

segments['HR'] = hr_list
segments['HR_aliased'] = hr_alias_list
# -

# #### Save segment-wise HR info to disk

path = osp.join(Resources_Dir,'HR_segmentinfo.csv')
segments.to_csv(path)
print('++ INFO: Save scan-wise HR info to disk [%s]' % path)
segments.head()

hv.Rectangles([(50/60,0,80/60,10)]).opts(alpha=0.5, color='r') * \
segments.reset_index(drop=True).infer_objects().hvplot.hist(y='HR', color='gray', bins=30, normed=True, title='(D) Distribution of Segment-Level Heart Rate',ylim=(0,5), xlim=(0,2)) * \
segments.reset_index(drop=True).infer_objects().hvplot.kde( y='HR', color='gray', xlabel='Heart Rate [Hz]', ylabel='Density', fontsize={'xticks':18,'yticks':18,'ylabel':18,'xlabel':18, 'title':18}).opts(toolbar=None)

hv.Rectangles([(0.03,0,0.07,10)]).opts(alpha=0.5, color='r') * \
segments.reset_index(drop=True).infer_objects().hvplot.hist(y='HR_aliased', color='gray', bins=30, normed=True, title='(E) Distribution of Segment-Level Aliased Heart Rate', ylim=(0,10), xlim=(-.1,.8)) * \
segments.reset_index(drop=True).infer_objects().hvplot.kde(y='HR_aliased', color='gray', xlabel='Aliased Heart Rate [Hz]', ylabel='Density', fontsize={'xticks':18,'yticks':18,'ylabel':18,'xlabel':18, 'title':18}, xlim=(-.1,.8)).opts(toolbar=None)

segments.reset_index(drop=True).infer_objects().hvplot.box(y='HR_aliased', by='Segment Type', 
                                                           title='(F) Aliased HR segregated by Segment Type',fontsize={'xticks':18,'yticks':18,'ylabel':18,'xlabel':18, 'title':18}, 
                                                           ylabel='Aliased HR [Hz]', color='Segment Type', cmap=['orange','lightblue'], legend=False).opts(toolbar=None)

print('++ INFO: Statistical Tests for differences in HR across segment types')
print('++ ==================================================================')
eo_hrs = segments[segments['Segment Type']=='EO']['HR_aliased']
ec_hrs = segments[segments['Segment Type']=='EC']['HR_aliased']
tt_s, tt_p = ttest_ind(    eo_hrs, ec_hrs, alternative='two-sided')
mw_s, mw_p = mannwhitneyu( eo_hrs, ec_hrs, alternative='two-sided')
kk_s, kk_p = kruskal(      eo_hrs, ec_hrs)
print('   T-Test                   [HR EO different than EC] T    = %2.2f | p=%0.5f' % (tt_s, tt_p))
print('   Mann-Whitney U Rank Test [HR EO different than EC] Stat = %2.2f | p=%0.5f' % (mw_s, mw_p))
print('   Kruskas-Wallis H Test    [HR EO different than EC] Stat = %2.2f | p=%0.5f' % (kk_s, kk_p))

# ***
# # Scan-wise Cardiac Amplitude

cardiac_amp_df = pd.DataFrame(cardiac_25hz_df.std(),columns=['C_Amplitude'])
for scanID in All_scans:
    # Add Scan Type
    if scanID in Awake_scans:
        cardiac_amp_df.loc[scanID,'Scan Type'] = 'Awake'
    if scanID in Drowsy_scans:
        cardiac_amp_df.loc[scanID,'Scan Type'] = 'Drowsy'

cardiac_amp_df.reset_index(drop=True).infer_objects().hvplot.box(y='C_Amplitude', ylabel='Amplitude',
                                                                 by='Scan Type', 
                                                                 title='(G) Amplitude of Cardiac Signal by Scan Type',fontsize={'xticks':18,'yticks':18,'ylabel':18,'xlabel':18, 'title':18}, color='Scan Type', cmap=['orange','lightblue'], legend=False).opts(toolbar=None)

print('++ INFO: Statistical Tests for differences in HR Amp across scan types')
print('++ ================================================================')
awake_amps  = cardiac_amp_df[cardiac_amp_df['Scan Type']=='Awake']['C_Amplitude']
drowsy_amps = cardiac_amp_df[cardiac_amp_df['Scan Type']=='Drowsy']['C_Amplitude']
tt_s, tt_p = ttest_ind(    awake_amps, drowsy_amps, alternative='two-sided')
mw_s, mw_p = mannwhitneyu( awake_amps, drowsy_amps, alternative='two-sided')
kk_s, kk_p = kruskal(      awake_amps, drowsy_amps)
print('   T-Test                   [HR Amp EO different than EC] T    = %2.2f | p=%0.5f' % (tt_s, tt_p))
print('   Mann-Whitney U Rank Test [HR Amp EO different than EC] Stat = %2.2f | p=%0.5f' % (mw_s, mw_p))
print('   Kruskas-Wallis H Test    [HR Amp EO different than EC] Stat = %2.2f | p=%0.5f' % (kk_s, kk_p))

# ***
# # Segment-wise Amplitude

# +
# %%time
amp_list = []
for r,row in segments.iterrows():
    scanID    = row['Run']
    segID     = row['Segment_UUID']
    onset     = pd.Timedelta(int(row['Onset']), unit='s')
    offset    = pd.Timedelta(int(row['Offset']), unit='s')
    time_mask = (cardiac_25hz_df.index >=onset) & (cardiac_25hz_df.index <=offset)
    ts        = cardiac_25hz_df.loc[time_mask,scanID]
    amp_list.append(ts.std())
    
segments['C_Amplitude'] = amp_list
# -

segments.reset_index(drop=True).infer_objects().hvplot.box(y='C_Amplitude', ylabel='Amplitude', hover_cols=['Run','Duration'], tools=['hover'],
                                                           by='Segment Type',
                                                           title='(H) Amplitude of Cardiac Signal by Segment Type',
                                                           fontsize={'xticks':18,'yticks':18,'ylabel':18,'xlabel':18, 'title':18}, color='Segment Type', cmap=['orange','lightblue'], legend=False).opts(toolbar=None)

segments.reset_index(drop=True).infer_objects().hvplot.kde(y='C_Amplitude', by='Segment Type', xlabel='Amplitude', 
                                                          fontsize={'xticks':18,'yticks':18,'ylabel':18,'xlabel':18, 'title':18}, color=['orange','lightblue']).opts(toolbar=None, legend_position='top_left')

cardiac_amp_df.reset_index(drop=True).infer_objects().hvplot.kde(y='C_Amplitude', by='Scan Type', xlabel='Amplitude', 
                                                          fontsize={'xticks':18,'yticks':18,'ylabel':18,'xlabel':18, 'title':18}, color=['orange','lightblue']).opts(toolbar=None, legend_position='top_left')

print('++ INFO: Statistical Tests for differences in HR across segment types')
print('++ ==================================================================')
eo_amps = segments[segments['Segment Type']=='EO']['C_Amplitude']
ec_amps = segments[segments['Segment Type']=='EC']['C_Amplitude']
tt_s, tt_p = ttest_ind(    eo_amps, ec_amps, alternative='two-sided')
mw_s, mw_p = mannwhitneyu( eo_amps, ec_amps, alternative='two-sided')
kk_s, kk_p = kruskal(      eo_amps, ec_amps)
print('   T-Test                   [HR Amp EO different than EC] T    = %2.2f | p=%0.5f' % (tt_s, tt_p))
print('   Mann-Whitney U Rank Test [HR Amp EO different than EC] Stat = %2.2f | p=%0.5f' % (mw_s, mw_p))
print('   Kruskas-Wallis H Test    [HR Amp EO different than EC] Stat = %2.2f | p=%0.5f' % (kk_s, kk_p))

print("++ INFO: Average amplitude of cardiac signal in EO segments: %.2f" % eo_amps.mean())
print("++ INFO: Average amplitude of cardiac signal in EC segments: %.2f" % ec_amps.mean())

# ***
# ***
# # END OF NOTEBOOK
# ***
# ***

# + active=""
# %matplotlib inline
# fig, axs = plt.subplots(1,2,figsize=(20,5))
# segments.boxplot('HR', by='Type', ax=axs[0], positions=[2,1])
# segments.boxplot('Cardiac Amplitude', by='Type', ax=axs[1],positions=[2,1])
