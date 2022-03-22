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
# 6. Check for statistical differences in amplitude of cardiac traces across scan types and segment types
#
# ### Import Libraries

# +
import pandas as pd
import numpy  as np
import panel  as pn
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

from IPython.display import Markdown as md

from bokeh.models.formatters import DatetimeTickFormatter
formatter = DatetimeTickFormatter(minutes = ['%Mmin:%Ssec'])
# -

# ### Load Scan Lists
#
# We will now load the list of scans that are used in the main part of the analyses. For each scan we also have their label: "drowsy" or "awake"

Drowsy_scans = get_available_runs(when='final',type='drowsy')
Awake_scans  = get_available_runs(when='final',type='awake')
All_scans    = get_available_runs(when='final',type='all')
print('++ INFO: Total number of scans:  %d' % len(All_scans))
print('++ INFO: Number of Drowsy scans: %d' % len(Drowsy_scans))
print('++ INFO: Number of Awake scans:  %d' % len(Awake_scans))

# ### Load HAPPY-estimated Cardiac Traces (25hz) into a Dataframe
#
# In a previous notebook, we estiamted cardiac traces for each resint-state scan using the "happy" software. We now load these into a pandas dataframe. The extracted traces have a sampling frequency of 25Hz. The resulting dataframe will have a time index based on this frquency. It will also have one column per scan containing the stimated cardiac traces

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

cardiac_25hz_df.head()

# ***
# # Scan-wise Analyses
#
# First, we will explore potential differences in cardiac function at the scan level (i.e. drowsy vs. awake scans). Later in the notebook we will conduct similar analyses at the segment level (EO vs. EC).
#
# #### Estimate Spectrogram for each Trace, extract average HR, and aliased freq
#
# 1. Compute the spectrogram of cardiac traces using the whole scan.
# 2. Find the fundamental frequency (e.g., cardiac frequency)
# 3. Compute its aliased equivalent based on the fMRI sampling frequency
#
# The next figure a few cells below exemplifies this process.

fs_card = 25 # Hz (Frequency in standarized HAPPY outputs)
fs_fmri = 1  # Hz (TR in fMRI data)

# +
# %%time
cardiac_welch_df = pd.DataFrame(columns=All_scans)
cardiac_hr_df = pd.DataFrame(index=All_scans, columns=['HR','HR_aliased','Scan Type'])
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

sample_scan = '995174_rfMRI_REST3_PA'
aux_sbj, aux_run = sample_scan.split('_',1)
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

plot = aux_data['cardiacfromfmri_dlfiltered_25.0Hz'].hvplot(c='k',width=1500,xformatter=formatter) + \
(cardiac_welch_df[sample_scan].hvplot(c='k', ylabel='Power Spectrum') * \
hv.Text(8,4,'HR = {hr:.2f} Hz --> HR_aliased = {hra:.2f} Hz'.format(hr=cardiac_hr_df.loc[sample_scan,'HR'],hra=cardiac_hr_df.loc[sample_scan,'HR_aliased'])))
pn.pane.HoloViews(plot).save('./figures/card_estim_example.png')

text="![](./figures/card_estim_example.png)"
md("%s"%text)

#
# #### Distribution of scan-level frequencies
#
# **Cardiac Frequency Distribution**: Here is the distribution of estimated cardiac rates at the scan level. In red, we show what are normal ranges for resting cardiac rates (50bpm or 0.83 Hz <- -> 80bpm or 1.33 Hz). We can observe that with a few exception, estimated cardiac frequencies fall within tose normal ranges

plot=hv.Rectangles([(50/60,0,80/60,10)]).opts(alpha=0.5, color='r') * \
cardiac_hr_df.reset_index(drop=True).infer_objects().hvplot.hist(y='HR', color='gray', bins=30, normed=True, title='',ylim=(0,5), xlim=(0,2), height=300, width=500) * \
cardiac_hr_df.reset_index(drop=True).infer_objects().hvplot.kde( y='HR', color='gray', xlabel='Heart Rate [Hz]', ylabel='Density', fontsize={'xticks':18,'yticks':18,'ylabel':18,'xlabel':18, 'title':18}).opts(toolbar=None)
pn.pane.HoloViews(plot).save('./figures/Revision1_Figure7_PanelA.png')
text="![](./figures/Revision1_Figure7_PanelA.png)"
md("%s"%text)

plot = cardiac_hr_df.reset_index(drop=True).infer_objects().hvplot.box(y='HR', by='Scan Type', title='', hover_cols=['Scan ID'], tools=['hover'],
                                                                fontsize={'xticks':18,'yticks':18,'ylabel':18,'xlabel':18, 'title':18}, ylabel='HR [Hz]', color='Scan Type', cmap=['orange','lightblue'], legend=False, width=300, height=300).opts(toolbar=None)
pn.pane.HoloViews(plot).save('./figures/Revision1_Figure7_PanelC.png')
text="![](./figures/Revision1_Figure7_PanelC.png)"
md("%s"%text)

print('++ INFO: Statistical Tests for differences in HR across scan types')
print('++ ================================================================')
awake_hrs  = cardiac_hr_df[cardiac_hr_df['Scan Type']=='Awake']['HR']
drowsy_hrs = cardiac_hr_df[cardiac_hr_df['Scan Type']=='Drowsy']['HR']
tt_s, tt_p = ttest_ind(    awake_hrs, drowsy_hrs, alternative='two-sided')
mw_s, mw_p = mannwhitneyu( awake_hrs, drowsy_hrs, alternative='two-sided')
kk_s, kk_p = kruskal(      awake_hrs, drowsy_hrs)
print('   T-Test                   [HR EO different than EC] T    = %2.2f | p=%0.5f' % (tt_s, tt_p))
print('   Mann-Whitney U Rank Test [HR EO different than EC] Stat = %2.2f | p=%0.5f' % (mw_s, mw_p))
print('   Kruskas-Wallis H Test    [HR EO different than EC] Stat = %2.2f | p=%0.5f' % (kk_s, kk_p))

# **Aliased Cardiac Frequency Distribution:** distribution of aliased cardiac frequencies. We can see that those overlap with the targeted range of frequencies of this study (in green).

plot=hv.Rectangles([(0.03,0,0.07,10)]).opts(alpha=0.5, color='g') * \
cardiac_hr_df.reset_index(drop=True).infer_objects().hvplot.hist(y='HR_aliased', color='gray', bins=30, normed=True, title='', ylim=(0,10), xlim=(-.1,.8), height=300, width=500) * \
cardiac_hr_df.reset_index(drop=True).infer_objects().hvplot.kde(y='HR_aliased', color='gray', xlabel='Aliased Heart Rate [Hz]', ylabel='Density', fontsize={'xticks':18,'yticks':18,'ylabel':18,'xlabel':18, 'title':18}, xlim=(-.1,.8)).opts(toolbar=None)
pn.pane.HoloViews(plot).save('./figures/Revision1_Figure7_PanelB.png')
text="![](./figures/Revision1_Figure7_PanelB.png)"
md("%s"%text)

# * Plot group differences in aliased HR across scan types 

plot = cardiac_hr_df.reset_index(drop=True).infer_objects().hvplot.box(y='HR_aliased', by='Scan Type', title='', hover_cols=['Scan ID'], tools=['hover'],
                                                                fontsize={'xticks':18,'yticks':18,'ylabel':18,'xlabel':18, 'title':18}, ylabel='Aliased HR [Hz]', color='Scan Type', cmap=['orange','lightblue'], legend=False, width=300, height=300).opts(toolbar=None)
pn.pane.HoloViews(plot).save('./figures/Revision1_Figure7_PanelC.png')
text="![](./figures/Revision1_Figure7_PanelC.png)"
md("%s"%text)

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
    wf, wc    = welch(ts, fs=fs_card, window=get_window(('tukey',0.25),750), noverlap=375, scaling='density', detrend='constant', nfft=1024)
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

plot = hv.Rectangles([(50/60,0,80/60,10)]).opts(alpha=0.5, color='r') * \
segments.reset_index(drop=True).infer_objects().hvplot.hist(y='HR', color='gray', bins=30, normed=True, title='',ylim=(0,5), xlim=(0,2), height=300, width=500) * \
segments.reset_index(drop=True).infer_objects().hvplot.kde( y='HR', color='gray', xlabel='Heart Rate [Hz]', ylabel='Density', fontsize={'xticks':18,'yticks':18,'ylabel':18,'xlabel':18, 'title':18}).opts(toolbar=None)
pn.pane.HoloViews(plot).save('./figures/Revision1_Figure7_PanelD.png')
text="![](./figures/Revision1_Figure7_PanelD.png)"
md("%s"%text)

plot = segments.reset_index(drop=True).infer_objects().hvplot.box(y='HR', by='Segment Type', width=300, height=300,
                                                           title='',fontsize={'xticks':18,'yticks':18,'ylabel':18,'xlabel':18, 'title':18}, 
                                                           ylabel='HR [Hz]', color='Segment Type', cmap=['orange','lightblue'], legend=False).opts(toolbar=None)
pn.pane.HoloViews(plot).save('./figures/Revision1_Figure7_PanelX.png')
text="![](./figures/Revision1_Figure7_PanelX.png)"
md("%s"%text)

print('++ INFO: Statistical Tests for differences in HR across segment types')
print('++ ==================================================================')
eo_hrs = segments[segments['Segment Type']=='EO']['HR']
ec_hrs = segments[segments['Segment Type']=='EC']['HR']
tt_s, tt_p = ttest_ind(    eo_hrs, ec_hrs, alternative='two-sided')
mw_s, mw_p = mannwhitneyu( eo_hrs, ec_hrs, alternative='two-sided')
kk_s, kk_p = kruskal(      eo_hrs, ec_hrs)
print('   T-Test                   [HR EO different than EC] T    = %2.2f | p=%0.5f' % (tt_s, tt_p))
print('   Mann-Whitney U Rank Test [HR EO different than EC] Stat = %2.2f | p=%0.5f' % (mw_s, mw_p))
print('   Kruskas-Wallis H Test    [HR EO different than EC] Stat = %2.2f | p=%0.5f' % (kk_s, kk_p))

plot = hv.Rectangles([(0.03,0,0.07,10)]).opts(alpha=0.5, color='g') * \
segments.reset_index(drop=True).infer_objects().hvplot.hist(y='HR_aliased', color='gray', bins=30, normed=True, title='', ylim=(0,10), xlim=(-.1,.8), height=300, width=500) * \
segments.reset_index(drop=True).infer_objects().hvplot.kde(y='HR_aliased', color='gray', xlabel='Aliased Heart Rate [Hz]', ylabel='Density', fontsize={'xticks':18,'yticks':18,'ylabel':18,'xlabel':18, 'title':18}, xlim=(-.1,.8)).opts(toolbar=None)
pn.pane.HoloViews(plot).save('./figures/Revision1_Figure7_PanelE.png')
text="![](./figures/Revision1_Figure7_PanelE.png)"
md("%s"%text)

plot = segments.reset_index(drop=True).infer_objects().hvplot.box(y='HR_aliased', by='Segment Type', width=300, height=300,
                                                           title='',fontsize={'xticks':18,'yticks':18,'ylabel':18,'xlabel':18, 'title':18}, 
                                                           ylabel='Aliased HR [Hz]', color='Segment Type', cmap=['orange','lightblue'], legend=False).opts(toolbar=None)
pn.pane.HoloViews(plot).save('./figures/Revision1_Figure7_PanelF.png')
text="![](./figures/Revision1_Figure7_PanelF.png)"
md("%s"%text)

print('++ INFO: Statistical Tests for differences in aliased HR across segment types')
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

print('++ INFO: C_Amplitude for EO [mean +/- stdev] = %.2f +/- %.2f' % (segments.groupby(by='Segment Type').mean().loc['EO','C_Amplitude'],segments.groupby(by='Segment Type').std().loc['EO','C_Amplitude']))
print('++ INFO: C_Amplitude for EC [mean +/- stdev] = %.2f +/- %.2f' % (segments.groupby(by='Segment Type').mean().loc['EC','C_Amplitude'],segments.groupby(by='Segment Type').std().loc['EC','C_Amplitude']))

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

segments.reset_index(drop=True).infer_objects().hvplot.kde(y='C_Amplitude', by='Segment Type', xlabel='Amplitude', 
                                                          fontsize={'xticks':18,'yticks':18,'ylabel':18,'xlabel':18, 'title':18}, color=['orange','lightblue']).opts(toolbar=None, legend_position='top_left')

cardiac_amp_df.reset_index(drop=True).infer_objects().hvplot.kde(y='C_Amplitude', by='Scan Type', xlabel='Amplitude', 
                                                          fontsize={'xticks':18,'yticks':18,'ylabel':18,'xlabel':18, 'title':18}, color=['orange','lightblue']).opts(toolbar=None, legend_position='top_left')

# ***
# ***
# # END OF NOTEBOOK
# ***
# ***
