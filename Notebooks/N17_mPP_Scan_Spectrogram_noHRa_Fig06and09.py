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
# #### Description - Create Figures regarding the Spectrogram Analyses
#
# The primary outputs of this notebook include:
#
# * Figure 3: Representative subject and description of the spectrogram analyses
# -

# # Import Libraries

# +
import os
import pandas as pd
import numpy  as np
import os.path as osp
from utils.variables import DATA_DIR, Resources_Dir
from utils.basics import get_available_runs

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
import hvplot.pandas
import panel as pn

from scipy.stats import ttest_ind

# %matplotlib inline
# -

# # Analysis Configuration Variables

region             = 'V4lt_grp' # Use the Group-Level FV ROI
gs_region          = 'GM'       # Region used to define the global signal (options='GM','FB')
gs_step            = 'mPP'      # Options: 'mPP','Reference','Basic','Basicpp','Behzadi_COMPCOR','Behzadi_COMPCORpp' | Manuscript: 'mPP'
spectrogram_windur = 60         # Spectrogram Window Duration (In seconds)
Nacq               = 890        # Number of acquisitions

# # Generate Time Index for Windowed Results
#
# Having these two indexes will help us plot original data and windowed frequency information in a way that aligns visually and helps with interpretation

# First we generate a regular time index (the one that corresponds to the fMRI TR)
time_index         = pd.timedelta_range(start='0 s', periods=Nacq, freq='s')
print('++ Time Index (first 10 values):')
print(time_index[0:10])

# Create empty dataframe with index having time_delta in steps of seconds
aux = pd.DataFrame(np.ones(Nacq),index=time_index)
# Simulate rolling windows of spectrogram_windur to gather which index should have data
aux                    = aux.rolling(window=spectrogram_windur, center=True).mean()
aux                    = aux.dropna()
windowed_time_index    = aux.index
print('++ Window Index (first 10 values):')
print(windowed_time_index[0:10])

# + [markdown] tags=[]
# ***
# # Figure 6: Evolution of PSD across scan types
#
# ### Load list of runs classified as awake and drowsy that do not overlap with HRa
# -

scan_HR_info    = pd.read_csv(osp.join(Resources_Dir,'HR_scaninfo.csv'), index_col=0)
scan_HR_info    = scan_HR_info[(scan_HR_info['HR_aliased']< 0.03) | (scan_HR_info['HR_aliased']> 0.07)]
Manuscript_Runs = list(scan_HR_info.index)
Awake_Runs      = list(scan_HR_info[scan_HR_info['Scan Type']=='Awake'].index)
Drowsy_Runs     = list(scan_HR_info[scan_HR_info['Scan Type']=='Drowsy'].index)
print('++ INFO: Number of Runs = %d' % len(Manuscript_Runs))
print('++ INFO: Number of Awake  Runs = %d' % len(Awake_Runs))
print('++ INFO: Number of Drowsy Runs = %d' % len(Drowsy_Runs))

# + [markdown] tags=[]
# ***
# ### Load PSD in Sleep Band
#
# This data is already the result of a windowing operation (computation of the spectrogram). As such, we ensure the index has the correct values corresponding to windowed information

# +
# %%time
sleep_psd_DF = pd.DataFrame(index=windowed_time_index, columns=Manuscript_Runs)
ctrol_psd_DF = pd.DataFrame(index=windowed_time_index, columns=Manuscript_Runs)

for sbj_run in Manuscript_Runs:
    sbj,run = sbj_run.split('_',1)
    file    = '{RUN}_mPP.Signal.{REGION}.Spectrogram_BandLimited.pkl'.format(RUN=run, REGION=region)
    path    = osp.join(DATA_DIR,sbj,run,file)
    aux     = pd.read_pickle(path)
    sleep_psd_DF.loc[(windowed_time_index,sbj_run)] = aux['sleep'].values
    ctrol_psd_DF.loc[(windowed_time_index,sbj_run)] = aux['control'].values

sleep_psd_cumsum_DF = sleep_psd_DF.cumsum()
ctrol_psd_cumsum_DF = ctrol_psd_DF.cumsum()

print('++ INFO: Shape of PSD Dataframe [sleep_psd_DF] is %s' % str(sleep_psd_DF.shape))

# + [markdown] tags=[]
# ***
# # Prepare PSD Dataframes for plotting with Seaborn & Compute Cumulative Version of PSD Estimates
# -

# Putting average PSD data into tidy-form with axis in terms of seconds (not delta time)
sleep_psd_DF_copy               = sleep_psd_DF.copy()
sleep_psd_DF_copy.index         = sleep_psd_DF_copy.index.total_seconds()
sleep_psd_cumsum_DF_copy        = sleep_psd_cumsum_DF.copy()
sleep_psd_cumsum_DF_copy.index  = sleep_psd_cumsum_DF_copy.index.total_seconds()
sleep_psd_stacked_DF            = pd.DataFrame(sleep_psd_DF_copy.stack()).reset_index()
sleep_psd_cumsum_stacked_DF     = pd.DataFrame(sleep_psd_cumsum_DF_copy.stack()).reset_index()
# Putting average PSD data into tidy-form with axis in terms of seconds (not delta time)
ctrol_psd_DF_copy               = ctrol_psd_DF.copy()
ctrol_psd_DF_copy.index         = ctrol_psd_DF_copy.index.total_seconds()
ctrol_psd_cumsum_DF_copy        = ctrol_psd_cumsum_DF.copy()
ctrol_psd_cumsum_DF_copy.index  = ctrol_psd_cumsum_DF_copy.index.total_seconds()
ctrol_psd_stacked_DF            = pd.DataFrame(ctrol_psd_DF_copy.stack()).reset_index()
ctrol_psd_cumsum_stacked_DF     = pd.DataFrame(ctrol_psd_cumsum_DF_copy.stack()).reset_index()
# Delete temporary Objects
del sleep_psd_DF_copy, sleep_psd_cumsum_DF_copy, ctrol_psd_DF_copy, ctrol_psd_cumsum_DF_copy

# Rename columns in tidy form dataframes and add extra column with run classification (Drowsy or Awake)
for df in [sleep_psd_stacked_DF, sleep_psd_cumsum_stacked_DF,
           ctrol_psd_stacked_DF, ctrol_psd_cumsum_stacked_DF]:
    df.columns = ['Time [seconds]','Subject','PSD']
    df['Scan Type'] = 'N/A'
    df.loc[(df['Subject'].isin(Awake_Runs),'Scan Type')] = 'Awake'
    df.loc[(df['Subject'].isin(Drowsy_Runs),'Scan Type')] = 'Drowsy'
sleep_psd_stacked_DF['Frequency Band']        = 'Sleep (0.03 - 0.07 Hz)'
sleep_psd_cumsum_stacked_DF['Frequency Band'] = 'Sleep (0.03 - 0.07 Hz)'
ctrol_psd_stacked_DF['Frequency Band']        = 'Control (0.1 - 0.2 Hz)'
ctrol_psd_cumsum_stacked_DF['Frequency Band'] = 'Control (0.1 - 0.2 Hz)'

# + [markdown] tags=[]
# ## Save into a single Dataframe Object for Plotting
# -

df_cum_plot = pd.concat([sleep_psd_cumsum_stacked_DF,ctrol_psd_cumsum_stacked_DF])
df_plot = pd.concat([sleep_psd_stacked_DF,ctrol_psd_stacked_DF])

# ## Plot the data

sns.set(font_scale=1.5)
sns.set_style("whitegrid",{"xtick.major.size": 0.1,
    "xtick.minor.size": 0.05,'grid.linestyle': '--'})

fig, ax = plt.subplots(2,1,figsize=(30,15))
sns.lineplot(data=df_plot, x='Time [seconds]', hue='Scan Type', y='PSD', style='Frequency Band', style_order=['Sleep (0.03 - 0.07 Hz)','Control (0.1 - 0.2 Hz)'], estimator=np.mean, n_boot=100, ax=ax[0])
ax[0].set_title('(A) Average PSD in 4th ventricle for the two bands of interest')
sns.lineplot(data=df_cum_plot, x='Time [seconds]', hue='Scan Type', y='PSD', style='Frequency Band', style_order=['Sleep (0.03 - 0.07 Hz)','Control (0.1 - 0.2 Hz)'], estimator=np.mean, n_boot=100, ax=ax[1])
ax[1].set_title('(A) Average Cumulative PSD in 4th ventricle for the two bands of interest')
ax[1].legend(ncol=2)

# > **<u>FINDING:</u>**: As scanning progresses, we see a differentiation in terms of instaneous PSD between subjects who kept their eyes open continously and those who did not. The second group tends to have a higher PSD at longer scanning times. Now becuase periods of eye closure do not necessarily align across subjects, the strength observed here is somehow weak.
#
# > **<u>FINDING:</u>**: As scanning progresses, we see a differentiation in terms of cumulative PSD between subjects who kept their eyes open continously and those who did not. The second group tends to have a higher cumulative PSD at longer scanning times. By looking at cumulative PSD we overcome the limitation mentioned above, and the differentiation across groups becomes more evident

fig.savefig('./figures/Fig06_PSDacrossTime.{region}.noHRa.png'.format(region=region))

# ***
# # Scan Ranking based on PSDsleep and GS
#
# ## Sort by PSD
#
# Create a figure that shows all scans as colored bars (color by scan type) sorted by PSDsleep rank. This will be one panel in Figure 9

scans_rank                  = pd.DataFrame(columns=['Scan Type'])
scans_rank['PSDsleep Rank'] = sleep_psd_DF.mean().rank(ascending=False).astype(int)
scans_rank['PSDsleep']      = sleep_psd_DF.mean()
scans_rank['Scan Type']     = 'N/A'
scans_rank.loc[(scans_rank.index.isin(Awake_Runs),'Scan Type')] = 'Awake'
scans_rank.loc[(scans_rank.index.isin(Drowsy_Runs),'Scan Type')] = 'Drowsy'

# + [markdown] tags=[]
# ## Sort by StDev of Global Signal
#
# First, we need to load the GS associated with each scan
# -

# %%time
GS_df = pd.DataFrame(columns=Manuscript_Runs)
for item in Manuscript_Runs:
    sbj,run  = item.split('_',1)
    aux_path = osp.join(DATA_DIR,sbj,run,'{RUN}_mPP.Signal.FB.1D'.format(RUN=run))
    aux_data = np.loadtxt(aux_path)
    assert aux_data.shape[0] == Nacq, "{file} has incorrect length {length}".format(file=aux_path, length=str(aux_data.shape[0]))
    GS_df[item] = aux_data

# Next, we add the new information to the ranking dataframe

scans_rank['GSamplitude']      = GS_df.std()
scans_rank['GSamplitude Rank'] = GS_df.std().rank(ascending=False).astype(int)
scans_rank.head()

# ## Generate Figure Panels

fig_ranked_scans = scans_rank.sort_values(by='PSDsleep', ascending=False).hvplot.bar(x='PSDsleep Rank', 
                                                                            y='PSDsleep', 
                                                                            color='Scan Type', 
                                                                            cmap={'Drowsy':'lightblue','Awake':'orange'}, 
                                                                            hover_cols=['index'], size=20, width=800, line_width=0, legend='top_right',
                                                                            title='(A) Scans ranked by PSDsleep of fMRI signal from the 4th Ventricle',
                                                                            fontsize={'minor_ticks':14, 'ticks':14, 'title':16, 'labels':14, 'legend':14}).opts(toolbar=None)
fig_ranked_scans

# Create a second bar plot with the amount of scans of each type in the following rank ranges (0 - 100, 100 - 200, 200 - 300, etc).

df_plot                           = pd.DataFrame(index=['1-100','100-200','200-300','300-366'],columns=['Drowsy','Awake'])
df_plot.loc[('1-100','Awake')]    = scans_rank[(scans_rank['PSDsleep Rank']<=100) & (scans_rank['Scan Type']=='Awake')].shape[0]
df_plot.loc[('1-100','Drowsy')]   = scans_rank[(scans_rank['PSDsleep Rank']<=100) & (scans_rank['Scan Type']=='Drowsy')].shape[0]
df_plot.loc[('100-200','Awake')]  = scans_rank[(scans_rank['PSDsleep Rank']>100) & (scans_rank['PSDsleep Rank']<=200) & (scans_rank['Scan Type']=='Awake')].shape[0]
df_plot.loc[('100-200','Drowsy')] = scans_rank[(scans_rank['PSDsleep Rank']>100) & (scans_rank['PSDsleep Rank']<=200) & (scans_rank['Scan Type']=='Drowsy')].shape[0]
df_plot.loc[('200-300','Awake')]  = scans_rank[(scans_rank['PSDsleep Rank']>200) & (scans_rank['PSDsleep Rank']<=300) & (scans_rank['Scan Type']=='Awake')].shape[0]
df_plot.loc[('200-300','Drowsy')] = scans_rank[(scans_rank['PSDsleep Rank']>200) & (scans_rank['PSDsleep Rank']<=300) & (scans_rank['Scan Type']=='Drowsy')].shape[0]
df_plot.loc[('300-366','Awake')]  = scans_rank[(scans_rank['PSDsleep Rank']>300) & (scans_rank['PSDsleep Rank']<367) & (scans_rank['Scan Type']=='Awake')].shape[0]
df_plot.loc[('300-366','Drowsy')] = scans_rank[(scans_rank['PSDsleep Rank']>300) & (scans_rank['PSDsleep Rank']<367) & (scans_rank['Scan Type']=='Drowsy')].shape[0]

fig_rank_segments_total      = df_plot.hvplot.bar(title='(C) Number of Scans per Rank Segment (Based on PSDsleep)',
                                             width=800,
                                             fontsize={'minor_ticks':14, 'ticks':14, 'title':16,'labels':14},
                                             cmap={'Drowsy':'lightblue','Awake':'orange'}, 
                                             ylabel='Number of Scans', xlabel='Rank Segments').opts(toolbar=None)
fig_rank_segments_total

# Next, we generate a bar graph with one bar per scan colored by scan type and sorted according to the rank based on GS (Full brain)

fig_gs_ranked_scans = scans_rank.sort_values(by='GSamplitude', ascending=False).hvplot.bar(x='GSamplitude Rank', 
                                                                            y='GSamplitude', 
                                                                            color='Scan Type', 
                                                                            cmap={'Drowsy':'lightblue','Awake':'orange'}, 
                                                                            hover_cols=['index'], size=20, width=800,line_width=0, legend='top_right',
                                                                            title='(B) Scans ranked by GSamplitude',
                                                                            fontsize={'minor_ticks':14, 'ticks':14, 'title':16, 'labels':14, 'legend':14}).opts(toolbar=None)
fig_gs_ranked_scans

# Create a second bar plot with the amount of scans of each type in the following rank ranges (0 - 100, 100 - 200, 200 - 300, etc).

df_plot                           = pd.DataFrame(index=['1-100','100-200','200-300','300-366'],columns=['Drowsy','Awake'])
df_plot.loc[('1-100','Awake')]    = scans_rank[(scans_rank['GSamplitude Rank']<=100) & (scans_rank['Scan Type']=='Awake')].shape[0]
df_plot.loc[('1-100','Drowsy')]   = scans_rank[(scans_rank['GSamplitude Rank']<=100) & (scans_rank['Scan Type']=='Drowsy')].shape[0]
df_plot.loc[('100-200','Awake')]  = scans_rank[(scans_rank['GSamplitude Rank']>100) & (scans_rank['GSamplitude Rank']<=200) & (scans_rank['Scan Type']=='Awake')].shape[0]
df_plot.loc[('100-200','Drowsy')] = scans_rank[(scans_rank['GSamplitude Rank']>100) & (scans_rank['GSamplitude Rank']<=200) & (scans_rank['Scan Type']=='Drowsy')].shape[0]
df_plot.loc[('200-300','Awake')]  = scans_rank[(scans_rank['GSamplitude Rank']>200) & (scans_rank['GSamplitude Rank']<=300) & (scans_rank['Scan Type']=='Awake')].shape[0]
df_plot.loc[('200-300','Drowsy')] = scans_rank[(scans_rank['GSamplitude Rank']>200) & (scans_rank['GSamplitude Rank']<=300) & (scans_rank['Scan Type']=='Drowsy')].shape[0]
df_plot.loc[('300-366','Awake')]  = scans_rank[(scans_rank['GSamplitude Rank']>300) & (scans_rank['GSamplitude Rank']<367) & (scans_rank['Scan Type']=='Awake')].shape[0]
df_plot.loc[('300-366','Drowsy')] = scans_rank[(scans_rank['GSamplitude Rank']>300) & (scans_rank['GSamplitude Rank']<367) & (scans_rank['Scan Type']=='Drowsy')].shape[0]

fig_gs_rank_segments_total = df_plot.hvplot.bar(title='(D) Number of Scans per Rank Segment (Based on GS Amplitude)',
                                             width=800,
                                             fontsize={'minor_ticks':14, 'ticks':14, 'title':16,'labels':14},
                                             cmap={'Drowsy':'lightblue','Awake':'orange'}, 
                                             ylabel='Number of Scans', xlabel='Rank Segments').opts(toolbar=None)
fig_gs_rank_segments_total

figure10 = pn.Column(pn.Row(fig_ranked_scans,fig_gs_ranked_scans),
          pn.Row(fig_rank_segments_total,fig_gs_rank_segments_total))

figure10.save('./figures/Fig10_ScanRakings.{region}.noHRa.png'.format(region=region))

figure10

# ## Save the list of scans on the top and bottom 100 for both GS and PSD, so we can use those later on N13 to look for network differences

# +
GS_Top100_Runs  = scans_rank[(scans_rank['GSamplitude Rank']<=100)].index.to_list()
GS_Bot100_Runs  = scans_rank[(scans_rank['GSamplitude Rank']>266)].index.to_list()
PSD_Top100_Runs = scans_rank[(scans_rank['PSDsleep Rank']<=100)].index.to_list()
PSD_Bot100_Runs = scans_rank[(scans_rank['PSDsleep Rank']>266)].index.to_list()

for items,filename in zip([GS_Top100_Runs,GS_Bot100_Runs,PSD_Top100_Runs,PSD_Bot100_Runs],
                      ['Run_List_GS_Top100.{region}.noHRa.txt'.format(region=region),
                       'Run_List_GS_Bot100.{region}.noHRa.txt'.format(region=region),
                       'Run_List_PSD_Top100.{region}.noHRa.txt'.format(region=region),
                       'Run_List_PSD_Bot100.{region}.noHRa.txt'.format(region=region)]):
    path = osp.join(Resources_Dir,filename)
    print('++ Saving scan list: %s' % filename)
    with open(path, 'w') as filehandle:
        for listitem in items:
            filehandle.write('%s\n' % listitem)
# -


