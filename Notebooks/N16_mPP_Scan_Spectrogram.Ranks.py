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

sns.set(font_scale=1.5)
sns.set_style("whitegrid",{"xtick.major.size": 0.1,
    "xtick.minor.size": 0.05,'grid.linestyle': '--'})

# # Analysis Configuration Variables

region             = 'V4lt_grp' # Use the Group-Level FV ROI
gs_region          = 'GM'       # Region used to define the global signal (options='GM','FB')
gs_step            = 'mPP'      # Options: 'mPP','Reference','Basic','Basicpp','Behzadi_COMPCOR','Behzadi_COMPCORpp' | Manuscript: 'mPP'
remove_overlap_HR  = False      # If true, do analyses after removing scans with overlapping alisaed HR in the sleep band
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
# # Figure: Evolution of PSD across scan types
#
# ### Load list of runs classified as awake and drowsy
# -

# %%time
if remove_overlap_HR:
    scenario        = 'noHRa'
    scan_HR_info    = pd.read_csv(osp.join(Resources_Dir,'HR_scaninfo.csv'), index_col=0)
    scan_HR_info    = scan_HR_info[(scan_HR_info['HR_aliased']< 0.03) | (scan_HR_info['HR_aliased']> 0.07)]
    Manuscript_Runs = list(scan_HR_info.index)
    Awake_Runs      = list(scan_HR_info[scan_HR_info['Scan Type']=='Awake'].index)
    Drowsy_Runs     = list(scan_HR_info[scan_HR_info['Scan Type']=='Drowsy'].index)
else:
    scenario        = 'all'
    Manuscript_Runs = get_available_runs(when='final', type='all')
    Awake_Runs      = get_available_runs(when='final', type='awake')
    Drowsy_Runs     = get_available_runs(when='final', type='drowsy')
print('++ INFO: Number of Runs: Total = %d | Awake = %d | Drowsy = %d' % (len(Manuscript_Runs), len(Awake_Runs), len(Drowsy_Runs)))

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
# -

# ***
# ## Scan Ranking based on PSDsleep and GS
#
# ### Sort by PSD
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
    aux_path = osp.join(DATA_DIR,sbj,run,'{RUN}_{gs_step}.Signal.{gs_region}.1D'.format(RUN=run, gs_region=gs_region, gs_step=gs_step))
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

df_plot                           = pd.DataFrame(index=['1-100','100-200','200-300','300-404'],columns=['Drowsy','Awake'])
df_plot.loc[('1-100','Awake')]    = scans_rank[(scans_rank['PSDsleep Rank']<=100) & (scans_rank['Scan Type']=='Awake')].shape[0]
df_plot.loc[('1-100','Drowsy')]   = scans_rank[(scans_rank['PSDsleep Rank']<=100) & (scans_rank['Scan Type']=='Drowsy')].shape[0]
df_plot.loc[('100-200','Awake')]  = scans_rank[(scans_rank['PSDsleep Rank']>100) & (scans_rank['PSDsleep Rank']<=200) & (scans_rank['Scan Type']=='Awake')].shape[0]
df_plot.loc[('100-200','Drowsy')] = scans_rank[(scans_rank['PSDsleep Rank']>100) & (scans_rank['PSDsleep Rank']<=200) & (scans_rank['Scan Type']=='Drowsy')].shape[0]
df_plot.loc[('200-300','Awake')]  = scans_rank[(scans_rank['PSDsleep Rank']>200) & (scans_rank['PSDsleep Rank']<=300) & (scans_rank['Scan Type']=='Awake')].shape[0]
df_plot.loc[('200-300','Drowsy')] = scans_rank[(scans_rank['PSDsleep Rank']>200) & (scans_rank['PSDsleep Rank']<=300) & (scans_rank['Scan Type']=='Drowsy')].shape[0]
df_plot.loc[('300-404','Awake')]  = scans_rank[(scans_rank['PSDsleep Rank']>300) & (scans_rank['PSDsleep Rank']<405) & (scans_rank['Scan Type']=='Awake')].shape[0]
df_plot.loc[('300-404','Drowsy')] = scans_rank[(scans_rank['PSDsleep Rank']>300) & (scans_rank['PSDsleep Rank']<405) & (scans_rank['Scan Type']=='Drowsy')].shape[0]

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

df_plot                           = pd.DataFrame(index=['1-100','100-200','200-300','300-404'],columns=['Drowsy','Awake'])
df_plot.loc[('1-100','Awake')]    = scans_rank[(scans_rank['GSamplitude Rank']<=100) & (scans_rank['Scan Type']=='Awake')].shape[0]
df_plot.loc[('1-100','Drowsy')]   = scans_rank[(scans_rank['GSamplitude Rank']<=100) & (scans_rank['Scan Type']=='Drowsy')].shape[0]
df_plot.loc[('100-200','Awake')]  = scans_rank[(scans_rank['GSamplitude Rank']>100) & (scans_rank['GSamplitude Rank']<=200) & (scans_rank['Scan Type']=='Awake')].shape[0]
df_plot.loc[('100-200','Drowsy')] = scans_rank[(scans_rank['GSamplitude Rank']>100) & (scans_rank['GSamplitude Rank']<=200) & (scans_rank['Scan Type']=='Drowsy')].shape[0]
df_plot.loc[('200-300','Awake')]  = scans_rank[(scans_rank['GSamplitude Rank']>200) & (scans_rank['GSamplitude Rank']<=300) & (scans_rank['Scan Type']=='Awake')].shape[0]
df_plot.loc[('200-300','Drowsy')] = scans_rank[(scans_rank['GSamplitude Rank']>200) & (scans_rank['GSamplitude Rank']<=300) & (scans_rank['Scan Type']=='Drowsy')].shape[0]
df_plot.loc[('300-404','Awake')]  = scans_rank[(scans_rank['GSamplitude Rank']>300) & (scans_rank['GSamplitude Rank']<405) & (scans_rank['Scan Type']=='Awake')].shape[0]
df_plot.loc[('300-404','Drowsy')] = scans_rank[(scans_rank['GSamplitude Rank']>300) & (scans_rank['GSamplitude Rank']<405) & (scans_rank['Scan Type']=='Drowsy')].shape[0]

fig_gs_rank_segments_total = df_plot.hvplot.bar(title='(D) Number of Scans per Rank Segment (Based on GS Amplitude)',
                                             width=800,
                                             fontsize={'minor_ticks':14, 'ticks':14, 'title':16,'labels':14},
                                             cmap={'Drowsy':'lightblue','Awake':'orange'}, 
                                             ylabel='Number of Scans', xlabel='Rank Segments').opts(toolbar=None)
fig_gs_rank_segments_total

figure10 = pn.Column(pn.Row(fig_ranked_scans,fig_gs_ranked_scans),
          pn.Row(fig_rank_segments_total,fig_gs_rank_segments_total))

figure10.save('./figures/Fig10_ScanRakings.{region}.{scenario}.png'.format(region=region, scenario=scenario))

figure10

# ## Save the list of scans on the top and bottom 100 for both GS and PSD, so we can use those later on N13 to look for network differences

# +
GS_Top100_Runs  = scans_rank[(scans_rank['GSamplitude Rank']<=100)].index.to_list()
GS_Bot100_Runs  = scans_rank[(scans_rank['GSamplitude Rank']>304)].index.to_list()
PSD_Top100_Runs = scans_rank[(scans_rank['PSDsleep Rank']<=100)].index.to_list()
PSD_Bot100_Runs = scans_rank[(scans_rank['PSDsleep Rank']>304)].index.to_list()

for items,filename in zip([GS_Top100_Runs,GS_Bot100_Runs,PSD_Top100_Runs,PSD_Bot100_Runs],
                      ['Run_List_GS_Top100.{region}.{scenario}.txt'.format(region=region, scenario=scenario),
                       'Run_List_GS_Bot100.{region}.{scenario}.txt'.format(region=region, scenario=scenario),
                       'Run_List_PSD_Top100.{region}.{scenario}.txt'.format(region=region, scenario=scenario),
                       'Run_List_PSD_Bot100.{region}.{scenario}.txt'.format(region=region, scenario=scenario)]):
    path = osp.join(Resources_Dir,filename)
    print('++ Saving scan list: %s' % filename)
    with open(path, 'w') as filehandle:
        for listitem in items:
            filehandle.write('%s\n' % listitem)
# -

# # Statistical Differences in PSDsleep and GS at the Scan Level

ScanLevelMetrics_df = pd.DataFrame(index=GS_df.columns, columns=['PSDsleep','GSamplitude','Scan Type'])
ScanLevelMetrics_df['GSamplitude'] = GS_df.std().values
ScanLevelMetrics_df['PSDsleep']    = sleep_psd_DF.mean()
ScanLevelMetrics_df.loc[(scans_rank.index.isin(Awake_Runs),'Scan Type')] = 'Awake'
ScanLevelMetrics_df.loc[(scans_rank.index.isin(Drowsy_Runs),'Scan Type')] = 'Drowsy'

ScanLevelMetrics_df.head()

Fig09_panelA = ScanLevelMetrics_df.hvplot.box(y='GSamplitude',by='Scan Type',c='Scan Type', cmap={'Awake':'orange','Drowsy':'lightblue'}, legend=False, title='(A)',fontsize={'title':16,'ylabel':14, 'xlabel':14, 'xticks':14, 'yticks':14}).opts(toolbar=None)
Fig09_panelC = ScanLevelMetrics_df.hvplot.box(y='PSDsleep',by='Scan Type',c='Scan Type', cmap={'Awake':'orange','Drowsy':'lightblue'}, legend=False, title='(C)',fontsize={'title':16,'ylabel':14, 'xlabel':14, 'xticks':14, 'yticks':14}).opts(toolbar=None)

print('++ INFO: Ttest results for GSamplitude')
ttest_ind(ScanLevelMetrics_df['GSamplitude'][ScanLevelMetrics_df['Scan Type']=='Drowsy'],ScanLevelMetrics_df['GSamplitude'][ScanLevelMetrics_df['Scan Type']=='Awake'], alternative='greater')

print('++ INFO: Scan Level Results (PSDsleep)')
print(' +       '+ str(ttest_ind(ScanLevelMetrics_df['PSDsleep'][ScanLevelMetrics_df['Scan Type']=='Drowsy'],ScanLevelMetrics_df['PSDsleep'][ScanLevelMetrics_df['Scan Type']=='Awake'], alternative='greater')))
print('++ INFO: Scan Level Results (GSamplitude)')
print(' +       '+ str(ttest_ind(ScanLevelMetrics_df['GSamplitude'][ScanLevelMetrics_df['Scan Type']=='Drowsy'],ScanLevelMetrics_df['GSamplitude'][ScanLevelMetrics_df['Scan Type']=='Awake'], alternative='greater')))

# # Statistical Differences in PSDsleep and GS at the Segment Level

sleep_psd_DF.index = sleep_psd_DF.index.total_seconds()
EO_segments_info = pd.read_pickle('../Resources/EO_Segments_Info.pkl')
EO_segments_info = EO_segments_info[EO_segments_info['Duration']>=60]
EC_segments_info = pd.read_pickle('../Resources/EC_Segments_Info.pkl')
EC_segments_info = EC_segments_info[EC_segments_info['Duration']>=60]
XX_segments_info = pd.concat([EO_segments_info,EC_segments_info])
XX_segments_info.reset_index(drop=True, inplace=True)

Number_of_Segments = XX_segments_info.shape[0]
SegmentLevelMetrics_df = pd.DataFrame(index=np.arange(Number_of_Segments),columns=['PSDsleep','GSamplitude','Segment Type'])

# + tags=[]
SegmentType_Label_dict = {'EC':'Eyes Closed','EO':'Eyes Open'}
for r,row in XX_segments_info.iterrows():
    if row['Run'] in sleep_psd_DF:
        SegmentLevelMetrics_df.loc[r,'GSamplitude']  = GS_df.loc[int(row['Onset']):int(row['Offset']), row['Run']].std()
        SegmentLevelMetrics_df.loc[r,'PSDsleep']     = sleep_psd_DF.loc[(sleep_psd_DF.index >= row['Onset']) & (sleep_psd_DF.index <= row['Offset']),row['Run']].mean()
        SegmentLevelMetrics_df.loc[r,'Segment Type'] = SegmentType_Label_dict[row['Type']]
    else:
        SegmentLevelMetrics_df.drop(index=r,inplace=True)
        print('++ Warning: PSDsleep not available for %s. Check if you are removing HRa cases' % row['Run'])
SegmentLevelMetrics_df = SegmentLevelMetrics_df.infer_objects()
print(SegmentLevelMetrics_df.shape)
# -

Fig09_panelB = SegmentLevelMetrics_df.hvplot.box(y='GSamplitude',by='Segment Type',c='Segment Type', cmap={'Eyes Open':'orange','Eyes Closed':'lightblue'}, legend=False, title='(B)',fontsize={'title':16,'ylabel':14, 'xlabel':14, 'xticks':14, 'yticks':14}, shared_axes=False).opts(toolbar=None)
Fig09_panelD = SegmentLevelMetrics_df.hvplot.box(y='PSDsleep',   by='Segment Type',c='Segment Type', cmap={'Eyes Open':'orange','Eyes Closed':'lightblue'}, legend=False, title='(D)',fontsize={'title':16,'ylabel':14, 'xlabel':14, 'xticks':14, 'yticks':14}, shared_axes=False).opts(toolbar=None)

print('++ INFO: Segment Level Results (PSDsleep)')
print(' +       '+ str(ttest_ind(SegmentLevelMetrics_df['PSDsleep'][SegmentLevelMetrics_df['Segment Type']=='Eyes Closed'],SegmentLevelMetrics_df['PSDsleep'][SegmentLevelMetrics_df['Segment Type']=='Eyes Open'], alternative='greater')))
print('++ INFO: Segment Level Results (GSamplitude)')
print(' +       '+ str(ttest_ind(SegmentLevelMetrics_df['GSamplitude'][SegmentLevelMetrics_df['Segment Type']=='Eyes Closed'],SegmentLevelMetrics_df['GSamplitude'][SegmentLevelMetrics_df['Segment Type']=='Eyes Open'], alternative='greater')))

Figure09 = pn.Column(pn.Row(Fig09_panelA,Fig09_panelB),
                     pn.Row(Fig09_panelC,Fig09_panelD))

Figure09

Figure09.save('./figures/Fig09_StatDiff_PSDsleep_GSamplitude.{region}.{scenario}.png'.format(region=region, scenario=scenario))


