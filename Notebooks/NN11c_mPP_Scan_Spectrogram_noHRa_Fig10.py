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

region             = 'V4_grp' # Use the Group-Level FV ROI
spectrogram_windur = 60       # Spectrogram Window Duration (In seconds)
Nacq               = 890      # Number of acquisitions

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
# ### Load list of runs classified as awake and drowsy 

# +
# %%time
Run_Lists={}
Run_Lists[('ALL','Manuscript')] = get_available_runs(when='final', type='all')
Run_Lists[('ALL','Awake')]      = get_available_runs(when='final', type='awake')
Run_Lists[('ALL','Drowsy')]     = get_available_runs(when='final', type='drowsy')

scan_HR_info    = pd.read_csv(osp.join(Resources_Dir,'HR_scaninfo.csv'), index_col=0)
scan_HR_info    = scan_HR_info[(scan_HR_info['HR aliased']< 0.03) | (scan_HR_info['HR aliased']> 0.07)]
Run_Lists[('noHRa','Manuscript')] = list(scan_HR_info.index)
Run_Lists[('noHRa','Awake')]      = list(scan_HR_info[scan_HR_info['Scan Type']=='Awake'].index)
Run_Lists[('noHRa','Drowsy')]     = list(scan_HR_info[scan_HR_info['Scan Type']=='Drowsy'].index)

print('++ INFO: Number of Runs (ALL):   Total = %d | Awake = %d | Drowsy = %d' % (len(Run_Lists[('ALL','Manuscript')]),   len(Run_Lists[('ALL','Awake')]),   len(Run_Lists[('ALL','Drowsy')])))
print('++ INFO: Number of Runs (noHRa): Total = %d | Awake = %d | Drowsy = %d' % (len(Run_Lists[('noHRa','Manuscript')]), len(Run_Lists[('noHRa','Awake')]), len(Run_Lists[('noHRa','Drowsy')])))

# + [markdown] tags=[]
# ***
# ### Load PSD in Sleep Band
#
# This data is already the result of a windowing operation (computation of the spectrogram). As such, we ensure the index has the correct values corresponding to windowed information

# +
# %%time
PSDs = {}
PSDs[('ALL','sleep')]     = pd.DataFrame(index=windowed_time_index, columns=Run_Lists[('ALL','Manuscript')])
PSDs[('ALL','control')]   = pd.DataFrame(index=windowed_time_index, columns=Run_Lists[('ALL','Manuscript')])
PSDs[('noHRa','sleep')]   = pd.DataFrame(index=windowed_time_index, columns=Run_Lists[('noHRa','Manuscript')])
PSDs[('noHRa','control')] = pd.DataFrame(index=windowed_time_index, columns=Run_Lists[('noHRa','Manuscript')])

for scenario in ['ALL','noHRa']:
    for sbj_run in Run_Lists[(scenario,'Manuscript')]:
        sbj,run = sbj_run.split('_',1)
        file    = '{RUN}_mPP.Signal.{REGION}.Spectrogram_BandLimited.pkl'.format(RUN=run, REGION=region)
        path    = osp.join(DATA_DIR,sbj,run,file)
        aux     = pd.read_pickle(path)
        PSDs[(scenario,'sleep')].loc[(windowed_time_index,sbj_run)] = aux['sleep'].values
        PSDs[(scenario,'control')].loc[(windowed_time_index,sbj_run)] = aux['control'].values

    PSDs[(scenario,'sleep_cumsum')]   = PSDs[(scenario,'sleep')].cumsum()
    PSDs[(scenario,'control_cumsum')] = PSDs[(scenario,'control')].cumsum()

print("++ INFO: Shape of PSD Dataframe PSDs[('ALL','sleep')]   is %s" % str(PSDs[('ALL','sleep')].shape))
print("++ INFO: Shape of PSD Dataframe PSDs[('noHRa','sleep')] is %s" % str(PSDs[('noHRa','sleep')].shape))

# + [markdown] tags=[]
# ## Load Global Signal
# -

# %%time
GSs = {'ALL':pd.DataFrame(columns=Run_Lists[('ALL','Manuscript')]), 'noHRa':pd.DataFrame(columns=Run_Lists[('noHRa','Manuscript')])}
for scenario in ['ALL','noHRa']:
    for item in Run_Lists[(scenario,'Manuscript')]:
        sbj,run  = item.split('_',1)
        aux_path = osp.join(DATA_DIR,sbj,run,'{RUN}_mPP.Signal.FB.1D'.format(RUN=run))
        aux_data = np.loadtxt(aux_path)
        assert aux_data.shape[0] == Nacq, "{file} has incorrect length {length}".format(file=aux_path, length=str(aux_data.shape[0]))
        GSs[scenario][item] = aux_data

# ***
# # Scan Ranking based on PSDsleep and GS

scan_ranks = {}
for scenario in ['ALL','noHRa']:
    scan_ranks[scenario]                     = pd.DataFrame(columns=['Scan Type'])
    scan_ranks[scenario]['PSDsleep Rank']    = PSDs[(scenario,'sleep')].mean().rank(ascending=False).astype(int)
    scan_ranks[scenario]['PSDsleep']         = PSDs[(scenario,'sleep')].mean()
    scan_ranks[scenario]['Scan Type']        = 'N/A'
    scan_ranks[scenario]['GSamplitude']      = GSs[scenario].std()
    scan_ranks[scenario]['GSamplitude Rank'] = GSs[scenario].std().rank(ascending=False).astype(int)
    scan_ranks[scenario].loc[(scan_ranks[scenario].index.isin(Run_Lists[(scenario,'Awake')]),'Scan Type')] = 'Awake'
    scan_ranks[scenario].loc[(scan_ranks[scenario].index.isin(Run_Lists[(scenario,'Drowsy')]),'Scan Type')] = 'Drowsy'

# # Statistical Differences in PSDsleep and GS at the Scan Level

ScanLevelMetrics_df                = pd.DataFrame(index=Run_Lists[('noHRa','Manuscript')], columns=['PSDsleep','GSamplitude','Scan Type'])
ScanLevelMetrics_df['GSamplitude'] = GSs['noHRa'].std().values
ScanLevelMetrics_df['PSDsleep']    = PSDs[('noHRa','sleep')].mean()
ScanLevelMetrics_df.loc[(scan_ranks['noHRa'].index.isin(Run_Lists[('noHRa','Awake')]) ,'Scan Type')] = 'Awake'
ScanLevelMetrics_df.loc[(scan_ranks['noHRa'].index.isin(Run_Lists[('noHRa','Drowsy')]),'Scan Type')] = 'Drowsy'

ScanLevelMetrics_df

Fig09_panelA = ScanLevelMetrics_df.hvplot.box(y='GSamplitude',by='Scan Type',c='Scan Type', cmap={'Awake':'orange','Drowsy':'lightblue'}, legend=False, title='(A)',fontsize={'title':16,'ylabel':14, 'xlabel':14, 'xticks':14, 'yticks':14}).opts(toolbar=None)
Fig09_panelC = ScanLevelMetrics_df.hvplot.box(y='PSDsleep',by='Scan Type',c='Scan Type', cmap={'Awake':'orange','Drowsy':'lightblue'}, legend=False, title='(C)',fontsize={'title':16,'ylabel':14, 'xlabel':14, 'xticks':14, 'yticks':14}).opts(toolbar=None)

print('++ INFO: Ttest results for GSamplitude')
ttest_ind(ScanLevelMetrics_df['GSamplitude'][ScanLevelMetrics_df['Scan Type']=='Drowsy'],ScanLevelMetrics_df['GSamplitude'][ScanLevelMetrics_df['Scan Type']=='Awake'], alternative='greater')

print('++ INFO: Scan Level Results (PSDsleep)')
print(' +       '+ str(ttest_ind(ScanLevelMetrics_df['PSDsleep'][ScanLevelMetrics_df['Scan Type']=='Drowsy'],ScanLevelMetrics_df['PSDsleep'][ScanLevelMetrics_df['Scan Type']=='Awake'], alternative='greater')))
print('++ INFO: Scan Level Results (GSamplitude)')
print(' +       '+ str(ttest_ind(ScanLevelMetrics_df['GSamplitude'][ScanLevelMetrics_df['Scan Type']=='Drowsy'],ScanLevelMetrics_df['GSamplitude'][ScanLevelMetrics_df['Scan Type']=='Awake'], alternative='greater')))

# # Statistical Differences in PSDsleep and GS at the Segment Level

for scenario in ['ALL','noHRa']:
    PSDs[(scenario,'sleep')].index   = PSDs[(scenario,'sleep')].index.total_seconds()
    PSDs[(scenario,'control')].index = PSDs[(scenario,'control')].index.total_seconds()

XX_segments_info = pd.read_csv(osp.join(Resources_Dir,'HR_segmentinfo.csv'), index_col=0)
XX_segments_info = XX_segments_info[((XX_segments_info['HR aliased']<0.03) | (XX_segments_info['HR aliased']>0.07))]
XX_segments_info.reset_index(drop=True, inplace=True)
XX_segments_info.sort_values(by='Duration')

Number_of_Segments = XX_segments_info.shape[0]
SegmentLevelMetrics_df = pd.DataFrame(index=np.arange(Number_of_Segments),columns=['PSDsleep','GSamplitude','Segment Type'])

SegmentType_Label_dict = {'EC':'Eyes Closed','EO':'Eyes Open'}
for r,row in XX_segments_info.iterrows():
    SegmentLevelMetrics_df.loc[r,'GSamplitude']  = GSs['ALL'].loc[int(row['Onset']):int(row['Offset']), row['Run']].std()
    SegmentLevelMetrics_df.loc[r,'PSDsleep']     = PSDs[('ALL','sleep')].loc[(PSDs[('ALL','sleep')].index >= row['Onset']) & (PSDs[('ALL','sleep')].index <= row['Offset']),row['Run']].mean()
    SegmentLevelMetrics_df.loc[r,'Segment Type'] = SegmentType_Label_dict[row['Type']]
SegmentLevelMetrics_df = SegmentLevelMetrics_df.infer_objects()

Fig09_panelB = SegmentLevelMetrics_df.hvplot.box(y='GSamplitude',by='Segment Type',c='Segment Type', cmap={'Eyes Open':'orange','Eyes Closed':'lightblue'}, legend=False, title='(B)',fontsize={'title':16,'ylabel':14, 'xlabel':14, 'xticks':14, 'yticks':14}, shared_axes=False).opts(toolbar=None)
Fig09_panelD = SegmentLevelMetrics_df.hvplot.box(y='PSDsleep',   by='Segment Type',c='Segment Type', cmap={'Eyes Open':'orange','Eyes Closed':'lightblue'}, legend=False, title='(D)',fontsize={'title':16,'ylabel':14, 'xlabel':14, 'xticks':14, 'yticks':14}, shared_axes=False).opts(toolbar=None)

print('++ INFO: Segment Level Results (PSDsleep)')
print(' +       '+ str(ttest_ind(SegmentLevelMetrics_df['PSDsleep'][SegmentLevelMetrics_df['Segment Type']=='Eyes Closed'],SegmentLevelMetrics_df['PSDsleep'][SegmentLevelMetrics_df['Segment Type']=='Eyes Open'], alternative='greater')))
print('++ INFO: Segment Level Results (GSamplitude)')
print(' +       '+ str(ttest_ind(SegmentLevelMetrics_df['GSamplitude'][SegmentLevelMetrics_df['Segment Type']=='Eyes Closed'],SegmentLevelMetrics_df['GSamplitude'][SegmentLevelMetrics_df['Segment Type']=='Eyes Open'], alternative='greater')))

Figure09 = pn.Column(pn.Row(Fig09_panelA,Fig09_panelB),
                     pn.Row(Fig09_panelC,Fig09_panelD))

Figure09

Figure09.save('./figures/Fig09_StatDiff_PSDsleep_GSamplitude.{region}.noHRa.png'.format(region=region))


