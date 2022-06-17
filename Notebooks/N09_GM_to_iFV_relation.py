# -*- coding: utf-8 -*-
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
# This notebook explores the relationship between the GM global signal and its first derivative to the iFV signal. More particularly, we see if we can reproduce some of the original observations made by Fultz et al. regarding how the signal in the FV relates to the global signal.
#
# Observing similar relationships in our sample, provides additional evidence that we are looking at the same signal despite differences in acquisition

# +
import pandas as pd
import numpy  as np
import xarray as xr
import hvplot.pandas
import os.path as osp
from bokeh.models.formatters import DatetimeTickFormatter
from scipy.stats import wilcoxon, mannwhitneyu, ttest_ind
formatter = DatetimeTickFormatter(minutes = ['%Mmin:%Ssec'])
from scipy.stats import zscore
import matplotlib.pyplot as plt
import holoviews as hv
import seaborn as sns
import panel as pn
import random
from IPython.display import Markdown as md

# %matplotlib inline

# +
from utils.basics    import get_available_runs, load_segments
from utils.variables import DATA_DIR, Resources_Dir
from nilearn.masking import apply_mask

fontsize_opts    = {'title':16,'ylabel':14, 'xlabel':14, 'xticks':14, 'yticks':14,'ticks':12}

# +
pipelines_dict   = {'Minimal':'mPP','Smoothing':'Reference','Smoothing+':'Referencepp','Basic':'BASIC','Basic+':'BASICpp','CompCor':'Behzadi_COMPCOR','CompCor+':'Behzadi_COMPCORpp'}
remove_HRa_scans = False

N_acq           = 890  # Number of volumes
TR              = 1    # Seconds
fmri_time_index = pd.timedelta_range(start='0 s', periods=N_acq, freq='{tr}ms'.format(tr=1000*TR))
# -

# ***
#
# **Load list of scans and segments**

# +
# %%time
if remove_HRa_scans:
    scan_selection  = 'noHRa'
    scan_HR_info    = pd.read_csv(osp.join(Resources_Dir,'HR_scaninfo.csv'), index_col=0)
    scan_HR_info    = scan_HR_info[(scan_HR_info['HR_aliased']< 0.03) | (scan_HR_info['HR_aliased']> 0.07)]
    Manuscript_Runs = list(scan_HR_info.index)
    Awake_Runs      = list(scan_HR_info[scan_HR_info['Scan Type']=='Awake'].index)
    Drowsy_Runs     = list(scan_HR_info[scan_HR_info['Scan Type']=='Drowsy'].index)
else:
    scan_selection  = 'all'
    Manuscript_Runs = get_available_runs(when='final', type='all')
    Awake_Runs      = get_available_runs(when='final', type='awake')
    Drowsy_Runs     = get_available_runs(when='final', type='drowsy')

print('++ INFO: Number of Runs: Total = %d | Awake = %d | Drowsy = %d' % (len(Manuscript_Runs), len(Awake_Runs), len(Drowsy_Runs)))
# -

EC_segments = load_segments('EC',Manuscript_Runs,60)
EO_segments = load_segments('EO',Manuscript_Runs,60)

Ex_segments = pd.concat([EO_segments,EC_segments],axis=0)
print('++ INFO: Ex_segments has shape: %s' % str(Ex_segments.shape))

# ***
# **Load GM for all pipelines**

# %%time
GS = {key:pd.DataFrame(columns=Manuscript_Runs, index=fmri_time_index) for key in pipelines_dict.keys()}
for pipeline in GS.keys():
    for item in Manuscript_Runs:
        sbj,run  = item.split('_',1)
        path     = osp.join(DATA_DIR,sbj,run,'{run}_{pipeline}.Signal.GM.1D'.format(run=run, pipeline=pipelines_dict[pipeline]))
        GS[pipeline][item] = np.loadtxt(path)
    GS[pipeline].columns.name='Scan ID'
    GS[pipeline].name = 'Global Signal'

# **Load inferior FV signal**

# %%time
iFV = pd.DataFrame(columns=Manuscript_Runs, index=fmri_time_index)
for item in Manuscript_Runs:
    sbj,run   = item.split('_',1)
    path      = osp.join(DATA_DIR,sbj,run,'{run}_mPP.det_bpf.Signal.V4lt_grp.1D'.format(run=run))
    iFV[item] = np.loadtxt(path)
iFV.columns.name='Scan ID'
iFV.name = 'iFV Signal'

# **Compute -d/dt(GM) zeroing negative values**
#
# In Fultz et al. the authors state: *"We hypothesized that the BOLD oscillations corresponded to an oscillation in cerebral blood volume and that, because of constant intracranial volume, more CSF flows into the head when less volume is occupied by the blood (25, 26). This hypothesis predicts that the CSF signal should approximately match the negative derivative of the BOLD oscillation, after setting negative values to zero (materials and methods)."*
#
# To test if we can find the same relationship, we now change the sign of the derivative and zero the negative values. We save that version of the signal in GMd_modified_df.

# %%time
GSd = {key:pd.DataFrame(columns=Manuscript_Runs, index=fmri_time_index) for key in pipelines_dict.keys()}
for pipeline in GSd.keys():
    for item in Manuscript_Runs:
        sbj,run  = item.split('_',1)
        path     = osp.join(DATA_DIR,sbj,run,'{run}_{pipeline}.Signal.GM.der.1D'.format(run=run, pipeline=pipelines_dict[pipeline]))
        data     = np.loadtxt(path)
        data     = - data
        data[data<0] = 0
        GSd[pipeline][item] = data
GS[pipeline].columns.name='Scan ID'
GS[pipeline].name = '-d/dt GS'

# ***
#
# ### 1. Amplitude of GM and iFV signal increases duing EC segments
#
# Statement in Fultz et al. *"We observed an increase in BOLD signal amplitude in the cortical graymatter fMRI signal during sleep, as compared with wakefulness (Fig. 3A versus B and C) [mean = 3.28 dB; CI = (0.09, 6.54); P = 0.032, signed-rank test], consistent with previous reports of low-frequency BOLD fluctuations during sleep (9, 10, 24)."*
#
# > **OBSERVATION:** Here we find an equivalent increase in both GM signal and also on the iFV signal when comparing long periods of EC to long periods of EO.
#
# **Compute Amplitude (std across time) for these two signals separately for long EO and EC segments**

# %%time
GM_segment_amp = {key:[] for key in pipelines_dict.keys()}
for pipeline in pipelines_dict.keys():
    for r,row in Ex_segments.iterrows():
        onset, offset, run, suuid, type = int(row['Onset']), int(row['Offset']), row['Run'], row['Segment_UUID'], row['Type']
        data = GS[pipeline].reset_index(drop=True).loc[onset:offset,run]
        GM_segment_amp[pipeline].append(data.std())

# **Compute segment-wise amplitude of the iFV signal**

# %%time
iFV_segment_amp = []
for r,row in Ex_segments.iterrows():
    onset, offset, run, suuid, type = int(row['Onset']), int(row['Offset']), row['Run'], row['Segment_UUID'], row['Type']
    data = iFV.reset_index(drop=True).loc[onset:offset,run]
    iFV_segment_amp.append(data.std())

# **Add information about amplitude to the segment dataframe**

# %%time
Ex_segments['iFV'] = iFV_segment_amp
for pipeline in pipelines_dict.keys():
    Ex_segments[pipeline] = GM_segment_amp[pipeline]

Ex_segments.sample(10)

# **Plot distributions of signals' amplitudes**

segments_cmap  = {'EO':'orange','EC':'lightblue'}

df = Ex_segments.set_index(['Run','Type']).drop(['Segment_UUID','Segment_Index','Onset','Offset','Duration','Scan_Type'],axis=1).melt(ignore_index=False,var_name='Pipeline',value_name='Amplitude').reset_index()

df.sample(5)

# + active=""
# #Example of similar graph with hvplot
# fig    = df.hvplot.box(by=['Pipeline','Type'], color='Type', cmap=segments_cmap,legend=False, fontsize=fontsize_opts, ylabel='Signal Amplitude', width=1000).opts(toolbar=None)
# figure = pn.pane.HoloViews(fig)
# figure.save('./figures/GM_iFV_increase_during_EC.{ss}.png'.format(ss=scan_selection))
# -

fig,ax = plt.subplots(1,1,figsize=(12,5))
sns.set(font_scale=2)
plot = sns.boxplot(data=df,x='Pipeline',hue='Type',y='Amplitude', order=['Minimal','Smoothing','Smoothing+','Basic','Basic+','CompCor','CompCor+'], hue_order=['EC','EO'], width=.3)
plot.set_ylabel('$\mathregular{GM_{amplitude}}$')
plot.xaxis.set_tick_params(rotation=45)

fig,ax = plt.subplots(1,1,figsize=(3,5))
sns.set(font_scale=2)
plot = sns.boxplot(data=df,x='Pipeline',hue='Type',y='Amplitude', order=['iFV'], hue_order=['EC','EO'], width=.3)
plot.set_ylabel('$\mathregular{iFV_{amplitude}}$')
ax.get_legend().remove()
plot.xaxis.set_tick_params(rotation=45)

fig.savefig('./figures/Revision1_Figure8_PanelA.{ss}.png'.format(ss=scan_selection))

stats_table = pd.DataFrame(columns=['Eyes Closed','Eyes Open','Eyes Closed - Eyes Open','T','p','U','p2'], index=pipelines_dict.keys())
stats_table.index.name = 'Preprocessing Pipeline'

for pipeline in pipelines_dict.keys():
    ec           = Ex_segments[Ex_segments['Type']=='EC'][pipeline]
    eo           = Ex_segments[Ex_segments['Type']=='EO'][pipeline]
    stats_table.loc[pipeline,'Eyes Closed'] = '%.2f' % ec.mean()
    stats_table.loc[pipeline,'Eyes Open'] = '%.2f' % eo.mean()
    stats_table.loc[pipeline,'Eyes Closed - Eyes Open'] = '%.2f' % (ec.mean() - eo.mean())
    stats_table.loc[pipeline,'100 * (Eyes Closed - Eyes Open)/Eyes Closed'] = 100 * (ec.mean() - eo.mean()) / ec.mean()
    t_t,t_p      = ttest_ind(ec,eo,alternative='greater')
    stats_table.loc[pipeline,'T'] = '%.2f' % t_t
    stats_table.loc[pipeline,'p'] = '%.1e' % t_p
    mwu_m, mwu_p =  mannwhitneyu(ec,eo,alternative='greater')
    stats_table.loc[pipeline,'U'] = '%2.2e' % mwu_m
    stats_table.loc[pipeline,'p2'] = '%.1e' % mwu_p


pn.pane.DataFrame(stats_table.drop(['Minimal'],axis=0).drop(['U','p2'],axis=1), width=700)

1 - (0.07/0.24)

ec           = Ex_segments[Ex_segments['Type']=='EC']['iFV']
eo           = Ex_segments[Ex_segments['Type']=='EO']['iFV']
t_t,t_p      = ttest_ind(ec,eo,alternative='greater')
mwu_m, mwu_p =  mannwhitneyu(ec,eo,alternative='greater')
print('++ [iFV]:\tT-Test [EC > EO T=%.2f, p=%.2e] | MWU [EC > EO M=%.2f p=%.2e]' % (t_t,t_p,mwu_m,mwu_p))


# ***
#
# ## 2. Relationship between GM, -dGM/dt and iFV: Plot representative Timeseries for long EC segment
#
# Fultz et al. made two additional observations:
#
# * *"Furthermore, the CSF signal was tightly temporally coupled to the cortical gray-matter BOLD oscillation during sleep (Fig. 3, A and B), exhibiting a strong anticorrelation (fig. S3) (maximal r = −0.48 at lag 2 s, P < 0.001, shuffling)."*
#
# * *"Consistent with this hypothesis, the CSF time series and the thresholded derivative BOLD signals were strongly correlated (Fig. 3, D and E) (maximal r = 0.59 at lag −1.8 s; P < 0.001, shuffling)."*

# **Selection of a good segment**

# Let's find a scan with good correlations, and then from those let's find a segment of similar duration to that reported in Fultz et al. so the figures are visualize comparable
sel = GS['Minimal'][Drowsy_Runs].corrwith(iFV[Drowsy_Runs]).sort_values().head(40)
EC_segments[EC_segments['Run'].isin(sel.index)].sort_values(by='Duration',ascending=False)

# **Plot GM, -dGM/dt and iFV for this sample segment**
#
# For comparison purposes, here we also reproduce the figures from Fultz et al.

sample_run_idx = 5843
sample_run    = EC_segments.loc[sample_run_idx,'Run']
sample_onset  = EC_segments.loc[sample_run_idx,'Onset']
sample_offset = EC_segments.loc[sample_run_idx,'Offset']
print('++ Selected Sample: Run = %s | Onset = %d secs | Offset = %d secs' %(sample_run, sample_onset, sample_offset))
sample_run_iFV = pd.DataFrame(columns=[sample_run], index=fmri_time_index)
path           = osp.join(DATA_DIR,sample_run.split('_',1)[0],sample_run.split('_',1)[1],'{run}_mPP.Signal.V4lt_grp.1D'.format(run=sample_run.split('_',1)[1]))
sample_run_iFV[sample_run] = np.loadtxt(path)
sample_run_iFV.columns.name='Scan ID'
sample_run_iFV.name = 'iFV Signal'

# +
font = {'size'   : 14}
sns.set_theme(style='white')
plt.rc('font', **font)
plt.rc('xtick', labelsize=14) 
fig,axs = plt.subplots(2,1,figsize=(20,9))
# Plot GM in green
ax1 = axs[0]
GS['Minimal'][sample_run].reset_index(drop=True).plot(ax=ax1,c='g',lw=2)
ax1.tick_params(axis='y', labelcolor='g')
ax1.set_ylabel('Global Signal',color='g', fontsize=20)
ax1.set_ylim(-2,3)
# Plot iFV in purple
ax2 = ax1.twinx()
ax1.set_xticks([])
sample_run_iFV[sample_run].reset_index(drop=True).plot(ax=ax2,c='m',lw=2)
ax2.set_ylabel('iFV Signal',color='m', fontsize=20)
ax2.tick_params(axis='y', labelcolor='m')
ax1.set_xlim(sample_onset,sample_offset)
ax2.set_ylim(-7,15)
ax1.grid(False)
ax2.grid(False)
ax1.tick_params(axis='both', which='major', labelsize=20)
ax2.tick_params(axis='both', which='major', labelsize=20)

# Plot -dGM/dt in blue
ax3 = axs[1]
GSd['Minimal'][sample_run].reset_index(drop=True).plot(ax=ax3,c='b',lw=2)
ax3.tick_params(axis='y', labelcolor='b')
ax3.set_ylabel('-d/dt Global Signal',color='b', fontsize=20)
# Plot iFV in purple
ax4 = ax3.twinx()
sample_run_iFV[sample_run].reset_index(drop=True).plot(ax=ax4,c='m',lw=2)
ax4.set_ylabel('iFV Signal',color='m', fontsize=20)
ax4.tick_params(axis='y', labelcolor='m')
ax3.set_xlim(sample_onset,sample_offset)
ax4.set_ylim(-7,15)
#ax1.set_title('(A) GM Signal and iFV Signal for a Drowsy Scan', loc='left')
#ax3.set_title('(B) -d/dt GM Signal and iFV Signal for a Drowsy Scan', loc='left')
ax3.set_xlabel('Time [seconds]', fontsize=20)
ax3.grid(False)
ax4.grid(False)
ax3.tick_params(axis='both', which='major', labelsize=20)
ax4.tick_params(axis='both', which='major', labelsize=20)
#plt.close()
# -

fig.savefig('./figures/Revision1_Figure8_PanelsBC.{ss}.png'.format(ss=scan_selection))

combined_figure = pn.Row(pn.Column(pn.pane.PNG('./figures/Fultz_Fig3B_Timeseres_GM_vs_iFV.png', height=200),
                  pn.pane.PNG('./figures/Fultz_Fig3E_Timeseries_dGM_vs_iFV.png', height=200)),
                  pn.pane.PNG('./figures/Revision1_Figure8_PanelsBC.{ss}.png'.format(ss=scan_selection), height=450))

combined_figure.save('./figures/GM_dGM_iFV_RepTS_both_papers.png')

text="![](./figures/GM_dGM_iFV_RepTS_both_papers.png)"
md("%s"%(text))

# **Cross-correlation between GM, -dGM/dt and iFV**

# Work only with lags -20s to 20s (as in Fultz et al.)
lags = np.arange(41)-20
def crosscorr(datax, datay, lag=0):
    """ Lag-N cross correlation. 
    Parameters
    ----------
    lag : int, default 0
    datax, datay : pandas.Series objects of equal length

    Returns
    ----------
    crosscorr : float
    """
    return datax.corr(datay.shift(lag))


# %%time
segments_xcorr = pd.DataFrame(index=pd.MultiIndex.from_product((EC_segments['Segment_UUID'].values,lags), names=['Segment_UUID','Lag']), columns=['xcorr_dGM_iFV','xcorr_GM_iFV'])
data = {}
for r,row in EC_segments.iterrows():
    aux_onset, aux_offset, aux_run, aux_suuid = int(row['Onset']), int(row['Offset']), row['Run'], row['Segment_UUID']
    data[r] = {}
    data[r]['dGM'] = GSd['Minimal'].reset_index(drop=True).loc[aux_onset:aux_offset,aux_run]
    data[r]['GM']  = GS['Minimal'].reset_index(drop=True).loc[aux_onset:aux_offset,aux_run]
    data[r]['iFV'] = iFV.reset_index(drop=True).loc[aux_onset:aux_offset,aux_run]
    # Case A: -d/dt GM & iFV
    for lag in lags:
        segments_xcorr.loc[(aux_suuid, lag),'xcorr_dGM_iFV'] = crosscorr(data[r]['dGM'], data[r]['iFV'], lag=lag)
    # Case B: GM & iFV
    for lag in lags:
        segments_xcorr.loc[(aux_suuid, lag),'xcorr_GM_iFV'] = crosscorr(data[r]['GM'], data[r]['iFV'], lag=lag)
segments_xcorr = segments_xcorr.reset_index()
segments_xcorr = segments_xcorr.infer_objects()

# **Shuffled version for reference confidence interval in plots**

# %%time
num_shuffles = 1000
shuffled_segments_xcorr = pd.DataFrame(index=pd.MultiIndex.from_product((np.arange(0,num_shuffles),lags),names=['Shuffle_Num','Lag']),  columns=['xcorr_dGM_iFV','xcorr_GM_iFV'])
for shuffle_iter in np.arange(0,num_shuffles):
    x_idx = EC_segments.iloc[random.randint(0,EC_segments.shape[0]-1)].name
    y_idx = EC_segments.iloc[random.randint(0,EC_segments.shape[0]-1)].name
    GM_signal  = data[x_idx]['GM'].reset_index(drop=True)[0:59]
    dGM_signal = data[x_idx]['dGM'].reset_index(drop=True)[0:59]
    iFV_signal = data[y_idx]['iFV'].reset_index(drop=True)[0:59]
    for lag in lags:
        shuffled_segments_xcorr.loc[(shuffle_iter, lag),'xcorr_dGM_iFV'] = crosscorr(dGM_signal, iFV_signal, lag=lag)
        shuffled_segments_xcorr.loc[(shuffle_iter, lag),'xcorr_GM_iFV']  = crosscorr(GM_signal, iFV_signal, lag=lag)
shuffled_segments_xcorr = shuffled_segments_xcorr.infer_objects()

max_lags_GM_iFV, max_lags_dGM_iFV = [], []
for shuffle_iter in np.arange(0,num_shuffles):
    max_lags_GM_iFV.append(shuffled_segments_xcorr.loc[shuffle_iter,:].abs().idxmax()['xcorr_GM_iFV'])
    max_lags_dGM_iFV.append(shuffled_segments_xcorr.loc[shuffle_iter,:].abs().idxmax()['xcorr_dGM_iFV'])

# **BOLD-CSF Cross Correlation Plots**

sns.set_theme(style='white')
fig, axs = plt.subplots(1,1,figsize=(4,6))
sns.lineplot(data=shuffled_segments_xcorr,x='Lag',y='xcorr_GM_iFV',color='w', lw=0, ax=axs, ci=95, err_kws={'linestyle':'dashed','linewidth':2,'facecolor':'w','edgecolor':'k', 'alpha':1})
xcorr_plot_GM2iFV = sns.lineplot(data=segments_xcorr,x='Lag',y='xcorr_GM_iFV', ax=axs, ci=95, color='darkblue')
#xcorr_plot_GM2iFV.grid()
xcorr_plot_GM2iFV.set_xlim(-20,20)
xcorr_plot_GM2iFV.set_xlabel('Lag (s)', fontsize=16)
xcorr_plot_GM2iFV.set_ylabel('R(GM,iFV)', fontsize=16)
#xcorr_plot_GM2iFV.set_title('BOLD-CSF cross-correlation\n EC segments', fontsize=16)
axs.tick_params(axis='both', which='major', labelsize=16)
fig.savefig('./figures/Xcorr_GMvsiFV.{ss}.png'.format(ss=scan_selection))
idx_max_corr = segments_xcorr.groupby('Lag').mean().abs().idxmax()['xcorr_GM_iFV']
max_corr     = segments_xcorr.groupby('Lag').mean().loc[idx_max_corr]['xcorr_GM_iFV']
print('++ INFO: Maximal X Correlation Delay is %d seconds [%.2f]' % (idx_max_corr,max_corr))
fig.savefig('./figures/Revision1_Figure8_PanelD.{ss}.png'.format(ss=scan_selection))
plt.close()

combined_figure = pn.Row(pn.pane.PNG('./figures/Fultz_FigS3_Xcorr_GM_vs_iFV.png'),
       pn.pane.PNG('./figures/Revision1_Figure8_PanelD.{ss}.png'.format(ss=scan_selection),height=390))
combined_figure.save('./figures/Xcorr_GMvsiFV_both_papers.{ss}.png'.format(ss=scan_selection))

text="![](./figures/Xcorr_GMvsiFV_both_papers.{ss}.png)".format(ss=scan_selection)
md("%s"%(text))

fig, axs = plt.subplots(1,1,figsize=(4,6))
sns.lineplot(data=shuffled_segments_xcorr,x='Lag',y='xcorr_dGM_iFV',color='w', lw=0, ax=axs, ci=95, err_kws={'linestyle':'dashed','linewidth':2,'facecolor':'w','edgecolor':'k', 'alpha':1})
xcorr_plot_GM2iFV = sns.lineplot(data=segments_xcorr,x='Lag',y='xcorr_dGM_iFV', ax=axs, ci=95, color='darkblue')
#xcorr_plot_GM2iFV.grid()
xcorr_plot_GM2iFV.set_xlim(-20,20)
xcorr_plot_GM2iFV.set_xlabel('Lag (s)', fontsize=16)
xcorr_plot_GM2iFV.set_ylabel('R(-dGM/dt,iFV)', fontsize=16)
axs.tick_params(axis='both', which='major', labelsize=16)
#xcorr_plot_GM2iFV.set_title('-d/dt GM to CSF cross-correlation\n EC segments', fontsize=16)
fig.savefig('./figures/Xcorr_dGMvsiFV.{ss}.png'.format(ss=scan_selection))
idx_max_corr = segments_xcorr.groupby('Lag').mean().abs().idxmax()['xcorr_dGM_iFV']
max_corr     = segments_xcorr.groupby('Lag').mean().loc[idx_max_corr]['xcorr_dGM_iFV']
print('++ INFO: Maximal X Correlation Delay is %d seconds [%.2f]' % (idx_max_corr,max_corr))
fig.savefig('./figures/Revision1_Figure8_PanelE.{ss}.png'.format(ss=scan_selection))
plt.close()

combined_figure = pn.Row(pn.pane.PNG('./figures/Fultz_Fig3D_Xcorr_dGM_vs_iFV.png'),
       pn.pane.PNG('./figures/Revision1_Figure8_PanelE.{ss}.png'.format(ss=scan_selection),height=390))
combined_figure.save('./figures/Xcorr_dGMvsiFV_both_papers.{ss}.png'.format(ss=scan_selection))

text="![](./figures/Xcorr_dGMvsiFV_both_papers.{ss}.png)".format(ss=scan_selection)
md("%s"%(text))

print('++ INFO: Maximal X Correlation Delay is %d seconds' % segments_xcorr.groupby('Lag').mean().abs().idxmax()['xcorr_dGM_iFV'])

# ***
# ## 3. GS before and after removal of lagregressors - scan-level
#
# For us it is not only about verification of the signal properties, but also about what happens when it gets removed. The following analyses look at how the global signal changes changes with the different regression schemes inspected in this work.

# +
# %%time
df = {'All':   pd.DataFrame(index=pipelines_dict.keys(), columns=pipelines_dict.keys()),
      'Drowsy':pd.DataFrame(index=pipelines_dict.keys(), columns=pipelines_dict.keys()),
      'Awake': pd.DataFrame(index=pipelines_dict.keys(), columns=pipelines_dict.keys())}

for scan_selection,scan_list in zip(['All','Drowsy','Awake'],[Manuscript_Runs,Drowsy_Runs,Awake_Runs]):
    for x in df[scan_selection].index:
        for y in df[scan_selection].columns:
            if x == y:
                df[scan_selection].loc[x,y] = 1
            else:
                df[scan_selection].loc[x,y] = GS[x][scan_list].corrwith(GS[y][scan_list]).median()
# -

value_dimension             = hv.Dimension('value', value_format=lambda x: '%.2f' % x)
matrix_plot_all             = df['All'].hvplot.heatmap(title='R(GM,GM) - All Scans', aspect='square', fontsize=fontsize_opts, cmap='RdBu_r').opts(toolbar=None,xrotation=45, frame_width=400)
labelled_matrix_plot_all    = matrix_plot_all * hv.Labels(matrix_plot_all, vdims=value_dimension)
matrix_plot_awake           = df['Awake'].hvplot.heatmap(title='R(GM,GM) - Awake Scans', aspect='square', fontsize=fontsize_opts, cmap='RdBu_r').opts(toolbar=None,xrotation=45, frame_width=400)
labelled_matrix_plot_awake  = matrix_plot_awake * hv.Labels(matrix_plot_awake, vdims=value_dimension)
matrix_plot_drowsy          = df['Drowsy'].hvplot.heatmap(title='R(GM,GM) - Drowsy Scans', aspect='square', fontsize=fontsize_opts, cmap='RdBu_r').opts(toolbar=None,xrotation=45, frame_width=400)
labelled_matrix_plot_drowsy = matrix_plot_drowsy * hv.Labels(matrix_plot_drowsy, vdims=value_dimension)
all_plots = labelled_matrix_plot_all + labelled_matrix_plot_awake + labelled_matrix_plot_drowsy
pn.pane.HoloViews(all_plots).save('./figures/Rgm_to_gm_across_regressions.{ss}.png'.format(ss=scan_selection))

text="![](./figures/Rgm_to_gm_across_regressions.{ss}.png)".format(ss=scan_selection)
md("%s"%(text))

# %%time
xr_segs = {'EC':  xr.DataArray(dims=['rs_x','rs_y','segment'],
                  coords={'rs_x':list(pipelines_dict.keys()),'rs_y':list(pipelines_dict.keys()),'segment':Ex_segments[Ex_segments['Type']=='EC']['Segment_UUID']}),
           'EO':  xr.DataArray(dims=['rs_x','rs_y','segment'],
                  coords={'rs_x':list(pipelines_dict.keys()),'rs_y':list(pipelines_dict.keys()),'segment':Ex_segments[Ex_segments['Type']=='EO']['Segment_UUID']})}

# %%time
for segment_type in ['EO','EC']:
    for r,row in Ex_segments[Ex_segments['Type']==segment_type].iterrows():
        aux_onset, aux_offset, aux_item, aux_suuid, aux_type = int(row['Onset']), int(row['Offset']), row['Run'], row['Segment_UUID'], row['Type']
        sbj,run = aux_item.split('_',1)
        for rs_x in pipelines_dict.keys():
            for rs_y in pipelines_dict.keys():
                aux_x_path = osp.join(DATA_DIR,sbj,run,'{run}_{rs_x}.Signal.GM.1D'.format(run=run,rs_x=pipelines_dict[rs_x])) 
                aux_y_path = osp.join(DATA_DIR,sbj,run,'{run}_{rs_y}.Signal.GM.1D'.format(run=run,rs_y=pipelines_dict[rs_y]))
                aux_x      = np.loadtxt(aux_x_path)[aux_onset:aux_offset]
                aux_y      = np.loadtxt(aux_y_path)[aux_onset:aux_offset]
                xr_segs[segment_type].loc[rs_x,rs_y,aux_suuid] = np.corrcoef(aux_x,aux_y)[0,1]

matrix_plot_ec          = xr_segs['EC'].mean(axis=2).to_pandas().hvplot.heatmap(title='R(GM,GM) - EC Segments', aspect='square', fontsize=fontsize_opts, cmap='RdBu_r').opts(toolbar=None,xrotation=45, frame_width=400)
labelled_matrix_plot_ec = matrix_plot_ec * hv.Labels(matrix_plot_ec, vdims=value_dimension)
matrix_plot_eo          = xr_segs['EO'].mean(axis=2).to_pandas().hvplot.heatmap(title='R(GM,GM) - EO Segments', aspect='square', fontsize=fontsize_opts, cmap='RdBu_r').opts(toolbar=None,xrotation=45, frame_width=400)
labelled_matrix_plot_eo = matrix_plot_eo * hv.Labels(matrix_plot_eo, vdims=value_dimension)
all_plots = labelled_matrix_plot_eo + labelled_matrix_plot_ec
pn.pane.HoloViews(all_plots).save('./figures/Rgm_to_gm_across_regressions_segments.{ss}.png'.format(ss=scan_selection))

text="![](./figures/Rgm_to_gm_across_regressions_segments.{ss}.png)".format(ss=scan_selection)
md("%s"%(text))


