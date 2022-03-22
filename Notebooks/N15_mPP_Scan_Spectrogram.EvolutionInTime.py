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
remove_HRa_scans  = False      # If true, do analyses after removing scans with overlapping alisaed HR in the sleep band
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

# +
# %%time
if remove_HRa_scans:
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

fig, ax = plt.subplots(2,1,figsize=(30,15))
sns.lineplot(data=df_plot, x='Time [seconds]', hue='Scan Type', y='PSD', style='Frequency Band', style_order=['Sleep (0.03 - 0.07 Hz)','Control (0.1 - 0.2 Hz)'], estimator=np.mean, n_boot=1000, ax=ax[0])
ax[0].set_xlim(df_plot['Time [seconds]'].min(),df_plot['Time [seconds]'].max())
ax[0].set_title('(A) Average PSD in iFV for the two bands of interest')
sns.lineplot(data=df_cum_plot, x='Time [seconds]', hue='Scan Type', y='PSD', style='Frequency Band', style_order=['Sleep (0.03 - 0.07 Hz)','Control (0.1 - 0.2 Hz)'], estimator=np.mean, n_boot=1000, ax=ax[1])
ax[1].set_title('(A) Average Cumulative PSD in iFV for the two bands of interest')
ax[1].set_xlim(df_cum_plot['Time [seconds]'].min(),df_cum_plot['Time [seconds]'].max())
ax[1].legend(ncol=2)

# > **<u>FINDING:</u>**: As scanning progresses, we see a differentiation in terms of instaneous PSD between subjects who kept their eyes open continously and those who did not. The second group tends to have a higher PSD at longer scanning times. Now becuase periods of eye closure do not necessarily align across subjects, the strength observed here is somehow weak.
#
# > **<u>FINDING:</u>**: As scanning progresses, we see a differentiation in terms of cumulative PSD between subjects who kept their eyes open continously and those who did not. The second group tends to have a higher cumulative PSD at longer scanning times. By looking at cumulative PSD we overcome the limitation mentioned above, and the differentiation across groups becomes more evident

fig.savefig('./figures/Revision1_Figure07.{region}.{scenario}.png'.format(region=region,scenario=scenario))


