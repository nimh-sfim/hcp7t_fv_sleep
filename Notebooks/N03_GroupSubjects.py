# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.9.1
#   kernelspec:
#     display_name: hcp7t_fv_sleep_env
#     language: python
#     name: hcp7t_fv_sleep_env
# ---

# # Description
#
# This notebook performs the following analyes:
#
# 1. Label runs as drowsy or awake based on the amount of time subjects closed their eyes during a given run
# 2. Checks there is no statistical difference in motion (mean FD) between both run types
# 3. Identifies the onset, offset and duration of every segment of eye closure and eye opening
# 4. Generates manuscript figures: 1 and 4.
#
# ### Outputs:
#
# * ```./Resources/Run_List_All.txt```: list of all runs that passed QA tests.
# * ```./Resources/Run_List_Awake.txt```: list of runs marked as awake.
# * ```./Resources/Run_List_Drowsy.txt```: list of runs marked as drowsy.
# * ```./Resources/Run_List_Discarded.txt```: list of runs marked as discarded.
# * ```./Resources/EC_Segments_Info.plk```: dataframe with information about each EC segment in the data (run, onset, offset, duration, etc.).
# * ```./Resources/EO_Segments_Info.plk```: dataframe with information about each EO segment in the data (run, onset, offset, duration, etc.).
#
# ### Manuscript Figures:
# * ```./Notebooks/figures/Figure01_ScanGroups.png```: main figure 1 in manuscript.
# * ```./Notebooks/figures/Figure04_runsECperTR.png```: main figure 4 in manuscript.
#
# ### Extra Figures:
#
# * ```./Notebooks/images/extramaterials01_segments_per_run.png```: un-published figure. bar plot of segment durations per run
# * ```./Notebooks/images/extramaterials02_ECdistribution.png```:  un-published figure. distribution of EC durations in relation to 60s threshold
# * ```./Notebooks/images/extramaterials03_EOdistribution```: un-published figure. distribution of EO durations in relation to 60s threshold

# +
import os
import pandas as pd
import numpy  as np
import hvplot.pandas
import matplotlib.pyplot as plt
import os.path as osp
import holoviews as hv
import seaborn as sns
hv.extension('bokeh')
from scipy.stats import pearsonr
from utils.variables import DATA_DIR, Resources_Dir, PercentWins_EC_Awake, PercentWins_EC_Sleep
import panel as pn
import uuid
import random
from scipy.stats import ttest_ind, mannwhitneyu

sns.set(font_scale=2)
# -

# ***
# # 1. Read Fully Pre-processed ET data into memory

ET_PupilSize_Proc_1Hz = pd.read_pickle(osp.join(Resources_Dir,'ET_PupilSize_Proc_1Hz_corrected.pkl'))
[Nacq, Nruns]         = ET_PupilSize_Proc_1Hz.shape
print("++ INFO: Number of available runs:  %d" % Nruns)
print("++ INFO: Number of volumes per run: %d" % Nacq)

# ***
# # 2. Separate runs into two groups: Awake OR Drowsy
#
# * **AWAKE RUNS**:  Those during which subjects kept their **eyes closed for less than 5%** of the time points in a given run.
# * **DROWSY RUNS**: Those during which subjects kept their **eyes closed somewhere between 25% and 75%** of the run.

Run_Lists = {}
Run_Lists['All'] = ET_PupilSize_Proc_1Hz.columns

# First, we find the **AWAKE RUNS**

aux                 = (ET_PupilSize_Proc_1Hz.isna().sum() <= Nacq * PercentWins_EC_Awake)
Run_Lists['Awake']  = list(aux[aux==True].index)
Nruns_awake         = len(Run_Lists['Awake'])
del aux
print('++ INFO: Awake [Num Runs = %d, Percentage = %0.2f %%' % (Nruns_awake,(100*Nruns_awake)/Nruns))

# Next, we find the **DROWSY RUNS**

a                   = (ET_PupilSize_Proc_1Hz.isna().sum() >= Nacq * PercentWins_EC_Sleep[0])
b                   = (ET_PupilSize_Proc_1Hz.isna().sum() <= Nacq * PercentWins_EC_Sleep[1]) 
c                   = a & b
Run_Lists['Drowsy'] = list(c[c==True].index)
Nruns_drowsy        = len(Run_Lists['Drowsy'])
del a,b,c
print('++ INFO: Drowsy [Num Runs = %d, Percentage = %0.2f %%' % (Nruns_drowsy,(100*Nruns_drowsy/Nruns)))

# Finally, we also identify runs that will be discarded becuase they don't fall within the definition of AWAKE or DROWSY

Run_Lists['Discarded'] = [r for r in Run_Lists['All'] if r not in Run_Lists['Drowsy']+Run_Lists['Awake']]
Nruns_discard          = len(Run_Lists['Discarded'])
print('++ INFO: Discarded [Num Runs = %d, Percentage = %0.2f %%' % (Nruns_discard,(100*Nruns_discard/Nruns)))

# Finally, we create a new list that contains only the runs that will be part of this work, namely those classified as either drwosy or awake

Manuscript_Runs = Run_Lists['Awake'] + Run_Lists['Drowsy']
print("++ INFO: Final list contains %d runs" % len(Manuscript_Runs))

# Save the different run lists to disk (in the Resources folder) so we can easily load them in subsequent notebooks

for key in Run_Lists:
    data = Run_Lists[key]
    path = osp.join(Resources_Dir,'Run_List_{key}.txt'.format(key=key))
    with open(path,'w') as fid:
        for item in data:
            fid.write("%s\n" % item)
    print('++ INFO: Run list saved to disk [%s]' % path)

# ***
# # 2. Load Framewise Displacement & Check for Signficant differences in motion between Awake and Drowsy

# %%time 
# Load FD traces per run
fd = pd.DataFrame(index=np.arange(Nacq))
for item in Manuscript_Runs:
    sbj,run = item.split('_',1)
    fd_path = osp.join(DATA_DIR,sbj,run,run+'_Movement_FD.txt')
    fd[item]= pd.read_csv(fd_path, sep='\t', index_col=0)
# Compute Mean FD per Run   
mean_fd = pd.DataFrame(fd.mean(), columns=['Mean_FD'])
# Add column with run type
mean_fd['Scan Type'] = 'N/A'
mean_fd.loc[(Run_Lists['Awake'],'Scan Type')]  = 'Awake'
mean_fd.loc[(Run_Lists['Drowsy'],'Scan Type')] = 'Drowsy'
mean_fd = mean_fd.reset_index(drop=True)

# Check for significant difference in Mean FD across run types using T-test

ttest_ind(mean_fd[mean_fd['Scan Type']=='Awake']['Mean_FD'],mean_fd[mean_fd['Scan Type']=='Drowsy']['Mean_FD'])

# Check for significant difference in Mean FD across run types using the Mann Whitney Test

mannwhitneyu(mean_fd[mean_fd['Scan Type']=='Awake']['Mean_FD'],mean_fd[mean_fd['Scan Type']=='Drowsy']['Mean_FD'], alternative='two-sided')

# ***
# # 3. Create Figure 1

sample_drowsy = '680957_rfMRI_REST2_AP'  #random.sample(Run_Lists[('B','Drowsy')],1)
sample_awake  = '365343_rfMRI_REST2_AP'  #random.sample(Run_Lists[('B','Awake')],1)

# Calculate the percent of missing samples per run
Percent_Missing_Samples = (100*ET_PupilSize_Proc_1Hz.isna().sum()/ET_PupilSize_Proc_1Hz.shape[0])

FIG1_PANELA = hv.Rectangles([(0, 0, 100*PercentWins_EC_Awake, 120, 1), (100*PercentWins_EC_Sleep[0], 0, 100*PercentWins_EC_Sleep[1], 120, 2)], 
                         vdims='value').opts(alpha=0.5, color='value', cmap=['lightblue','orange'], toolbar=None) * \
           Percent_Missing_Samples.hvplot.hist(bins=100, xlim=(0,99), ylim=(0,120), 
                                               title='(A) Scan labeling based on total duration of eye closure', 
                                               xlabel='Percentage of Eye Closure Duration', ylabel='Number of Scans') * \
           hv.Arrow(20,60,'           Drowsy Scans ',) * \
           hv.Arrow(5, 80,' Awake Scans ' , ) * \
           hv.Arrow(90, 60,' ' , direction='>' )

FIG1_PANELB = mean_fd.hvplot.kde(by='Scan Type',color=['lightblue','orange'], legend='top_right', title='(B) Distribution of Mean Framewise Displacement for "awake" and "drowsy" scans', xlabel='Mean Framewise Displacement / Head Motion').opts(toolbar=None)

ET_PupilSize_Proc_1Hz_COPY = ET_PupilSize_Proc_1Hz.copy()
ET_PupilSize_Proc_1Hz_COPY.index = ET_PupilSize_Proc_1Hz_COPY.index.total_seconds()
FIG1_PANELSCD = (ET_PupilSize_Proc_1Hz_COPY[sample_awake].hvplot(width=1400, height=200, 
                                                             ylabel='Pupil Size', 
                                                             title='(C) Representative Pupil Size Trace for one scan labeled "Awake"',) + \
             ET_PupilSize_Proc_1Hz_COPY[sample_drowsy].hvplot(width=1400, height=200, 
                                                              ylabel='Pupil Size', 
                                                              title='(D) Representative Pupil Size Trace for one scan labeled "Drowsy"', 
                                                              xlabel='Time [seconds]', color='orange')).cols(1).opts(toolbar=None)

# > NOTE: Many figures on this project were generated with holoviz libraries that create dynamic plots. Such dynamic plots are not visible in github. To go around that issue, we also generate static versions of the figures using the save function. This save function may require that you install firefox in your jyputer environment. Also, if you rather see the dynamic version of the figure, simply create a new cell with ```FIGURE1``` and run that cell

FIGURE1 = pn.Column(pn.Row(FIG1_PANELA,FIG1_PANELB),FIG1_PANELSCD)

FIGURE1.save('./figures/Figure01_ScanGroups.png')

# <img src='./figures/Figure01_ScanGroups.png'>

# ***
# # 4. Create Figure 4
#
# Although we expect subjects to close their eyes at different times, previous research suggests that we should still see a trend in which as scanning progresses, a larger number of subjects will have their eyes closed at a given TR. This is the result of more, and more subjects falling sleep as scanning progresses.
#
# Figure 4 shows that such is indeed the case for the 7T HCP resting-state sample
#
# > At this point we remove from the memory copy of ET_PupilSize_Proc_1Hz any run that does not fall under the definition of "awake" or "drowsy"

ET_PupilSize_Proc_1Hz = ET_PupilSize_Proc_1Hz[Run_Lists['Awake']+Run_Lists['Drowsy']]
ET_PupilSize_Proc_1Hz.shape

# +
sns.set(font_scale=1.2, style='whitegrid', font='sans-serif')

# Compute the percent of runs with eyes closed per TR

FIGURE4, ax        = plt.subplots(1,1,figsize=(10,5))
df             = pd.DataFrame(100*ET_PupilSize_Proc_1Hz.isna().sum(axis=1)/ET_PupilSize_Proc_1Hz.shape[1], columns=['% Runs with eyes closed'])
df['Time [s]'] = df.index.total_seconds()
# Plot the results as a scatter plot and also a linear fit
sns.regplot(data=df,x='Time [s]',y='% Runs with eyes closed', scatter_kws={'color':'k'}, line_kws={'color':'k'})
ax.set_ylim(0,60)
# Compute correlation for the linear fit
r,p = pearsonr(df['Time [s]'],df['% Runs with eyes closed'])
# Add correlation value as an annotation to the figure
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
plt.text(705, 25,'R = %0.2f' %(r), bbox=props)
# Add title to figure
ax.set_title('Percentage of Runs with Eyes Closed at a given TR', fontdict={'size':22})
# Save figure to disk
FIGURE4.savefig('./figures/Figure04_runsECperTR.png')
# Remove temporary variables
del df
# -

print('++ INFO: Correlation between scan time & subjects with eyes closed: R = %0.2f (p = %f)' %(r,p))
FIGURE4

# ***
# # 5. Identify individual periods of EO and EC
#
# First, we create a dictionary with two empty dataframes (one for EC segments, one for EO segments). 
#
# For each segment of each type, there will be an corresponding entry (row) in the appropriate dataframe.
#
# That row will contain: 
# * Run: run name containing the segment 
# * Type: EC or EO segment
# * Segment_Index: integer index for this segment entry (integer)
# * Segment_UUID: segment unique identifier. Used for naming files
# * Onset: segment onset TR (in seconds)
# * Offset: segment offset TR (in seconds)
# * Duration:segment duration (in seconds)
# * Scan_Type: awake or drowsy

Scan_Segments = {'EC': pd.DataFrame(columns=['Run','Type','Segment_Index','Segment_UUID','Onset','Offset','Duration','Scan_Type']),
                 'EO': pd.DataFrame(columns=['Run','Type','Segment_Index','Segment_UUID','Onset','Offset','Duration','Scan_Type'])}

# Second, we identify all EC segmenst and add their information to the Scan_Segments dictionary

# %%time
print ('++ INFO: List of runs with no EC segments')
for run in Run_Lists['Awake']+Run_Lists['Drowsy']:
    df         = pd.DataFrame(ET_PupilSize_Proc_1Hz[run].copy())
    df.index   = df.index.total_seconds()
    df.columns = ['data']
    ec_durs    = df.data.isnull().astype(int).groupby(df.data.notnull().astype(int).cumsum()).sum()
    ec_onsets  = df.data.isnull().astype(int).groupby(df.data.notnull().astype(int).cumsum()).cumsum()
    ec_onsets  = ec_onsets[ec_onsets== 1].index.values
    ec_durs    = ec_durs[ec_durs>0].values
    ec_idx     = np.arange(ec_durs.shape[0])
    if run in Run_Lists['Awake']:
        scan_type = 'Awake'
    else:
        scan_type = 'Drowsy'
        
    if ec_onsets.shape[0] == 0:
        # I need to keep track of those entries for the plots underneath (otherwise there is no space for the ones with no segments)
        print('%s |' % (run), end='')
        Scan_Segments['EC'] = Scan_Segments['EC'].append({'Run':run,'Type':'XX','Segment_Index': np.nan, 
                                                          'Segment_UUID':np.nan, 'Onset':np.nan, 'Offset':np.nan,'Duration':np.nan,'Scan_Type':scan_type}, ignore_index=True)
    else:
        for onset,dur,idx in zip(ec_onsets,ec_durs,ec_idx):
            segment_uuid = uuid.uuid4()
            Scan_Segments['EC'] = Scan_Segments['EC'].append({'Run':run,'Type':'EC','Segment_Index':idx ,
                                                              'Segment_UUID':segment_uuid, 'Onset':onset, 'Offset':onset + dur,'Duration':dur,'Scan_Type':scan_type}, ignore_index=True)
    del df, ec_durs, ec_onsets
Scan_Segments['EC'] = Scan_Segments['EC'].infer_objects()

# Third, we identify all EO segmenst and add their information to the Scan_Segments dictionary

# +
# %%time
print ('++ INFO: List of runs with no EO segments')
for run in Run_Lists['Awake']+Run_Lists['Drowsy']:
    df         = pd.DataFrame(ET_PupilSize_Proc_1Hz[run].copy())
    df.index   = df.index.total_seconds()
    df.columns = ['data']
    eo_durs   = df.data.notnull().astype(int).groupby(df.data.isnull().astype(int).cumsum()).sum()
    eo_onsets = df.data.notnull().astype(int).groupby(df.data.isnull().astype(int).cumsum()).cumsum()
    eo_onsets = eo_onsets[eo_onsets== 1].index.values
    eo_durs   = eo_durs[eo_durs>0].values
    eo_idx    = np.arange(eo_durs.shape[0])
    if run in Run_Lists['Awake']:
        scan_type = 'Awake'
    else:
        scan_type = 'Drowsy'
    if eo_onsets.shape[0] == 0:
        print('%s |' % (run),end='')
        Scan_Segments['EO'] = Scan_Segments['EO'].append({'Run':run,'Type':'XX','Segment_Index': np.nan,'Segment_UUID':np.nan, 
                                                          'Onset':np.nan, 'Offset':np.nan,'Duration':np.nan,'Scan_Type':scan_type}, ignore_index=True)
    else:
        for onset,dur,idx in zip(eo_onsets,eo_durs,eo_idx):
            segment_uuid = uuid.uuid4()
            Scan_Segments['EO'] = Scan_Segments['EO'].append({'Run':run,'Type':'EO','Segment_Index':idx ,'Segment_UUID':segment_uuid, 
                                                              'Onset':onset, 'Offset':onset + dur,'Duration':dur,'Scan_Type':scan_type}, ignore_index=True)
    del df, eo_durs, eo_onsets

Scan_Segments['EO'] = Scan_Segments['EO'].infer_objects()
# -

# Finally, we save the information about scan segments to disk

# Save results to disk
# ====================
Scan_Segments['EC'].to_pickle(osp.join(Resources_Dir,'EC_Segments_Info.pkl'))
Scan_Segments['EO'].to_pickle(osp.join(Resources_Dir,'EO_Segments_Info.pkl'))

# ***
#
# # 6. Additional Visualizations not included in manuscript

# I need to take run and period index and make them into a multiindex Index so that then I can plot stacked bars per run to show both the duration of individual
# EC periods, and the total duration of eye closure for a given run
# ==============================================================================================================================================================
Scan_Segments_table = {'EC':Scan_Segments['EC'].copy().set_index(['Run','Segment_Index','Scan_Type']),
                       'EO':Scan_Segments['EO'].copy().set_index(['Run','Segment_Index','Scan_Type'])}

# ### 6.1. Total duration of EC and EO per scan separated by scan type
#
# In the following figure, each scan is represented by one vertical bar. The total highth of the bar will be the total length of EC (top panel) or EO (bottom panel). Runs are sorted so that awake runs are shown on the left of the figure (light blue color) and drowsy scans on the right (orange color). 
#
# Each bar is then subdivided into shorter bars. Each of these shorter/stacked bars corresponds to each individual segment marked as EC or EO.

TOP_PANEL = Scan_Segments_table['EC'].loc[:,:,'Awake'].hvplot.bar(stacked=True, y='Duration', by='Segment_Index', color=['lightblue'], legend=False).opts(width=1500, title='Eye Closure Segments (Awake Scans on Left | Drowsy Scans on Right)') * \
            Scan_Segments_table['EC'].loc[:,:,'Drowsy'].hvplot.bar(stacked=True, y='Duration', by='Segment_Index', color=['orange'], legend=False).opts(width=1500)
BOT_PANEL = Scan_Segments_table['EO'].loc[:,:,'Awake'].hvplot.bar(stacked=True, y='Duration', by='Segment_Index', color=['lightblue'], legend=False).opts(width=1500, title='Eye Opening Segments (Awake Scans on Left | Drowsy Scans on Right)') * \
            Scan_Segments_table['EO'].loc[:,:,'Drowsy'].hvplot.bar(stacked=True, y='Duration', by='Segment_Index', color=['orange'], legend=False).opts(width=1500)
EXTRA_FIGURE1 = pn.Column(TOP_PANEL,BOT_PANEL)

EXTRA_FIGURE1.save('./images/extramaterials01_segments_per_run.png')

# > On the top panel, we can observe that EC segments are almost not present in awake scans (left) comparted to drowsy scans (right). For drowsy scans there is variability in the total amount of time that subjects kept their eyes closed. In many instances is by no means the majority of the scan.
#
# > On the bottom panel, we can observe that EO segments are much more promininent in awake scans (left). In fact in all instances they account for more than 95% of the scan duration (which is enforced by how we defined awake scans). Conversely for drowsy scans, EO periods account for much less of a run (right)
#
# <img src='./images/extramaterials01_segments_per_run.png'>
#
# ***
#
# ### 6.2. Distribution of EC/EO durations
#
# First, get counts of the number of EO and EC segments that last more than 60 seconds. We will show this info as annotations in the plots below

Num_EC_Segments_LongerThan60 = (Scan_Segments['EC']['Duration']>60).sum()
Num_EO_Segments_LongerThan60 = (Scan_Segments['EO']['Duration']>60).sum()

# Second, generate figure with information about EC segments

# +
EM2_LEFT = Scan_Segments['EC'].hvplot.box(y='Duration', ylim=(0,900), width=200, ylabel='Duration [Seconds]').opts(toolbar=None) * \
           hv.HLine(60).opts(line_width=1,line_dash='dashed',line_color='black').opts(fontsize={'ticks':14, 'labels':14})

EM2_RIGHT = Scan_Segments['EC'].hvplot.area(y='Duration',width=1000,ylim=(0,900),ylabel='Duration [seconds]',xlabel='Individual Scan Segments',xticks=np.arange(0,Scan_Segments['EC'].shape[0],1000), line_color='gray', color='gray') * \
            hv.HLine(60).opts(line_width=1,line_dash='dashed',line_color='black').opts(fontsize={'ticks':14, 'labels':14}) * \
            hv.Text(5200,675,'Number of EC Segments with duration > 60 seconds = %d' % Num_EC_Segments_LongerThan60, fontsize=14)

EXTRA_FIGURE2 = pn.Column(pn.pane.Markdown('## Distribution of Eye Closure segments across all scans'),
                          pn.Row(EM2_LEFT,EM2_RIGHT))

# -

EXTRA_FIGURE2.save('./images/extramaterials02_ECdistribution.png')

# * Left panel: box plot + outliers of all EC durations across all scans. 
# * Right panel: shows one vertical bar per EC segment. The 60s threshold is depicted as an horizontal dashed line.
#
# > We can see the majority of eye closure segments fall below the 60 second mark
#
# <img src='./images/extramaterials02_ECdistribution.png'>
#
# Third, generate figure with information about eye opening segments

# +
EM3_LEFT  = Scan_Segments['EO'].hvplot.box(y='Duration', ylim=(0,900), width=200, ylabel='Duration [Seconds]').opts(toolbar=None) * \
           hv.HLine(60).opts(line_width=1,line_dash='dashed',line_color='black').opts(fontsize={'ticks':14, 'labels':14})

EM3_RIGHT = Scan_Segments['EO'].hvplot.area(y='Duration',width=1000,ylim=(0,900),ylabel='Duration [seconds]',xlabel='Individual Scan Segments',xticks=np.arange(0,Scan_Segments['EC'].shape[0],1000), line_color='gray', color='gray') * \
            hv.HLine(60).opts(line_width=1,line_dash='dashed',line_color='black').opts(fontsize={'ticks':14, 'labels':14}) * \
            hv.Text(5200,675,'Number of EO Segments with duration > 60 seconds = %d' % Num_EO_Segments_LongerThan60, fontsize=14)

EXTRA_FIGURE3 = pn.Column(pn.pane.Markdown('## Distribution of Eye Opening segments across all scans'),
                          pn.Row(EM3_LEFT,EM3_RIGHT))
# -
EXTRA_FIGURE3.save('./images/extramaterials03_EOdistribution.png')


# Equivalent depiction for EO semgents.
#
# <img src='./images/extramaterials03_EOdistribution.png'>
