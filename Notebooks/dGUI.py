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

import pandas as pd
import xarray as xr
import hvplot.xarray
import numpy as np
import hvplot.pandas
import panel as pn
import seaborn as sns
import matplotlib.pyplot as plt
import os.path as osp
import os
from utils.variables import Resources_Dir, DATA_DIR
from utils.basics import get_available_runs, get_time_index, get_window_index
from utils.basics import load_fv_timeseries, load_motion_FD, load_motion_DVARS, load_PSD
from matplotlib.figure import Figure

# ***
# ## Setup Port Variable to already tunneled connection

port_tunnel = int(os.environ['PORT2'])
print('++ INFO: Second Port available: %d' % port_tunnel)

# ## Setup Configuration Variables

region             = 'V4_grp'
# Infer how the index should look for a spectrogram computed with a given window duration
spectrogram_windur = 60  # In seconds
nacqs              = 890 # Number of acquisitions
TR                 = 1   # In seconds

# ## Setup Time & Window Indexes for joint plotting

time_index = get_time_index(nacqs=nacqs, tr=TR)
windowed_time_index = get_window_index(nacqs=nacqs,tr=TR, win_dur=spectrogram_windur)

# ***
# ## Load Eye Tracker Data

ET_PupilSize_Proc_1Hz_B = pd.read_pickle(osp.join(Resources_Dir,'ET_PupilSize_Proc_1Hz_corrected.pkl'))
print('++ INFO: Data Frame shape %s' % str(ET_PupilSize_Proc_1Hz_B.shape))
ET_PupilSize_Proc_1Hz_B.head(2)

# + [markdown] tags=[]
# ***
# ## Load Subject Lists
# -

Manuscript_Runs = get_available_runs(when='final',type='all')
Awake_Runs      = get_available_runs(when='final',type='awake')
Drowsy_Runs     = get_available_runs(when='final',type='drowsy')
print('++ INFO: Number of Runs = [All = %d | Awake = %d | Drowsy = %d' %(len(Manuscript_Runs),len(Awake_Runs),len(Drowsy_Runs)))

# ***
# ## Load EC / EO segments

# Load Information about Segments of Eye Closure
# ==============================================
et_segm_limits = {'min_dur':60,'max_dur':475}
ec_path        = osp.join(Resources_Dir,'EC_Segments_Info.pkl')
eo_path        = osp.join(Resources_Dir,'EO_Segments_Info.pkl')
ec_segments_df = pd.read_pickle(ec_path)
eo_segments_df = pd.read_pickle(eo_path)
print('++ INFO: ec_segment_df has shape: %s' % str(ec_segments_df.shape))
ec_segments_df.head(2)

print('++ INFO: eo_segment_df has shape: %s' % str(eo_segments_df.shape))
eo_segments_df.head(2)

# ***
# ## Load 4th Ventricle Time series

# %%time
roits_df = load_fv_timeseries(Manuscript_Runs,region, index=time_index)
roits_df.head(2)

# ***
# ## Load Motion Information (Framewise Displacement and DVARS)
#
# > I computed SMRS instead of DVARS following advice from the AFNI people. Basically SRMS is a version of DVARS that is not modified by re-scaling, what makes it amenable to inter-subject comparisons

# %%time
motTR_FD_df = load_motion_FD(runs=Manuscript_Runs, nacqs=nacqs, index=time_index)
motTR_FD_df.head(2)

# 2. Create a DataFrame with the rolling windowed version of the motion data
# --------------------------------------------------------------------------
motWIN_FD_df = motTR_FD_df.rolling(window=spectrogram_windur,center=True).mean()
print('++ INFO: motWIN_FD_df has shape %s' % str(motWIN_FD_df.shape))
motWIN_FD_df.head(2)

# %%time
motTR_DVARS_df = load_motion_DVARS(runs=Manuscript_Runs, index=time_index)
motTR_DVARS_df.head(2)

# 2. Create a DataFrame with the rolling windowed version of the motion data
# --------------------------------------------------------------------------
motWIN_DVARS_df = motTR_DVARS_df.rolling(window=spectrogram_windur,center=True).mean()
print('++ INFO: motWIN_DVARS_df has shape %s' % str(motWIN_DVARS_df.shape))
motWIN_DVARS_df.head(2)

# ***
# ## Load Peridiogram (continous frequency analysis)
# > Scan-level results

# %%time
sel_suffix='mPP'
# Load All the periodograms for all subjects
# ==========================================
peridiogram_SCAN_df = pd.DataFrame(columns=Manuscript_Runs)
for sbj_run in Manuscript_Runs:
    sbj,run  = sbj_run.split('_',1)
    out_file  = osp.join(DATA_DIR,sbj,run,'{run}_{suffix}.Signal.{region}.welch.pkl'.format(run=run, region=region, suffix=sel_suffix))
    if osp.exists(out_file):
        aux = pd.read_pickle(out_file)
        peridiogram_SCAN_df[sbj_run] = aux['PSD (a.u./Hz)']
    else:
        print('++ WARNING: File missing [%s]' % out_file)
print('++ INFO: peridiogram_SCAN_df has shape %s' %  str(peridiogram_SCAN_df.shape))
peridiogram_SCAN_df.head(2)

# > Segment-level results

peridiogram_EC_df = pd.read_pickle(osp.join(Resources_Dir,'ET_Peridiograms_perSegments_EC.pkl'))
print('++ INFO: peridiogram_EC_df has shape %s' % str(peridiogram_EC_df.shape))
peridiogram_EC_df.head(2)

peridiogram_EO_df = pd.read_pickle(osp.join(Resources_Dir,'ET_Peridiograms_perSegments_EO.pkl'))
print('++ INFO: peridiogram_EO_df has shape %s' % str(peridiogram_EO_df.shape))
peridiogram_EO_df.head(2)

# ## Load Spectrogram and associated summary metrics (scan-level only so far)

sleep_psd_df     = load_PSD(Manuscript_Runs,band='sleep',index=windowed_time_index)
non_sleep_psd_df = load_PSD(Manuscript_Runs,band='non_sleep',index=windowed_time_index)
ratio_psd_df     = load_PSD(Manuscript_Runs,band='ratio_w_non_sleep',index=windowed_time_index)

sleep_psd_cumsum_df     = sleep_psd_df.cumsum()
non_sleep_psd_cumsum_df = non_sleep_psd_df.cumsum()
ratio_psd_cumsum_df    = ratio_psd_df.cumsum()
print('++ INFO: sleep_psd_cumsum_df has shape %s' % str(sleep_psd_cumsum_df.shape))
print('++ INFO: non_sleep_psd_cumsum_df has shape %s' % str(non_sleep_psd_cumsum_df.shape))
print('++ INFO: ratio_cumsum_df has shape %s' % str(ratio_psd_cumsum_df.shape))
sleep_psd_cumsum_df.head(2)

# %%time
# Take Time and Freq Dimensions from first run
sample_item             = Manuscript_Runs[0]
sample_sbj, sample_run  = sample_item.split('_',1)
sample_spectrogram_path = osp.join(DATA_DIR,sample_sbj,sample_run,'{RUN}_{SUFFIX}.Signal.{REGION}.Spectrogram.pkl'.format(RUN=sample_run, REGION=region, SUFFIX='mPP'))
sample_spectrogram_df   = pd.read_pickle(sample_spectrogram_path)
# Create Spectrogram XArray with correct dimensions
spectrogram_xr = xr.DataArray(dims=['Run','Frequency','Time'],coords={'Run':Manuscript_Runs, 'Frequency':sample_spectrogram_df.index, 'Time':sample_spectrogram_df.columns})
# Load all data from disk
for item in Manuscript_Runs:
    sbj,run = item.split('_',1)
    spectrogram_path   = osp.join(DATA_DIR,sbj,run,'{RUN}_{SUFFIX}.Signal.{REGION}.Spectrogram.pkl'.format(RUN=run, REGION=region, SUFFIX='mPP'))
    aux_spectrogram_df = pd.read_pickle(spectrogram_path)
    spectrogram_xr.loc[item,:,:] = aux_spectrogram_df

# + [markdown] tags=[]
# ***
# # Plotting Functions

# +
Run_Lists = {'Awake':Awake_Runs,'Drowsy':Drowsy_Runs, 'All':Manuscript_Runs}
select_list     = pn.widgets.Select(name='Run List:',options=['Awake', 'Drowsy'],       value='Awake')
select_run      = pn.widgets.Select(name='Run:',     options=Run_Lists[select_list.value], value=Run_Lists[select_list.value][0])
select_run_auto = pn.widgets.AutocompleteInput(name='Run (Auto-complete):', options=Run_Lists[select_list.value], value =Run_Lists[select_list.value][0])
toggle_eoc_segs = pn.widgets.Toggle(name='Show Periods of EC/OC', button_type='default')
fd_range        = pn.widgets.RangeSlider(name='Motion (FD) Range', start=0.0, end=10.0, value=(0.0, 1.0), step=0.1) 
dvars_range     = pn.widgets.RangeSlider(name='Motion (DVARS) Range', start=0.0, end=2.0, value=(0.0, 0.1), step=0.05) 
psd_range       = pn.widgets.RangeSlider(name='PSDs Range', start=0, end=500, value=(0, 250), step=10) 

@pn.depends(select_list.param.value, watch=True)
def _update_runs(selected_list):
    runs = Run_Lists[selected_list]
    select_run.options = runs
    select_run.value = runs[0]

@pn.depends(select_run_auto.param.value, watch=True)
def _update_from_auto_complete(selected_run):
    select_run.options = Run_Lists['All']
    select_run.value   = selected_run
    
@pn.depends(select_run.param.value, watch=True)
def _update_auto_complete(selected_run):
    select_run_auto.value = selected_run


# -

@pn.depends(select_run.param.value)
def show_run_info(run):
    classification = 'Unknown'
    if run in Awake_Runs:
        classification = 'Awake'
    if run in Drowsy_Runs:
        classification = 'Drowsy'
    num_ec_segments=ec_segments_df[ (ec_segments_df['Run']==run) & (ec_segments_df['Type']=='EC') ].shape[0]
    num_eo_segments=eo_segments_df[ (eo_segments_df['Run']==run) & (eo_segments_df['Type']=='EO') ].shape[0]
    eo_min_dur = eo_segments_df[(eo_segments_df['Run']==run) & (eo_segments_df['Type']=='EO') ]['Duration'].min()
    eo_max_dur = eo_segments_df[(eo_segments_df['Run']==run) & (eo_segments_df['Type']=='EO') ]['Duration'].max()
    ec_min_dur = ec_segments_df[(ec_segments_df['Run']==run) & (ec_segments_df['Type']=='EC') ]['Duration'].min()
    ec_max_dur = ec_segments_df[(ec_segments_df['Run']==run) & (ec_segments_df['Type']=='EC') ]['Duration'].max()

    output='<p style="text-align: center; background-color:powderblue; font-size:120%;"> {RUN} | {classif} | Num EO Segs = {neos} [min={eomin}s,max={eomax}s] | Num EC Segm = {neoc} [min={ecmin}s,max={ecmax}s]</p>'.format(RUN=run.rstrip(),classif=classification,
                                                                                                                                               neos=num_eo_segments,neoc=num_ec_segments,
                                                                                                                                               eomin=str(eo_min_dur),eomax=str(eo_max_dur),
                                                                                                                                               ecmin=str(ec_min_dur),ecmax=str(ec_max_dur))
    return pn.pane.Markdown(output,width=1400)


def plot_xy(primary,secondary,
            primary_label,secondary_label,
            primary_color='k',secondary_color='r',
            primary_alpha=1.0, secondary_alpha=1.0, 
            primary_line_width=1, secondary_line_width=1, 
            primary_min=None, primary_max=None, 
            secondary_min=None, secondary_max=None, 
            add_eoc_segments=False, run=None):
    fig = Figure(figsize=(20, 3))
    ax  = fig.subplots()
    primary     = primary.copy()
    secondary   = secondary.copy()
    r   = primary.corr(secondary)
    ax.plot(primary.index.total_seconds(),primary, c=primary_color, lw=primary_line_width, alpha=primary_alpha)
    ax.set_xlim(0,900)
    ax.set_ylabel(primary_label)
    ax.grid(False)
    if primary_max is not None:
        ax.set_ylim(top=primary_max)
    if primary_min is not None:
        ax.set_ylim(bottom=primary_min)
    ax.set_title('R[{primary},{secondary}]={r}'.format(primary=primary_label,secondary=secondary_label,r="{:.2f}".format(r)))
    ax_twin = ax.twinx()
    ax_twin.plot(secondary.index.total_seconds(),secondary, c=secondary_color, lw=secondary_line_width, alpha=secondary_alpha)
    ax_twin.set_ylabel(secondary_label)
    ax_twin.grid(False)
    if secondary_max is not None:
        ax_twin.set_ylim(top=secondary_max)
    if secondary_min is not None:
        ax_twin.set_ylim(bottom=secondary_min)
    ymin, ymax = ax_twin.get_ylim()
    ymid = ymin + ((ymax-ymin)/2)
    if add_eoc_segments:
        ec_line = np.ones(900)*np.nan
        for r,row in ec_segments_df[ec_segments_df['Run']==run].iterrows():
            ec_line[int(row['Onset']):int(row['Offset'])]=0.9*ymax
        eo_line = np.ones(900)*np.nan
        for r,row in eo_segments_df[eo_segments_df['Run']==run].iterrows():
            eo_line[int(row['Onset']):int(row['Offset'])]=0.9*ymax
        ax_twin.plot(np.arange(900),ec_line,'s',color='red', markersize=5, alpha=0.8)
        #ax_twin.plot(np.arange(900),eo_line,'o',color='white', markersize=10, alpha=0.3)
        ax_twin.set_xlim(0,900)
    return fig


@pn.depends(select_run.param.value, toggle_eoc_segs.param.value, fd_range.param.value, dvars_range.param.value)
def get_motion_plots(run, eoc_segs_toggle, fd_y_range, dvars_y_range):
    fig = plot_xy(motTR_FD_df[run],motTR_DVARS_df[run][1::],'Motion (FD)','Motion (DVARS)',primary_min=fd_y_range[0],primary_max=fd_y_range[1], secondary_min=dvars_y_range[0], secondary_max=dvars_y_range[1], add_eoc_segments=eoc_segs_toggle, run=run)
    return fig


@pn.depends(select_run.param.value, toggle_eoc_segs.param.value, dvars_range.param.value)
def get_roits_plots(run, eoc_segs_toggle, dvars_y_range):
    fig = plot_xy(roits_df[run],motTR_DVARS_df[run][1::],'ROI Timeseries','Motion (DVARS)',
                  primary_line_width=2.0, primary_color='k',
                  secondary_alpha=0.5, secondary_color='k', secondary_min=dvars_y_range[0], secondary_max=dvars_y_range[1], 
                  add_eoc_segments=eoc_segs_toggle, run=run)
    return fig


@pn.depends(select_run.param.value, toggle_eoc_segs.param.value, dvars_range.param.value)
def get_et_pupil_plots(run, eoc_segs_toggle, dvars_y_range):
    fig = plot_xy(ET_PupilSize_Proc_1Hz_B[run],motTR_DVARS_df[run][1::],'Pupil Size','Motion (DVARS)',
                  primary_line_width=2.0, primary_color='b',
                  secondary_alpha=0.5, secondary_color='k', secondary_min=dvars_y_range[0], secondary_max=dvars_y_range[1],
                  add_eoc_segments=eoc_segs_toggle, run=run)
    return fig


@pn.depends(select_run.param.value, toggle_eoc_segs.param.value, psd_range.param.value, dvars_range.param.value, fd_range.param.value)
def get_psd_plots(run, eoc_segs_toggle, psd_y_range, dvars_range, fd_y_range):
    fig_sleep     = plot_xy(sleep_psd_df[run],motTR_FD_df[run][1::],'PSD (Sleep Band)','Motion (FD)',        secondary_alpha=0.5,primary_line_width=3.0,primary_color='green',primary_min=psd_y_range[0], primary_max=psd_y_range[1], secondary_color='k', secondary_min=fd_y_range[0], secondary_max=fd_y_range[1], add_eoc_segments=eoc_segs_toggle, run=run)
    fig_non_sleep = plot_xy(non_sleep_psd_df[run],motTR_FD_df[run][1::],'PSD (Non Sleep Band)','Motion (FD)',secondary_alpha=0.5,primary_line_width=3.0,primary_color='r',primary_min=psd_y_range[0], primary_max=psd_y_range[1], secondary_color='k', secondary_min=fd_y_range[0], secondary_max=fd_y_range[1], add_eoc_segments=eoc_segs_toggle, run=run)
    fig_ratio     = plot_xy(ratio_psd_df[run],motTR_FD_df[run][1::],'PSD (Sleep / Non Sleep)','Motion (FD)', secondary_alpha=0.5,primary_line_width=3.0,primary_color='r',primary_min=psd_y_range[0], primary_max=psd_y_range[1], secondary_color='k', secondary_min=fd_y_range[0], secondary_max=fd_y_range[1], add_eoc_segments=eoc_segs_toggle, run=run)
    fig           = pn.Tabs(('SLEEP BAND',fig_sleep),('NON-SLEEP BAND',fig_non_sleep),('SLEEP/NON-SLEEP',fig_ratio), tabs_location='right')
    return fig


data_dashboard = pn.Column(pn.Row(select_list,select_run,select_run_auto),
          pn.Row(fd_range,dvars_range,psd_range,toggle_eoc_segs),
          pn.pane.Markdown('***', width=1400),
          show_run_info,
          pn.pane.Markdown('***', width=1400),
          get_motion_plots,
          get_roits_plots,
          get_psd_plots,
          get_et_pupil_plots
         )

data_dashboard_server = data_dashboard.show(port=port_tunnel,open=False)

data_dashboard_server.stop()
