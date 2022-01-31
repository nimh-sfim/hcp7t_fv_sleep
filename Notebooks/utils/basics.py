from .variables import RUNS, DATA_DIR, ProjectFiles_DF_Path, QA1_Results_DF_Path, Resources_Dir
import pandas as pd
import numpy  as np
import os.path as osp

# Compute aliased frequency given the original frequency and the sampling frequency
# =================================================================================
def aliased_freq(fs,f):
    n  = round(f / fs) # Closer integer multiple of fs to f
    fa = abs(fs*n - f)
    return fa
   
# Frequency Analysis Functions
# ============================
def generate_sleep_psd(f,t,Sxx,SleepBand_BotFreq=0.03,SleepBand_TopFreq=0.07):
    # Put Spectrogram into pandas dataframe (tidy-data form)
    df = pd.DataFrame(Sxx.T, columns=f, index=t)
    df = pd.DataFrame(df.stack()).reset_index()
    df.columns=['TR','Freq','PSD']

    # Extract average timeseries of power at different bands
    sleep_band_power     = df[(df['Freq'] < SleepBand_TopFreq) & (df['Freq'] > SleepBand_BotFreq)].groupby('TR').mean()['PSD'].values
    non_sleep_band_power = df[(df['Freq'] > SleepBand_TopFreq) | (df['Freq'] < SleepBand_BotFreq)].groupby('TR').mean()['PSD'].values
    control_band_power   = df[(df['Freq'] > 0.1)               & (df['Freq'] < 0.2)].groupby('TR').mean()['PSD'].values
    total_power          = df.groupby('TR').mean()['PSD'].values

    # Create output dataframe, which will contain both the average power traces computed above,
    # as well as the ratio of power in sleep band to all other "control" conditions
    df3                   = pd.DataFrame(columns=['sleep','non_sleep','total','control','ratio_w_total','ratio_w_non_sleep','ratio_w_control'])
    df3['sleep']          = sleep_band_power
    df3['non_sleep']      = non_sleep_band_power
    df3['total']          = total_power
    df3['control']        = control_band_power

    df3['ratio_w_total']     = sleep_band_power/total_power
    df3['ratio_w_non_sleep'] = sleep_band_power/non_sleep_band_power
    df3['ratio_w_control']   = sleep_band_power/control_band_power
    # Set the time index for the output dataframe
    df3.index = t
    return df3
  
# Indexing related functions
# ==========================
def get_time_index(nacqs,tr):
    """
    Creates a pandas timedelta index for data that has one entry per TR
    
    INPUTS:
    
    nacqs: number of volumetric acquisitions
    tr:    repetition time (in seconds)
    
    OUPUTS:
    time_index: a TimedeltaIndex object with the timing for each TR/acquisition
    """
    time_index         = pd.timedelta_range(start='0 s', periods=nacqs, freq='{tr}s'.format(tr=tr))
    return time_index

def get_window_index(nacqs,tr,win_dur):
    """
    Creates a pandas timedelta index for data that has one entry per window. It assumes a window step
    of 1 repetition time.
    
    INPUTS:
    
    nacqs: number of volumetric acquistions
    tr:    repetition time (in seconds)
    win_dur: window_duration (in seconds)
    
    OUTPUTS:
    
    window_index: a TimedeltaIndex object with the timing for each window
    """
    # First we generate a regular time index (the one that corresponds to the fMRI TR)
    time_index         = pd.timedelta_range(start='0 s', periods=nacqs, freq='{tr}s'.format(tr=tr))
    # Create empty dataframe with index having time_delta in steps of seconds
    aux = pd.DataFrame(np.ones(nacqs),index=time_index)
    # Simulate rolling windows of spectrogram_windur to gather which index should have data
    aux               = aux.rolling(window=win_dur, center=True).mean()
    aux               = aux.dropna()
    window_time_index = aux.index
    return window_time_index
   
# Functions to get list of subjects and runs
# ==========================================
def get_7t_subjects():
  """
  This function will load a list of subjects that have at least one rest run on the 7T scanner
  """ 
  df = pd.read_pickle(ProjectFiles_DF_Path)
  sbj_list = list(df['Sbj'].unique())
  return sbj_list

def get_available_runs(when='post_download', type='all'):
    """
    This function returns a list with all run names available at different times during the 
    analyses according to the when parameter.
    
    Inputs
    ------
    when: str with three possible values: post_download, post_qa1, final
    type: str with three possible values: all, drowsy, awake. This only applies if when = final
    
    Outputs
    -------
    out_list: list of strings with run names.
    """
    out_list = []
    # List of runs for which fMRI data was acquired
    if when == 'post_download':
        data = pd.read_pickle(ProjectFiles_DF_Path)
    # List of runs for which ET exists
    if when == 'post_qa1':
        data = pd.read_pickle(QA1_Results_DF_Path)
        data = data[(data['ET_OK']==True) & (data['ET Avail']==True) & (data['Spatial Resolution OK']==True) & (data['TR OK']==True) & (data['Nacq OK']==True)]
        for index,row in data.iterrows():
            sbj = str(row['Sbj'])
            run = str(row['Run'])
            out_list.append('_'.join([sbj,run]))
    if when == 'final':
        path_awake  = osp.join(Resources_Dir,'Run_List_Awake.txt')
        path_drowsy = osp.join(Resources_Dir,'Run_List_Drowsy.txt')
        awake_list  = list(np.loadtxt(path_awake,dtype=str))
        drowsy_list = list(np.loadtxt(path_drowsy,dtype=str))
        if type == 'all':
            out_list    = awake_list + drowsy_list
        elif type == 'awake':
            out_list = awake_list
        elif type == 'drowsy':
            out_list = drowsy_list
    return out_list

# Functions to load data
# ======================
def load_segments(kind,runs=None,min_dur=None):
    """
    Load information about scan segments according to ET data
    
    INPUTS
    runs: list of scans ID to include
    kind: type of target segments (EC=eyes closed, EO=eyes open)
    min_dur: remove any segment with duration less than this threshold (in seconds)
    
    OUTPUTS
    df: dataframe with one row per segment of interest. For each segment it will include the runID,
        segment type, segment index, segment UUID, segment onset (secodns), segment offset (seconds), 
        segment duration (seconds) and scan label.
    """
    if kind == 'all':
       path_EC = osp.join(Resources_Dir,'EC_Segments_Info.pkl')
       path_EO = osp.join(Resources_Dir,'EO_Segments_Info.pkl')
       df_EC   = pd.read_pickle(path_EC)
       df_EO   = pd.read_pickle(path_EO)
       df      = pd.concat([df_EO, df_EC], axis=0).reset_index(drop=True)
    else:
       path = osp.join(Resources_Dir,'{k}_Segments_Info.pkl'.format(k=kind))
       df   = pd.read_pickle(path)
    
    if min_dur is not None:
        df=df[df['Duration']>min_dur] # JAVIER: may want to change to >=min_dur lateer #
      
    if runs is not None:
       df=df[df['Run'].isin(runs)]
    
    print('++ INFO: segment_df has shape: %s' % str(df.shape))
    
    return df
   
def load_motion_info(sbjs, verbose=False, fillnan=True, write_FD=False):
  """
  Load motion information for all subjects in provided list into a dataframe. In addition,
  if instructed to do so, it will generate a new file per run with the trace of framewise
  displacement, which is not directly available as a download from XNAT Central.
  
  INPUTS
  
  sbjs: list of subjects
  verbose: whether or not to give excessive messages (default=False)
  fillnan: if a run does not exists for a given subject, still create an entry on the resulting
           dataframe and fill it with np.nan (default=True)
  write_FD: create a text file with the traces of FD in each folder (default=False)
  
  OUTPUTS
  
  mot_df: dataframe with one row per rest run. For each run, it contians the subject, run, mean FD and max FD
  
  If instructed, this function will also write files to disk as described above
  """
  # Create Empty DataFramw with 4 columns: Subject ID, Run ID, Mean Framewise Displacement, Maximum Framewise Displacement
  mot_df = pd.DataFrame(columns=['Sbj','Run','FD_mean','FD_max'])
  for sbj in sbjs:
    for run in RUNS:
      aux_path = osp.join(DATA_DIR,str(sbj),run,'{run}_Movement_Regressors.txt'.format(run=run))
      if osp.exists(aux_path):
        aux = pd.DataFrame(np.loadtxt(aux_path), columns=['x','y','z','roll','pitch','yaw','d_x','d_y','d_z','d_roll','d_pitch','d_yaw'])
        aux['FD'] = aux['d_x'].abs() + \
                    aux['d_y'].abs() + \
                    aux['d_z'].abs() + \
                    (50*aux['d_roll'].apply(np.deg2rad)).abs() + \
                    (50*aux['d_pitch'].apply(np.deg2rad)).abs() + \
                    (50*aux['d_yaw'].apply(np.deg2rad)).abs()
        if write_FD:
            FD_df = pd.DataFrame(aux['FD'])
            FD_path = osp.join(DATA_DIR,str(sbj),run,'{run}_Movement_FD.txt'.format(run=run))
            FD_df.to_csv(FD_path, sep='\t')
            #print('++ INFO: FD data frame saved as CSV text file for: {sbj} {run} | {fpath}'.format(run=run, sbj=sbj, fpath=FD_path))
        mot_df = mot_df.append({'Sbj':sbj,'Run':run,'FD_mean':aux['FD'].mean(),'FD_max':aux['FD'].max() }, ignore_index=True)
      else:
        if fillnan:
          mot_df = mot_df.append({'Sbj':sbj,'Run':run,'FD_mean':np.nan, 'FD_max':np.nan }, ignore_index=True)
        if verbose:
          print('++ [WARNING] File Missing: %s' % aux_path)
  mot_df['index'] = mot_df.index
  print('++ Final Shape = %s' % str(mot_df.shape))
  return mot_df

def load_motion_FD(runs, nacqs=890, index=None):
  """
  Load timeseries of framewise displacement for a given list of runs. 
  
  INPUTS
  
  runs: list of runs
  nacqs: number of volumes to return starting from the end (to account for discarded volumes)
  index: optional index for the output dataframe
  
  OUTPUTS
  
  DF: dataframe with one column per run.
  """
  if index is not None:
      DF = pd.DataFrame(index=index, columns=runs)
  else:
      DF = pd.DataFrame(columns=runs)
  
  for sbj_run in runs:
      sbj,run = sbj_run.split('_',1)
      path    = osp.join(DATA_DIR,str(sbj),run,'{run}_Movement_FD.txt'.format(run=run))
      aux     = pd.read_csv(path,sep='\t', index_col=0)
      DF.loc[:,sbj_run] = aux['FD'].values[-nacqs:]

  print('++ INFO: Shape of returned Dataframe is %s' % str(DF.shape))
  return DF

def load_motion_DVARS(runs, index=None):
  """
  Load timeseries of DVARs estimates for a given list of runs. 
  
  INPUTS
  
  runs: list of runs
  nacqs: number of volumes to return starting from the end (to account for discarded volumes)
  index: optional index for the output dataframe
  
  OUTPUTS
  
  DF: dataframe with one column per run.
  """
  if index is not None:
      DF = pd.DataFrame(index=index, columns=runs)
  else:
      DF = pd.DataFrame(columns=runs)
  
  for sbj_run in runs:
      sbj,run = sbj_run.split('_',1)
      path    = osp.join(DATA_DIR,str(sbj),run,'{run}_Movement_SRMS.1D'.format(run=run))
      aux     = np.loadtxt(path)
      DF.loc[:,sbj_run] = aux

  print('++ INFO: Shape of returned Dataframe is %s' % str(DF.shape))
  return DF
 
def load_fv_timeseries(runs,region,index=None):
    """
    Load representative timeseries for the 4th ventricle for a given list of scans and place them on a dataframe
    
    INPUTS
    
    runs: list of scans
    region: ROI name (default = V4_grp)
    index: index for the output dataframe. If none is provided, then the output object will have integer-based indexing.
    
    OUTPUT
    
    DF: dataframe with one row per acquistion (i.e., TR) and one column per run.
    """
    if index is not None:
        DF = pd.DataFrame(index=index, columns=runs)
    else:
        DF = pd.DataFrame(columns=runs)
    for sbj_run in runs:
        sbj,run = sbj_run.split('_',1)
        path    = osp.join(DATA_DIR,sbj,run,'{RUN}_mPP.Signal.{REGION}.1D'.format(RUN=run, REGION=region))
        aux     = np.loadtxt(path)
        DF.loc[:,sbj_run] = aux

    print('++ INFO: Shape of returned Dataframe is %s' % str(DF.shape))
    return DF
   
def load_PSD(runs, band='sleep', region='V4_grp',index=None):
    """
    Load PSD timeseries for a given list of runs and returns them on a pandas dataframe.
    
    INPUTS
    
    runs: list of scans
    band: possible value are 'sleep', 'non_sleep', 'total', 'control', 'ratio_w_total','ratio_w_non_sleep', 'ratio_w_control'
    region: ROI name (default = V4_grp)
    index: index for the output dataframe. If none is provided, then the output object will have integer-based indexing.
    
    OUTPUT
    
    DF: dataframe with one row per windowed PSD estimate and one column per run.
    """
    if index is not None:
        DF = pd.DataFrame(index=index, columns=runs)
    else:
        DF = pd.DataFrame(columns=runs)

    for sbj_run in runs:
        sbj,run = sbj_run.split('_',1)
        file    = '{RUN}_mPP.Signal.{REGION}.Spectrogram_BandLimited.pkl'.format(RUN=run, REGION=region)
        path    = osp.join(DATA_DIR,sbj,run,file)
        aux     = pd.read_pickle(path)
        DF.loc[:,sbj_run] = aux[band].values
    DF = DF.infer_objects()
    print('++ INFO: Shape of returned Dataframe is %s' % str(DF.shape))
    return DF

# FUNCTIONS USED FOR EYE TRACKING PRE-PROCESSING
# ==============================================
def alt_mean(data):
    """
    This function provides an alternative way to compute the mean value for a window, taking into account how many missing points exists.
    
    If percentage of missing values is less or equal to pc_missing, then return the mean computed using all available datapoints
    
    Conversely, if the percentage of missing values is larger than the threshold, then return np.nan
    
    INPUTS:
    data: pandas dataframe with data for a given window
    pc_missing: percentage of missing data
    
    """
    pc_missing = 0.5
    l = data.shape[0]
    n_nans = data.isna().sum()
    if n_nans <= pc_missing:
        return data.mean()
    else:
        return np.nan
    
def smooth(x,window_len=11,window='hanning'):
    """
    This function was copied from: https://scipy-cookbook.readthedocs.io/items/SignalSmooth.html
    
    Smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
        
    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    
    see also: 
    
    np.hanning, np.hamming, np.bartlett, np.blackman, np.convolve
    scipy.signal.lfilter
 
    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise (ValueError, "smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise (ValueError, "Input vector needs to be bigger than window size.")


    if window_len<3:
        return x


    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise (ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")


    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    return y


