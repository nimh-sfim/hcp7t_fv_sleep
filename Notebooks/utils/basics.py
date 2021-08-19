from .variables import RUNS, DATA_DIR, ProjectFiles_DF_Path, QA1_Results_DF_Path
import pandas as pd
import numpy  as np
import os.path as osp

def get_7t_subjects():
  """
  This function will load a list of subjects that have at least one rest run on the 7T scanner
  """ 
  df = pd.read_pickle(ProjectFiles_DF_Path)
  sbj_list = list(df['Sbj'].unique())
  return sbj_list

def get_available_runs(when='post_download'):
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
    return out_list

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


