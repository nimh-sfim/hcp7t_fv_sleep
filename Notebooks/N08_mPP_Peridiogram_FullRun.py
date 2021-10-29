import sys, getopt
import numpy as np
import os.path as osp
import pandas as pd
from scipy.signal import welch, get_window
from utils.variables import DATA_DIR
def main(argv):
  SBJ     = ''
  RUN     = ''
  REGION  = ''
  SUFFIX  = ''
  WIN_LENGTH  = 60  #128        #60
  WIN_OVERLAP = 45  #64         #30
  NFFT        = 128 #128        #64
  SCALING     = 'density'  #'density'
  DETREND     = 'constant' #'linear'   #'constant'
  FS          = 1
  try:
    opts,args = getopt.getopt(argv,"hs:d:r:w:p:",["subject=","rundif=","region=","wdir=","prep="])
  except getopt.GetoptError:
    print ('ExtractROIs.py -s <subject> -d <run> -r <region> -w <working_dir> -p <prep_suffix>')
    sys.exit(2)
  for opt, arg in opts:
    if opt == '-h':
      print ('ExtractROIs.py -s <subject> -d <run> -r <region> -w <working_dir> -p <preprocessing-suffix>')
      sys.exit()
    elif opt in ("-s", "--subject"):
      SBJ = arg
    elif opt in ("-d", "--rundir"):
      RUN = arg
    elif opt in ("-p", "--preproc"):
      SUFFIX = arg
    elif opt in ("-r", "--region"):
      REGION = arg
    elif opt in ("-w", "--wdir"):
      PRJDIR = arg
  print('++ WARNING: We assume data has a TR of 1s. Please ensure that is correct')
  print('++ =====================================================================')
  print('++ Working on %s' % SBJ)
  print(' + Region        -->  %s' % REGION)
  print(' + Run Dir       -->  %s' % RUN)
  print(' + Preprocessing -->  %s [NOT USED]' % SUFFIX)
  print(' + Data Dir      -->  %s' % PRJDIR)
  print(' + Win Length    -->  %d TRs' % WIN_LENGTH) 
  print(' + Win Overlap   -->  %d TRs' % WIN_OVERLAP) 
  print(' + Detrend       -->  %s'     % str(DETREND)) 
  print(' + Sampling Freq -->  %f Hz' % FS)
 
  roits_path = osp.join(DATA_DIR,SBJ,RUN,'{RUN}_mPP.Signal.{REGION}.1D'.format(RUN=RUN, REGION=REGION))
  welch_path = osp.join(DATA_DIR,SBJ,RUN,'{RUN}_mPP.Signal.{REGION}.welch.pkl'.format(RUN=RUN, REGION=REGION))
  
  # Load ROI Timeseries
  # ===================
  roits = pd.read_csv(roits_path, header=None)
  roits.columns = [REGION]
  print('++ INFO: Time series loaded into memory from [%s]' % roits_path)
    
  # Compute Peridiogram
  # ===================
  print('++ INFO: Computing Peridiogram...')
  wf, wc = welch(roits[REGION], fs=FS, window=get_window(('tukey',0.25),WIN_LENGTH), noverlap=WIN_OVERLAP, scaling=SCALING, detrend=DETREND, nfft=NFFT)
  #wc     = 10*np.log10(wc)
  
  # Put results into a dataframe
  # ============================
  peridiogram_df = pd.DataFrame(wc,index=wf,columns=['PSD (a.u./Hz)'])
  peridiogram_df.index.rename('Frequency', inplace=True)
    
  # Save results to disk
  # ====================
  peridiogram_df.to_pickle(welch_path)
  print('++ INFO: Peridiogram written to disk [%s]' % welch_path)

if __name__ == "__main__":
  main(sys.argv[1:])
