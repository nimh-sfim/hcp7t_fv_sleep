import os.path as osp

# List of Possible Runs (not all runs may be available for all subjects)
# ----------------------------------------------------------------------
RUNS        = ['rfMRI_REST1_PA','rfMRI_REST2_AP','rfMRI_REST3_PA','rfMRI_REST4_AP']

# Name of Atlas used for Network Analyses
# ---------------------------------------
ATLAS_NAME  = 'Schaefer2018_200Parcels_7Networks'

# Primary Folders for this project
# ================================
DATA_DIR    = '/data/SFIMJGC_HCP7T/HCP7T'
SCRIPTS_DIR = '/data/SFIMJGC_HCP7T/hcp7t_fv_sleep'
ATLAS_DIR   = osp.join(DATA_DIR,'Atlases',ATLAS_NAME)

# XNAT Credentials
# ================
XNAT_USER     = 'your_username'
XNAT_PASSWORD = 'your_password' 

# Path to files that are used often throughout the project
# ========================================================
Resources_Dir                = osp.join(SCRIPTS_DIR,'Resources')              # Resources Directory --> will contain some derivatives such as .pkl files, .py files, etc.
SbjList_Orig_Path            = osp.join(Resources_Dir,'7T_SubjectList.csv')   # Path to the 7T Behavioral Info file downloaded from ConnectomeDB
ProjectFiles_DF_Path         = osp.join(Resources_Dir,'7T_ProjectFiles.pkl')  # Pickle file with info about what files were downloaded from ConnectomeDB using notebook N01.
Avail_fMRI_Runs_Info_DF_Path = osp.join(Resources_Dir,'7T_XNAT_Available_Runs_Info.pkl') # Pikle file with a list of available fMRI runs
QA1_Results_DF_Path          = osp.join(Resources_Dir,'QA_Part1_Results.pkl') # Pickle file with the results from the QA (e.g., ET available, resolution ok, etc.)
#ValidRunsList_Path           = osp.join(Resources_Dir,'QA2_List_of_valid_runs.txt')

# Path for summary objects regarding ET preprocessing
# ===================================================
ETinfo_path            = osp.join(Resources_Dir,'ET_info.pkl')
ET_PupilSize_Orig_path = osp.join(Resources_Dir,'ET_PupilSize_Orig.pkl')
ET_PupilSize_Proc_path = osp.join(Resources_Dir,'ET_PupilSize_Proc.pkl')
ET_Xposition_Orig_path = osp.join(Resources_Dir,'ET_Xposition.pkl')
ET_Yposition_Orig_path = osp.join(Resources_Dir,'ET_Yposition.pkl')
ET_Blinks_path         = osp.join(Resources_Dir,'ET_Blinks.pkl')
ET_BlinkFreq_path      = osp.join(Resources_Dir,'ET_BlinksFreq.pkl')


# Generic Variables to set analysis hyper-parameters
# ==================================================
ET_Blink_Dur_Thr         = 1000  # Blinks longer than this duration [ms] are considered eye closures due to sleep (not blinks)
ET_Blink_Buffer_NSamples = 50    # Buffer around onset and offset of blinks that gets removed and interpolated
                                 # (this takes care of spikes that happen in ET traces at the begining and end of blinks)
ET_MinNum_Fixations      = 20    # Minimum Number of Fixations per run to consider the fixation information viable
Ndiscard                 = 10    # Number of volumes discarded in pre-processing (to reach steady-state). In units of seconds.

# Thresholds for defining the awake and drowsy subjects (based on percentage of time they had their eyes closed)
# --------------------------------------------------------------------------------------------------------------
PercentWins_EC_Awake     = 0.05         # Subjects who kept their eyes closed less than 5% of the time --> awake subjects
PercentWins_EC_Sleep     = [0.20, 0.90] # Subjects who kept their eyes closed between 10 and 90% of the time --> drowsy subjects