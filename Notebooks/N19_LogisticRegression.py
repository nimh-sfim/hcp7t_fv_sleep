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

# # Description - Predictive Power at the Segment Level for PSDsleep and GSamplitude
#
# This notebook generates Figure 11
#
# *"To study how well the presence of fluctuations around 0.05Hz in the FV can help us predict reduced wakefulness, we used a logistic regression classifier that takes as inputs windowed (window length = 60s, window step = 1s) estimates of PSDsleep. A duration of 60s was selected to match the frequency analyses described before."*

# # Import Libraries

# +
import pandas as pd
import numpy  as np
import xarray as xr
import seaborn as sns
import hvplot.pandas
import panel as pn
import holoviews as hv
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt
import os.path as osp
from utils.basics import get_available_runs, load_motion_FD, load_motion_DVARS
from utils.variables import DATA_DIR, Resources_Dir

from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle
import sklearn 

from bokeh.models.formatters import DatetimeTickFormatter
formatter = DatetimeTickFormatter(minutes = ['%Mmin:%Ssec'])

print('pandas: %s' % str(pd.__version__))
print('sklearn: %s' % str(sklearn.__version__))
print('seaborn: %s' % str(sns.__version__))
# %matplotlib inline
# -

# ## Configuration Variables

k_folds            = 20
spectrogram_windur = 60 # In seconds
region             = 'V4lt_grp'
Nacq               = 890

# ## Load Lists of Scans (all, awake, drowsy)

Manuscript_Runs = get_available_runs(when='final', type='all')
Awake_Runs      = get_available_runs(when='final', type='awake')
Drowsy_Runs     = get_available_runs(when='final', type='drowsy')
print('++ INFO: Number of Runs [All = %d, Awake = %d, Drowsy = %d]' % (len(Manuscript_Runs),len(Awake_Runs),len(Drowsy_Runs)))

# ## Load Pre-processed ET Data and Downsample it to 1Hz (to match fMRI data)

ET_PupilSize_Proc_1Hz = pd.read_pickle(osp.join(Resources_Dir,'ET_PupilSize_Proc_1Hz_corrected.pkl'))

[Nacq, Nruns ]              =  ET_PupilSize_Proc_1Hz.shape
print('++ Number of acquisitions = %d' % Nacq)
print('++ Number of runs = %d' % Nruns)

ET_PupilSize_Proc_1Hz = ET_PupilSize_Proc_1Hz[Manuscript_Runs]
ET_PupilSize_Proc_1Hz.index.name = 'Time'
ET_PupilSize_Proc_1Hz.name = 'Pupil Size'
print('++ INFO: Shape of Pupil Size Dataframe [ET_PupilSize_Proc_1Hz] is %s' % str(ET_PupilSize_Proc_1Hz.shape))

# ## Create Classification Problem Labels
# **Compute percent missing samples per window**

EC_Windowed = ET_PupilSize_Proc_1Hz.isna().rolling(window=spectrogram_windur, center=True).sum().dropna()
EC_Windowed = 100*(EC_Windowed/spectrogram_windur)
EC_Windowed.index.name='Time'
EC_Windowed.name = 'Percent_EC'


# **Create labels for the classification problem**
#
# *"Classification labels were generated as follows. For each sliding window, we calculated the percentage of eye closing time for that window. If that percentage was above 60%, we assigned the label ???eyes closed/drowsy???. If the percentage was lower than 40%, we assigned the label ???eyes open/awake???. All other segments were discarded given their mix/uncertain nature."*"

def assign_labels(a):
    if a >= 60:
        return 'EC/Drowsy'
    if a <= 40:
        return 'EO/Awake'
    return 'Mixed'


EC_Windowed_Labels = EC_Windowed.applymap(assign_labels)

# **Plot an exmaple of the classification labels next to the eye tracking traces**

sample_scan = '283543_rfMRI_REST1_PA'
(ET_PupilSize_Proc_1Hz[sample_scan].hvplot(xformatter=formatter, width=1500, height=150, title='Eye Tracking') + \
EC_Windowed[sample_scan].hvplot(xformatter=formatter, width=1500, shared_axes=False, height=150, title='Percentage of Eye Closure per window') +
EC_Windowed_Labels[sample_scan].hvplot(xformatter=formatter, width=1500, height=150, shared_axes=False, title='Classification labels')).cols(1)

# ***
# ## Load PSD in Sleep Band
#
# This data is already the result of a windowing operation (computation of the spectrogram). As such, we ensure the index has the correct values corresponding to windowed information

# Infer how the index should look for a spectrogram computed with a given window duration
time_index         = pd.timedelta_range(start='0 s', periods=Nacq, freq='s')
# Create empty dataframe with index having time_delta in steps of seconds
aux = pd.DataFrame(np.ones(Nacq),index=time_index)
# Simulate rolling windows of spectrogram_windur to gather which index should have data
aux                    = aux.rolling(window=spectrogram_windur, center=True).mean()
aux                    = aux.dropna()
windowed_time_index    = aux.index
del aux

# %%time
PSD_Windowed     = pd.DataFrame(index=windowed_time_index, columns=Manuscript_Runs)
for sbj_run in Manuscript_Runs:
    sbj,run = sbj_run.split('_',1)
    file    = '{RUN}_mPP.Signal.{REGION}.Spectrogram_BandLimited.pkl'.format(RUN=run, REGION=region)
    path    = osp.join(DATA_DIR,sbj,run,file)
    aux     = pd.read_pickle(path)
    PSD_Windowed.loc[(windowed_time_index,sbj_run)] = aux['sleep'].values

PSD_Windowed.head(5)

# ***
# ## Load Global Signal (under different regression scenarios)

# +
GS_df={}
GS_scenario2file={'Reference':'Reference','Basic':'BASIC','Basic+':'BASICpp','CompCor':'Behzadi_COMPCOR','CompCor+':'Behzadi_COMPCORpp'}

for GS_scenario in ['Reference','Basic','Basic+','CompCor','CompCor+']:
    GS_df[GS_scenario] = pd.DataFrame(columns=Manuscript_Runs)
    for item in Manuscript_Runs:
        sbj,run  = item.split('_',1)
        aux_path = osp.join(DATA_DIR,sbj,run,'{RUN}_{REGRESSION}.Signal.GM.1D'.format(RUN=run, REGRESSION=GS_scenario2file[GS_scenario]))
        if not osp.exists(aux_path):
            print(' + --> WARNING: Missing file: %s' % aux_path)
            continue
        aux_data = np.loadtxt(aux_path)
        assert aux_data.shape[0] == Nacq, "{file} has incorrect length {length}".format(file=aux_path, length=str(aux_data.shape[0]))
        GS_df[GS_scenario][item] = aux_data
    print(' + INFO [%s]: df.shape = %s' % (GS_scenario,str(GS_df[GS_scenario].shape)))
# -

GS_Windowed={}
for GS_scenario in ['Reference','Basic','Basic+','CompCor','CompCor+']:
    GS_Windowed[GS_scenario]       = GS_df[GS_scenario].rolling(window=spectrogram_windur, center=True).std().dropna()
    GS_Windowed[GS_scenario].index = PSD_Windowed.index
    print(' + INFO [%s]: df.shape = %s' % (GS_scenario,str(GS_Windowed[GS_scenario].shape)))
GS_Windowed[GS_scenario].sample(5)

# ***
#
# ## Load Motion

motFD    = load_motion_FD(Manuscript_Runs, index=time_index)
motDVARS = load_motion_DVARS(Manuscript_Runs, index=time_index)

motFD_Windowed    = motFD.rolling(window=spectrogram_windur, center=True).quantile(0.99).dropna()
motDVARS_Windowed = motDVARS.rolling(window=spectrogram_windur, center=True).quantile(0.99).dropna()

# ## Generate Classification Problem

data = pd.DataFrame()
data['PSDsleep']      = PSD_Windowed.melt()['value']
data['GS[Reference]'] = GS_Windowed['Reference'].melt()['value']
data['GS[Basic]']     = GS_Windowed['Basic'].melt()['value']
data['GS[Basic+]']    = GS_Windowed['Basic+'].melt()['value']
data['GS[CompCor]']   = GS_Windowed['CompCor'].melt()['value']
data['GS[CompCor+]']  = GS_Windowed['CompCor+'].melt()['value']
data['EC']          = EC_Windowed.melt()['value']
data['FD']          = motFD_Windowed.melt()['value']
data['DVARS']          = motDVARS_Windowed.melt()['value']
data['Label']       = EC_Windowed_Labels.melt()['value']
print(data.shape)

data = data[(data['Label']=='EO/Awake') | (data['Label']=='EC/Drowsy')]
data = data.reset_index(drop=True)
print(data.shape)

data.sample(10)

# # Logistic Regression (Stratified k-Cross Validation)
#
# I think this will make things better because our sample is very much unbalanced

num_awake_umbalanced  = (data['Label']=='EO/Awake').sum()
num_drowsy_umbalanced = (data['Label']=='EC/Drowsy').sum()
num_mixed_umbalanced = (data['Label']=='Mixed').sum()
num_total_umbalanced  = data.shape[0]
print("++ INFO: Number of Awake Samples:  %d/%d [%.2f]" % (num_awake_umbalanced, num_total_umbalanced,100*num_awake_umbalanced/num_total_umbalanced))
print("++ INFO: Number of Drowsy Samples: %d/%d [%.2f]" % (num_drowsy_umbalanced,num_total_umbalanced,100*num_drowsy_umbalanced/num_total_umbalanced))
print("++ INFO: Number of Mixed Samples: %d/%d [%.2f]" % (num_mixed_umbalanced,num_total_umbalanced,100*num_mixed_umbalanced/num_total_umbalanced))

skf_umbalanced = StratifiedKFold(n_splits=k_folds, shuffle=True)

# ## Training / Testing in umbalanced sample

data_umbalanced = data.copy()

# +
# %%time
input_features         = ['PSDsleep','GSamp[Reference]','GSamp[BASIC]','GSamp[BASIC+]','GSamp[COMPCOR]','GSamp[COMPCOR+]','Control','PSD & GSamp[CompCor+]', 'PSD & Motion']
scores_umbalanced      = pd.DataFrame(index=np.arange(k_folds))
predictions_umbalanced = {}
true_vals_umbalanced   = {}
conf_mat_umbalanced    = xr.DataArray(dims=['Metric','K-Fold','True Label','Predicted Label'],
                                      coords={'Metric':input_features,
                                              'K-Fold':np.arange(k_folds),
                                              'True Label':['EC/Drowsy','EO/Awake'],
                                              'Predicted Label':['EC/Drowsy','EO/Awake']})
for metric in input_features:
    print('++ Working on %s ...' % metric)
    if metric == 'Control':
        X_aux = data_umbalanced['PSDsleep'].values.reshape(-1, 1)
        y_aux = data_umbalanced['Label']
        X,y = shuffle(X_aux,y_aux)
    if metric == 'PSDsleep':
        X = data_umbalanced['PSDsleep'].values.reshape(-1, 1)
        y = data_umbalanced['Label']
    if metric == 'GSamp[Reference]':
        X = data_umbalanced['GS[Reference]'].values.reshape(-1, 1)
        y = data_umbalanced['Label']
    if metric == 'GSamp[Basic]':
        X = data_umbalanced['GS[Basic]'].values.reshape(-1, 1)
        y = data_umbalanced['Label']
    if metric == 'GSamp[Basic+]':
        X = data_umbalanced['GS[Basic+]'].values.reshape(-1, 1)
        y = data_umbalanced['Label']
    if metric == 'GSamp[CompCor]':
        X = data_umbalanced['GS[CompCor]'].values.reshape(-1, 1)
        y = data_umbalanced['Label']
    if metric == 'GSamp[CompCor+]':
        X = data_umbalanced['GS[CompCor+]'].values.reshape(-1, 1)
        y = data_umbalanced['Label']
    if metric == 'PSD & GSamp[CompCor+]':
        X = data_umbalanced[['PSDsleep','GS[CompCor+]']].values
        y = data_umbalanced['Label']
    if metric == 'PSD & Motion':
        X = data_umbalanced[['PSDsleep','FD','DVARS']].values
        y = data_umbalanced['Label']
    if metric == 'All':
        X = data_umbalanced.drop(['Label','FD','DVARS','PSDsleep','EC'],axis=1).values
        y = data_umbalanced['Label']
    aux_num_awake  = (y=='EO/Awake').sum()
    aux_num_drowsy = (y=='EC/Drowsy').sum()
    aux_num_total  = y.shape[0]
    print(" + INFO: Number of Awake Samples:  %d/%d [%.2f]" % (aux_num_awake, aux_num_total,100*aux_num_awake/aux_num_total))
    print(" + INFO: Number of Drowsy Samples: %d/%d [%.2f]" % (aux_num_drowsy,aux_num_total,100*aux_num_drowsy/aux_num_total))
    for i, (train_index, test_index) in enumerate(skf_umbalanced.split(X, y)):
        log_reg = LogisticRegression(random_state=42, fit_intercept=True, max_iter=5000, n_jobs=-1, class_weight='balanced')
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        aux_num_awake_train  = (y_train=='EO/Awake').sum()
        aux_num_drowsy_train = (y_train=='EC/Drowsy').sum()
        aux_num_total_train = y_train.shape[0]
        aux_num_awake_test  = (y_test=='EO/Awake').sum()
        aux_num_drowsy_test = (y_test=='EC/Drowsy').sum()
        aux_num_total_test = y_test.shape[0]
        print(' + --> Split [%d - Training Set]: Awake [%d/%d [%.2f]] | Drowsy [%d/%d [%.2f]]' % (i,aux_num_awake_train, aux_num_total_train,100*aux_num_awake_train/aux_num_total_train,
                                                                                                                         aux_num_drowsy_train,aux_num_total_train,100*aux_num_drowsy_train/aux_num_total_train))
        print(' + --> Split [%d - Test Set    ]: Awake [%d/%d [%.2f]] | Drowsy [%d/%d [%.2f]]' % (i,aux_num_awake_test , aux_num_total_test ,100*aux_num_awake_test/aux_num_total_test,
                                                                                                                         aux_num_drowsy_test ,aux_num_total_test ,100*aux_num_drowsy_test/aux_num_total_test))
        
        log_reg.fit(X_train,y_train)
        scores_umbalanced.loc[i,metric]    = log_reg.score(X_test,y_test)
        predictions_umbalanced[(metric,i)] = log_reg.predict(X_test)
        true_vals_umbalanced[(metric,i)]   = y_test 
        conf_mat_umbalanced.loc[metric,i,:,:]  = confusion_matrix(true_vals_umbalanced[(metric,i)],predictions_umbalanced[(metric,i)], normalize='pred') #Precision

scores_umbalanced = scores_umbalanced.infer_objects()
# -

fig, ax = plt.subplots(1,1,figsize=(10,5))
umbal_boxplot = sns.boxplot(data=scores_umbalanced, ax=ax, color='grey')
umbal_boxplot.set_title('Classification Results', fontsize=18)
umbal_boxplot.set_xlabel('Input Features', fontsize=18)
umbal_boxplot.set_ylabel('Accuracy', fontsize=18)
umbal_boxplot.xaxis.set_tick_params(rotation=90, labelsize=14)
umbal_boxplot.yaxis.set_tick_params(labelsize=14)
umbal_boxplot.set_ylim(0,1)

data_umbalanced.drop(['Label','FD','DVARS','PSDsleep'],axis=1)

print('Statistical Test for Umbalanced/Original Sample Scenario')
print('========================================================')
print('PSD           vs. Control:  %s' % str(ttest_ind(scores_umbalanced['PSDsleep'],scores_umbalanced['Control'], alternative='two-sided')))
print('GS[Smoothing] vs. Control:   %s' % str(ttest_ind(scores_umbalanced['GSamp[Reference]'],scores_umbalanced['Control'], alternative='two-sided')))
print('GS[Basic]     vs. Control:   %s' % str(ttest_ind(scores_umbalanced['GSamp[BASIC]'],scores_umbalanced['Control'], alternative='two-sided')))
print('GS[Basic+]    vs. Control:   %s' % str(ttest_ind(scores_umbalanced['GSamp[BASIC+]'],scores_umbalanced['Control'], alternative='two-sided')))
print('GS[CompCor]   vs. Control:   %s' % str(ttest_ind(scores_umbalanced['GSamp[COMPCOR]'],scores_umbalanced['Control'], alternative='two-sided')))
print('GS[CompCor+]  vs. Control:   %s' % str(ttest_ind(scores_umbalanced['GSamp[COMPCOR+]'],scores_umbalanced['Control'], alternative='two-sided')))
print('-----')
print('PSD           vs. GS[Smoothing]:  %s' % str(ttest_ind(scores_umbalanced['PSDsleep'],scores_umbalanced['GSamp[Reference]'], alternative='two-sided')))
print('-----')
print('GS[CompCor+] vs. GS[Smoothing]:  %s' % str(ttest_ind(scores_umbalanced['GSamp[COMPCOR+]'],scores_umbalanced['GSamp[Reference]'], alternative='two-sided')))

scores_umbalanced.mean()

scores_umbalanced.mean()

PSD_Windowed


