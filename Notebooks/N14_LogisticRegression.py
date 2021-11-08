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

# # Import Libraries

# +
import pandas as pd
import numpy  as np
import xarray as xr
import seaborn as sns
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt
import os.path as osp
from utils.basics import get_available_runs
from utils.variables import DATA_DIR, Resources_Dir

from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle
import sklearn 

print('pandas: %s' % str(pd.__version__))
print('sklearn: %s' % str(sklearn.__version__))
print('seaborn: %s' % str(sns.__version__))
# %matplotlib inline
# -

# # Configuration Variables

k_folds = 20
spectrogram_windur = 60 # In seconds

# # Load Lists of Scans (all, awake, drowsy)

Manuscript_Runs = get_available_runs(when='final', type='all')
Awake_Runs      = get_available_runs(when='final', type='awake')
Drowsy_Runs     = get_available_runs(when='final', type='drowsy')
print('++ INFO: Number of Runs [All = %d, Awake = %d, Drowsy = %d]' % (len(Manuscript_Runs),len(Awake_Runs),len(Drowsy_Runs)))

# # Load Pre-processed ET Data and Downsample it to 1Hz (to match fMRI data)

ET_PupilSize_Proc_1Hz = pd.read_pickle(osp.join(Resources_Dir,'ET_PupilSize_Proc_1Hz_corrected.pkl'))

[Nacq, Nruns ]              =  ET_PupilSize_Proc_1Hz.shape
print('++ Number of acquisitions = %d' % Nacq)
print('++ Number of runs = %d' % Nruns)

ET_PupilSize_Proc_1Hz = ET_PupilSize_Proc_1Hz[Manuscript_Runs]
print('++ INFO: Shape of Pupil Size Dataframe [ET_PupilSize_Proc_1Hz] is %s' % str(ET_PupilSize_Proc_1Hz.shape))

EC_Windowed = ET_PupilSize_Proc_1Hz.isna().rolling(window=spectrogram_windur, center=True).sum().dropna()
EC_Windowed = 100*(EC_Windowed/spectrogram_windur)


def assign_labels(a):
    if a >= 60:
        return 'EC/Drowsy'
    if a <= 40:
        return 'EO/Awake'
    return 'Mixed'


EC_Windowed_Labels = EC_Windowed.applymap(assign_labels)

# # Load PSD in Sleep Band
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
    file    = '{RUN}_mPP.Signal.{REGION}.Spectrogram_BandLimited.pkl'.format(RUN=run, REGION='V4_grp')
    path    = osp.join(DATA_DIR,sbj,run,file)
    aux     = pd.read_pickle(path)
    PSD_Windowed.loc[(windowed_time_index,sbj_run)] = aux['sleep'].values

PSD_Windowed.head()

# # Load Global Signal

GS_df = pd.DataFrame(columns=Manuscript_Runs)
for item in Manuscript_Runs:
    sbj,run  = item.split('_',1)
    aux_path = osp.join(DATA_DIR,sbj,run,'{RUN}_mPP.Signal.FB.1D'.format(RUN=run))
    if not osp.exists(aux_path):
        print(' + --> WARNING: Missing file: %s' % aux_path)
        continue
    aux_data = np.loadtxt(aux_path)
    assert aux_data.shape[0] == Nacq, "{file} has incorrect length {length}".format(file=aux_path, length=str(aux_data.shape[0]))
    GS_df[item] = aux_data
print(' + INFO: df.shape = %s' % str(GS_df.shape))  

GS_Windowed = GS_df.rolling(window=spectrogram_windur, center=True).std().dropna()
GS_Windowed.index = PSD_Windowed.index

GS_Windowed.head()

# # Generate Classification Problem

data = pd.DataFrame()
data['PSDsleep']    = PSD_Windowed.melt()['value']
data['GSamplitude'] = GS_Windowed.melt()['value']
data['EC']          = EC_Windowed.melt()['value']
data['Label']       = EC_Windowed_Labels.melt()['value']
print(data.shape)

data = data[(data['Label']=='EO/Awake') | (data['Label']=='EC/Drowsy')]
data = data.reset_index(drop=True)
print(data.shape)

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
input_features         = ['PSDsleep','GSamplitude','Control','PSD & GS']
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
    if metric == 'GSamplitude':
        X = data_umbalanced['GSamplitude'].values.reshape(-1, 1)
        y = data_umbalanced['Label']
    if metric == 'PSD & GS':
        X = data_umbalanced[['PSDsleep','GSamplitude']].values
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

# +
fig_umbal = plt.figure(constrained_layout=True, figsize=(15,6))
gs = fig_umbal.add_gridspec(2, 3)
fig_umbal_box    = fig_umbal.add_subplot(gs[:, 0])
fig_umbal_cm_psd = fig_umbal.add_subplot(gs[0, 1])
fig_umbal_cm_gs  = fig_umbal.add_subplot(gs[0, 2])
fig_umbal_cm_rnd = fig_umbal.add_subplot(gs[1, 1])
fig_umbal_cm_pgs = fig_umbal.add_subplot(gs[1, 2])

umbal_boxplot = sns.boxplot(data=scores_umbalanced, ax=fig_umbal_box, color='grey')
umbal_boxplot.set_title('(A) Prediction Accuracy', fontsize=18)
umbal_boxplot.set_xlabel('Input Features', fontsize=18)
umbal_boxplot.set_ylabel('% Accuracy', fontsize=18)
umbal_boxplot.set_ylim(0,0.9)
cm_umbal_psd       = pd.DataFrame(conf_mat_umbalanced.median(axis=1).loc['PSDsleep'].values, columns=['EC/Drowsy','EO/Awake'], index=['EC/Drowsy','EO/Awake'])
cm_umbal_psd_plot  = sns.heatmap(data=cm_umbal_psd,cmap='rocket',annot=True, vmin=0.3, vmax=1, ax=fig_umbal_cm_psd)
cm_umbal_psd_plot.set_xlabel('Predicted Label', fontsize=18)
cm_umbal_psd_plot.set_ylabel('True Label', fontsize=18)
cm_umbal_psd_plot.set_title('(B) PSDsleep', fontsize=18)

cm_umbal_gs        = pd.DataFrame(conf_mat_umbalanced.median(axis=1).loc['GSamplitude'].values, columns=['EC/Drowsy','EO/Awake'], index=['EC/Drowsy','EO/Awake'])
cm_umbal_gs_plot   = sns.heatmap(data=cm_umbal_gs,cmap='rocket',annot=True, vmin=0.3, vmax=1, ax=fig_umbal_cm_gs)
cm_umbal_gs_plot.set_xlabel('Predicted Label', fontsize=18)
cm_umbal_gs_plot.set_ylabel('True Label', fontsize=18)
cm_umbal_gs_plot.set_title('(C) GSamplitude', fontsize=18)

cm_umbal_rand      = pd.DataFrame(conf_mat_umbalanced.median(axis=1).loc['Control'].values, columns=['EC/Drowsy','EO/Awake'], index=['EC/Drowsy','EO/Awake'])
cm_umbal_rand_plot = sns.heatmap(data=cm_umbal_rand,cmap='rocket',annot=True, vmin=0.3, vmax=1, ax=fig_umbal_cm_rnd)
cm_umbal_rand_plot.set_xlabel('Predicted Label', fontsize=18)
cm_umbal_rand_plot.set_ylabel('True Label', fontsize=18)
cm_umbal_rand_plot.set_title('(C) Control', fontsize=18)

cm_umbal_pgs      = pd.DataFrame(conf_mat_umbalanced.median(axis=1).loc['PSD & GS'].values, columns=['EC/Drowsy','EO/Awake'], index=['EC/Drowsy','EO/Awake'])
cm_umbal_pgs_plot = sns.heatmap(data=cm_umbal_pgs,cmap='rocket',annot=True, vmin=0.3, vmax=1, ax=fig_umbal_cm_pgs)
cm_umbal_pgs_plot.set_xlabel('Predicted Label', fontsize=18)
cm_umbal_pgs_plot.set_ylabel('True Label', fontsize=18)
cm_umbal_pgs_plot.set_title('(D) PSDsleep + GSamplitude', fontsize=18)

print('Statistical Test for Umbalanced/Original Sample Scenario')
print('========================================================')
print('PSD vs. Control:  %s' % str(ttest_ind(scores_umbalanced['PSDsleep'],scores_umbalanced['Control'], alternative='two-sided')))
print('GS vs. Control:   %s' % str(ttest_ind(scores_umbalanced['GSamplitude'],scores_umbalanced['Control'], alternative='two-sided')))
print('GS+ vs. Control:   %s' % str(ttest_ind(scores_umbalanced['PSD & GS'],scores_umbalanced['Control'], alternative='two-sided')))
print('-----')
print('PSD vs. GS:   %s' % str(ttest_ind(scores_umbalanced['PSDsleep'],scores_umbalanced['GSamplitude'], alternative='two-sided')))
print('GS+ vs. GS:   %s' % str(ttest_ind(scores_umbalanced['PSD & GS'],scores_umbalanced['GSamplitude'], alternative='two-sided')))
print('GS+ vs. GS:   %s' % str(ttest_ind(scores_umbalanced['PSD & GS'],scores_umbalanced['PSDsleep'], alternative='two-sided')))
# -

fig_umbal.savefig('./figures/Fig11_LogisticRegression.png')

# ## Testing/Training Balanced Sample (Not reported)

min_num_samples = np.min([num_drowsy_umbalanced,num_awake_umbalanced])
data_balanced = pd.concat([data[data['Label']=='EC/Drowsy'].sample(min_num_samples), 
                           data[data['Label']=='EO/Awake'].sample(min_num_samples)], axis=0).reset_index(drop=True)

num_awake_bal  = (data_balanced['Label']=='EO/Awake').sum()
num_drowsy_bal = (data_balanced['Label']=='EO/Awake').sum()
num_total_bal  = data_balanced.shape[0]
print("++ INFO: Number of Awake Samples :  %d/%d [%.2f]" % (num_awake_bal, num_total_bal,100*num_awake_bal/num_total_bal))
print("++ INFO: Number of Drowsy Samples: %d/%d [%.2f]" % (num_drowsy_bal,num_total_bal,100*num_drowsy_bal/num_total_bal))

skf_balanced = StratifiedKFold(n_splits=k_folds, shuffle=True)

# %%time
input_features       = ['PSDsleep','GSamplitude','Control','PSD & GS']
scores_balanced      = pd.DataFrame(index=np.arange(k_folds))
predictions_balanced = {}
true_vals_balanced   = {}
conf_mat_balanced    = xr.DataArray(dims=['Metric','K-Fold','True Label','Predicted Label'],
                                      coords={'Metric':input_features,
                                              'K-Fold':np.arange(k_folds),
                                              'True Label':['EC/Drowsy','EO/Awake'],
                                              'Predicted Label':['EC/Drowsy','EO/Awake']})
for metric in input_features:
    print('++ Working on %s ...' % metric)
    if metric == 'Control':
        X_aux = data_balanced['PSDsleep'].values.reshape(-1, 1)
        y_aux = data_balanced['Label']
        X,y = shuffle(X_aux,y_aux)
    if metric == 'PSDsleep':
        X = data_balanced['PSDsleep'].values.reshape(-1, 1)
        y = data_balanced['Label']
    if metric == 'GSamplitude':
        X = data_balanced['GSamplitude'].values.reshape(-1, 1)
        y = data_balanced['Label']
    if metric == 'PSD & GS':
        X = data_balanced[['PSDsleep','GSamplitude']].values
        y = data_balanced['Label']
    aux_num_awake  = (y=='EO/Awake').sum()
    aux_num_drowsy = (y=='EC/Drowsy').sum()
    aux_num_total  = y.shape[0]
    print(" + INFO: Number of Awake Samples : %d/%d [%.2f]" % (aux_num_awake, aux_num_total,100*aux_num_awake/aux_num_total))
    print(" + INFO: Number of Drowsy Samples: %d/%d [%.2f]" % (aux_num_drowsy,aux_num_total,100*aux_num_drowsy/aux_num_total))
    for i, (train_index, test_index) in enumerate(skf_balanced.split(X, y)):
        log_reg = LogisticRegression(random_state=42, fit_intercept=True, max_iter=5000, n_jobs=-1)
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
        scores_balanced.loc[i,metric]    = log_reg.score(X_test,y_test)
        predictions_balanced[(metric,i)] = log_reg.predict(X_test)
        true_vals_balanced[(metric,i)]   = y_test 
        conf_mat_balanced.loc[metric,i,:,:]  = confusion_matrix(true_vals_balanced[(metric,i)],predictions_balanced[(metric,i)], normalize='pred') #PRECISION
scores_balanced = scores_balanced.infer_objects()

# +
fig_bal = plt.figure(constrained_layout=True, figsize=(15,6))
gs = fig_bal.add_gridspec(2, 3)
fig_bal_box    = fig_bal.add_subplot(gs[:, 0])
fig_bal_cm_psd = fig_bal.add_subplot(gs[0, 1])
fig_bal_cm_gs  = fig_bal.add_subplot(gs[0, 2])
fig_bal_cm_rnd = fig_bal.add_subplot(gs[1, 1])
fig_bal_cm_pgs = fig_bal.add_subplot(gs[1, 2])

bal_boxplot = sns.boxplot(data=scores_balanced, ax=fig_bal_box, palette=sns.color_palette("Paired", 4))
bal_boxplot.set_title('(A) Prediction Accuracy', fontsize=18)
bal_boxplot.set_xlabel('Input Features', fontsize=18)
bal_boxplot.set_ylabel('% Accuracy', fontsize=18)

cm_bal_psd       = pd.DataFrame(conf_mat_balanced.median(axis=1).loc['PSDsleep'].values, columns=['EC/Drowsy','EO/Awake'], index=['EC/Drowsy','EO/Awake'])
cm_bal_psd_plot  = sns.heatmap(data=cm_bal_psd,cmap='rocket',annot=True, vmin=0.3, vmax=1, ax=fig_bal_cm_psd)
cm_bal_psd_plot.set_xlabel('Predicted Label', fontsize=18)
cm_bal_psd_plot.set_ylabel('True Label', fontsize=18)
cm_bal_psd_plot.set_title('(B) PSDsleep', fontsize=18)

cm_bal_gs        = pd.DataFrame(conf_mat_balanced.median(axis=1).loc['GSamplitude'].values, columns=['EC/Drowsy','EO/Awake'], index=['EC/Drowsy','EO/Awake'])
cm_bal_gs_plot   = sns.heatmap(data=cm_bal_gs,cmap='rocket',annot=True, vmin=0.3, vmax=1, ax=fig_bal_cm_gs)
cm_bal_gs_plot.set_xlabel('Predicted Label', fontsize=18)
cm_bal_gs_plot.set_ylabel('True Label', fontsize=18)
cm_bal_gs_plot.set_title('(C) GSamplitude', fontsize=18)

cm_bal_rand      = pd.DataFrame(conf_mat_balanced.median(axis=1).loc['Control'].values, columns=['EC/Drowsy','EO/Awake'], index=['EC/Drowsy','EO/Awake'])
cm_bal_rand_plot = sns.heatmap(data=cm_bal_rand,cmap='rocket',annot=True, vmin=0.3, vmax=1, ax=fig_bal_cm_rnd)
cm_bal_rand_plot.set_xlabel('Predicted Label', fontsize=18)
cm_bal_rand_plot.set_ylabel('True Label', fontsize=18)
cm_bal_rand_plot.set_title('(C) Control', fontsize=18)

cm_bal_pgs      = pd.DataFrame(conf_mat_balanced.median(axis=1).loc['PSD & GS'].values, columns=['EC/Drowsy','EO/Awake'], index=['EC/Drowsy','EO/Awake'])
cm_bal_pgs_plot = sns.heatmap(data=cm_bal_pgs,cmap='rocket',annot=True, vmin=0.3, vmax=1, ax=fig_bal_cm_pgs)
cm_bal_pgs_plot.set_xlabel('Predicted Label', fontsize=18)
cm_bal_pgs_plot.set_ylabel('True Label', fontsize=18)
cm_bal_pgs_plot.set_title('(D) PSDsleep + GSamplitude', fontsize=18)
# -

print('Statistical Test for Balanced Sample Scenario')
print('===============================================')
print('PSD vs. Control:  %s' % str(ttest_ind(scores_balanced['PSDsleep'],scores_balanced['Control'], alternative='two-sided')))
print('GS vs. Control:   %s' % str(ttest_ind(scores_balanced['GSamplitude'],scores_balanced['Control'], alternative='two-sided')))
print('PSD vs. GS:   %s' % str(ttest_ind(scores_balanced['PSDsleep'],scores_balanced['GSamplitude'], alternative='two-sided')))
print('GS+ vs. GS:   %s' % str(ttest_ind(scores_balanced['PSD & GS'],scores_balanced['GSamplitude'], alternative='two-sided')))


