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
# This notebook generate the figures that help us understand how the different regression scenarios explain variance. We rely on the "Percent Variance Reduction" metric introduced by deB. Frederick et al. "Physiological Denoising of BOLD fMRI data using Regressor Interpolation at Progressive Time Delays (RIPTiDe)" processing of concurrent fMRI and near-infrared spectroscopy (NIRS)" 

# +
import pandas as pd
import numpy as np
import os.path as osp
import os
from utils.basics import get_available_runs
from utils.basics import DATA_DIR

import subprocess
from nilearn.masking import apply_mask
from nilearn.plotting import plot_stat_map, plot_epi
import hvplot.pandas
import panel as pn
import matplotlib.pyplot as plt
import holoviews as hv
from IPython.display import Markdown as md
# %matplotlib inline

fontsize_opts={'xlabel':14,'ylabel':14,'xticks':12,'yticks':12,'legend':12,'legend_title':12,'title':14}
# -

remove_HRa_scans = False

# ***
# ## 1. Generation of Swarm Jobs
# **Load list of scans to analize**

# +
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

# **Create logging folder for swarm jobs if missing**

if not osp.exists('./N08_PercentVarianceRemoved.logs'):
    print('++ INFO: Creating logging dir')
    os.mkdir('./N08_PercentVarianceRemoved.logs')

# **Create swarm file for all scans**
#
# This will generate the following outputs per scan:
#
# * ```${DATA_DUR}/${SBJ}/${RUN}/${RUN}_PVR.BASIC.nii.gz```: Percent removed variance by **motion parameters + 1st derivative (BASIC)** relative to bandpassing and polort **(Refernece)**
# * ```${DATA_DUR}/${SBJ}/${RUN}/${RUN}_PVR.COMPCOR.nii.gz```: Percent removed variance by **motion parameters + 1st derivative + CompCor (CompCor)** relative to bandpassing and polort **(Reference)**
# * ```${DATA_DUR}/${SBJ}/${RUN}/${RUN}_PVR.BASICpp.nii.gz```:  Percent removed variance by **motion parameters + 1st derivative + lagged iFV (BASICpp)** relative to bandpassing and polort **(Reference)**
# * ```${DATA_DUR}/${SBJ}/${RUN}/${RUN}_PVR.COMPCORpp.nii.gz```: Percent removed variance by **motion parameters + 1st derivative + CompCor + lagged iFV (COMPCORpp)** relative to bandpassing and polort **(Reference)**

# Create Swarm file for extracting representative power
# ==========================================================
os.system('echo "#swarm -f ./N08_PercentVarianceRemoved.SWARM.sh -g 4 -t 4 -b 30 --time=00:05:00 --partition quick,norm --logdir ./N08_PercentVarianceRemoved.logs" > ./N08_PercentVarianceRemoved.SWARM.sh')
for sbj_run in Manuscript_Runs:
    sbj,run  = sbj_run.split('_',1)
    os.system('echo "export SBJ={sbj} RUN={run}; sh ./N08_PercentVarianceRemoved.sh" >> ./N08_PercentVarianceRemoved.SWARM.sh'.format(sbj=sbj, run=run, ddir=DATA_DIR))

# ***
# ## 2. Check Swarm Jobs Generated ouputs

# %%time
for item in Manuscript_Runs:
    sbj,run = item.split('_',1)
    for suffix in ['PVR.BASIC.nii.gz', 'PVR.BASICpp.nii.gz', 'PVR.COMPCOR.nii.gz', 'PVR.COMPCORpp.nii.gz']:
        path = osp.join(DATA_DIR,sbj,run,'{run}_{suffix}'.format(run=run, suffix=suffix))
        if not osp.exists(path):
            print ('++ WARNING: Primary output missing [%s]' % path)

# ***
#
# ## 3. Create and visualize group-level results

# **Generate group-level (average) PRV maps for all regression schemes**

# + tags=[]
# %%time
for scans_name, scans_list in zip(['ALL','DROWSY','AWAKE'],[Manuscript_Runs,Drowsy_Runs,Awake_Runs]):
    print('++ Working with %s scans [%d scans on the list]'  % (scans_name,len(scans_list)))
    for pipeline in ['BASIC','BASICpp','COMPCOR','COMPCORpp']:
        command       = 'module load afni; 3dMean -overwrite -prefix {DATA_DIR}/ALL/{sl}_PVR_{pl}.nii.gz '.format(DATA_DIR=DATA_DIR, sl=scans_name, pl=pipeline) + \
                    ' '.join(['/data/SFIMJGC_HCP7T/HCP7T/{sbj}/{run}/{run}_PVR.{pl}.nii.gz'.format(pl=pipeline, sbj=item.split('_',1)[0],run=item.split('_',1)[1]) for item in scans_list])
        output        = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
        print(output.strip().decode())
        command       = 'module load afni; cd {DATA_DIR}/ALL; 3dcalc -overwrite -a {sl}_PVR_BASICpp.nii.gz -b {sl}_PVR_BASIC.nii.gz -expr "a-b" -prefix {sl}_PVR_BASICpp_minus_BASIC.nii.gz'.format(DATA_DIR=DATA_DIR, sl=scans_name)
        output        = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
        print(output.strip().decode())
        command       = 'module load afni; cd {DATA_DIR}/ALL; 3dcalc -overwrite -a {sl}_PVR_COMPCORpp.nii.gz -b {sl}_PVR_COMPCOR.nii.gz -expr "a-b" -prefix {sl}_PVR_COMPCORpp_minus_COMPCOR.nii.gz'.format(DATA_DIR=DATA_DIR, sl=scans_name)
        output        = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
        print(output.strip().decode())
        command       = 'module load afni; cd {DATA_DIR}/ALL; 3dcalc -overwrite -a {sl}_PVR_COMPCOR.nii.gz -b {sl}_PVR_BASIC.nii.gz -expr "a-b" -prefix {sl}_PVR_COMPCOR_minus_BASIC.nii.gz'.format(DATA_DIR=DATA_DIR, sl=scans_name)
        output        = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
        print(output.strip().decode())
        command       = 'module load afni; cd {DATA_DIR}/ALL; 3dcalc -overwrite -a {sl}_PVR_COMPCORpp.nii.gz -b {sl}_PVR_BASIC.nii.gz -expr "a-b" -prefix {sl}_PVR_COMPCORpp_minus_BASIC.nii.gz'.format(DATA_DIR=DATA_DIR, sl=scans_name)
        output        = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
        print(output.strip().decode())
# -

# **Load voxel-wise PRV values per regression scheme into dataframe (to explore differences in distribution of values across the brain)**

mask_path    = '/data/SFIMJGC_HCP7T/HCP7T/ALL/ROI.ALL.GM.mPP.mask.nii.gz'
files2labels = {'BASIC':'Basic','BASICpp':'Basic+','COMPCOR':'CompCor','COMPCORpp':'CompCor+'}
df_index     = pd.MultiIndex.from_product([['ALL','DROWSY','AWAKE'],files2labels.keys()], names=['Scan List','Pipeline'])
df           = pd.DataFrame(columns=df_index)
df.columns.name='Regressors:'
for scans_list_name in ['ALL','AWAKE','DROWSY']:
    for pipeline in ['BASIC','BASICpp','COMPCOR','COMPCORpp']:
        data_path = osp.join(DATA_DIR,'ALL','{sln}_PVR_{pipeline}.nii.gz'.format(pipeline=pipeline, sln=scans_list_name))
        df[(scans_list_name,pipeline)] = apply_mask(data_path,mask_path)

df_index = pd.MultiIndex.from_product([['ALL','DROWSY','AWAKE'],['CompCor - Basic','CompCor+ - Basic','Basic+ - Basic','CompCor+ - CompCor']], names=['Scan List','Pipeline'])
files2labels = {'BASICpp_minus_BASIC':'Basic+ - Basic','COMPCORpp_minus_COMPCOR':'CompCor+ - CompCor', 'COMPCOR_minus_BASIC':'CompCor - Basic','COMPCORpp_minus_BASIC':'CompCor+ - Basic'}
df_diffs = pd.DataFrame(columns=df_index)
for scans_list_name in ['ALL','AWAKE','DROWSY']:
    for pipeline in ['COMPCOR_minus_BASIC','COMPCORpp_minus_BASIC','BASICpp_minus_BASIC','COMPCORpp_minus_COMPCOR']:
        data_path = osp.join(DATA_DIR,'ALL','{sln}_PVR_{pipeline}.nii.gz'.format(pipeline=pipeline, sln=scans_list_name))
        df_diffs[(scans_list_name,files2labels[pipeline])] = apply_mask(data_path,mask_path)

# **Plot Voxel-wise Maps**

# +
fig, axs = plt.subplots(2,3,figsize=(20,5))
map_1 = plot_stat_map(osp.join(DATA_DIR,'ALL','DROWSY_PVR_BASIC.nii.gz'),     osp.join(DATA_DIR,'ALL','ALL_T1w_restore_brain.nii.gz'), cmap='jet', vmax=50, axes=axs[0,0], title='(A) Basic')
map_2 = plot_stat_map(osp.join(DATA_DIR,'ALL','DROWSY_PVR_BASICpp.nii.gz'),   osp.join(DATA_DIR,'ALL','ALL_T1w_restore_brain.nii.gz'), cmap='jet', vmax=50, axes=axs[0,1], title='(C) Basic+')
map_3 = plot_stat_map(osp.join(DATA_DIR,'ALL','DROWSY_PVR_COMPCOR.nii.gz'),   osp.join(DATA_DIR,'ALL','ALL_T1w_restore_brain.nii.gz'), cmap='jet', vmax=50, axes=axs[1,0], title='(B) CompCor')
map_4 = plot_stat_map(osp.join(DATA_DIR,'ALL','DROWSY_PVR_COMPCORpp.nii.gz'), osp.join(DATA_DIR,'ALL','ALL_T1w_restore_brain.nii.gz'), cmap='jet', vmax=50, axes=axs[1,1], title='(D) CompCor+')
map_5 = plot_stat_map(osp.join(DATA_DIR,'ALL','DROWSY_PVR_BASICpp_minus_BASIC.nii.gz'), osp.join(DATA_DIR,'ALL','ALL_T1w_restore_brain.nii.gz'), cmap='jet', vmax=15, axes=axs[0,2], title='(E) Difference: (C) - (A)')
map_5 = plot_stat_map(osp.join(DATA_DIR,'ALL','DROWSY_PVR_COMPCORpp_minus_COMPCOR.nii.gz'), osp.join(DATA_DIR,'ALL','ALL_T1w_restore_brain.nii.gz'), cmap='jet', vmax=10, axes=axs[1,2], title='(F) Difference: (D) - (B)')

fig.suptitle('% Variance Removed (PVR)  Maps per Regression Scenario', fontsize=20, ha='center')
#plt.close()

# +
fig, axs = plt.subplots(2,3,figsize=(20,5))
map_1 = plot_stat_map(osp.join(DATA_DIR,'ALL','ALL_PVR_BASIC.nii.gz'),       osp.join(DATA_DIR,'ALL','ALL_T1w_restore_brain.nii.gz'), cmap='jet', vmax=50, axes=axs[0,0], title='(A) Basic')
map_2 = plot_stat_map(osp.join(DATA_DIR,'ALL','ALL_PVR_COMPCOR.nii.gz'),     osp.join(DATA_DIR,'ALL','ALL_T1w_restore_brain.nii.gz'), cmap='jet', vmax=50, axes=axs[0,1], title='(C) CompCor')
map_3 = plot_stat_map(osp.join(DATA_DIR,'ALL','ALL_PVR_COMPCORpp.nii.gz'),   osp.join(DATA_DIR,'ALL','ALL_T1w_restore_brain.nii.gz'), cmap='jet', vmax=50, axes=axs[1,0], title='(B) CompCor+')
map_4 = plot_stat_map(osp.join(DATA_DIR,'ALL','ALL_PVR_COMPCOR_minus_BASIC.nii.gz'), osp.join(DATA_DIR,'ALL','ALL_T1w_restore_brain.nii.gz'), cmap='jet', vmax=15, axes=axs[0,2], title='(E) CompCor - Basic')
map_5 = plot_stat_map(osp.join(DATA_DIR,'ALL','ALL_PVR_COMPCORpp_minus_BASIC.nii.gz'), osp.join(DATA_DIR,'ALL','ALL_T1w_restore_brain.nii.gz'), cmap='jet', vmax=20, axes=axs[1,2], title='(F) CompCor+ - Basic')

fig.suptitle('% Variance Removed (PVR)  Maps per Regression Scenario', fontsize=20, ha='center')
#plt.close()
# -

# **Create Histogram with distributions**

hist_plots = df_diffs[('ALL',)].hvplot.kde(title='(G) Differences in PVR', fontsize=fontsize_opts, xlabel='% Variance Removed (PVR)', width=400).opts(legend_position='top_right', toolbar=None)
hist_plots

df_diffs.mean()

# + [markdown] tags=[]
# ### Load ROI Names, and create labels and locations for matrix display

# +
# Load Info in the label table file created in N04b
roi_info_path = '/data/SFIMJGC_HCP7T/HCP7T/ALL/Schaefer2018_200Parcels_7Networks_order_mPP.GM_Ribbon_order.txt'
roi_info_df   = pd.read_csv(roi_info_path,header=None,index_col=0, sep='\t')
# We drop the last column, as I have no idea what it means and seems to contains zeros
roi_info_df.drop([5],axis=1, inplace=True)
# Add an extra column with the index (comes handy when plotting later)
roi_info_df.reset_index(inplace=True)
# We rename the columns
roi_info_df.columns = ['ROI_ID','ROI_Name','R','G','B']
# We add a new column with informaiton about network membership for each ROI
roi_info_df['Network']      = [item.split('_')[2] for item in roi_info_df['ROI_Name']]
roi_info_df['Network_Hemi'] = [item.split('_')[1] for item in roi_info_df['ROI_Name']]
roi_info_df['Node_ID']      = [item.split('_',1)[1] for item in roi_info_df['ROI_Name']]
# Transform the colors original provided by Shaefer (as a tuple) into HEX code that HVPLOT likes
cols = []
for row in roi_info_df.itertuples():
    cols.append('#%02x%02x%02x' % (row.R, row.G, row.B))
roi_info_df['RGB']=cols
# Remove the original color columns (as those are redundant and not useful)
roi_info_df.drop(['R','G','B'],axis=1,inplace=True)

print('++ INFO: Number of ROIs according to label file = %d' % roi_info_df.shape[0])
roi_info_df.head()
# -

nw_color_map = {'LH-Vis':'purple',      'RH-Vis':'purple','Vis':'purple',
                   'LH-SomMot':'lightblue'  ,'RH-SomMot':'lightblue','SomMot':'lightblue',
                   'LH-DorsAttn':'green'    ,'RH-DorsAttn':'green','DorsAttn':'green',
                   'LH-SalVentAttn':'violet','RH-SalVentAttn':'violet','SalVentAttn':'violet',
                   'LH-Limbic':'yellow','RH-Limbic':'yellow','Limbic':'yellow',
                   'LH-Cont':'orange','RH-Cont':'orange','Cont':'orange',
                   'LH-Default':'red','RH-Default':'red','Default':'red'}
hm_color_map = {'LH':'grey','RH':'darkgrey'}

# %%time
df = {}
for pipeline in ['COMPCOR_minus_BASIC','COMPCORpp_minus_BASIC']:
    df[pipeline ] = pd.DataFrame(columns=Manuscript_Runs, index=roi_info_df.set_index(['Network','Node_ID']).index)
    for item in Manuscript_Runs:
        sbj,run  = item.split('_',1)
        path     = osp.join(DATA_DIR,sbj,run,'{run}_PVR.{pl}.1D'.format(run=run,pl=pipeline))
        data     = np.loadtxt(path)
        df[pipeline][item] = data

df['COMPCOR_minus_BASIC'][Drowsy_Runs].groupby('Network').mean().T.hvplot.box(color='Network', cmap=nw_color_map, legend=False, title='PVR: Basic+ - Basic') + \
df['COMPCORpp_minus_BASIC'][Drowsy_Runs].groupby('Network').mean().T.hvplot.box(color='Network', cmap=nw_color_map, legend=False, title='PVR: CompCor+ - CompCor')

df['COMPCOR_minus_BASIC'][Manuscript_Runs].groupby('Network').mean().T.mean().sort_values()



# %%time
df = {}
for pipeline in ['BASIC','COMPCOR']:
    df[pipeline ] = pd.DataFrame(columns=Manuscript_Runs, index=roi_info_df.set_index(['Network','Node_ID']).index)
    for item in Manuscript_Runs:
        sbj,run  = item.split('_',1)
        path     = osp.join(DATA_DIR,sbj,run,'{run}_PVR.{pl}pp_minus_{pl}.1D'.format(run=run,pl=pipeline))
        data     = np.loadtxt(path)
        df[pipeline][item] = data

df['BASIC'][Drowsy_Runs].groupby('Network').mean().T.hvplot.box(color='Network', cmap=nw_color_map, legend=False, title='PVR: Basic+ - Basic') + \
df['COMPCOR'][Drowsy_Runs].groupby('Network').mean().T.hvplot.box(color='Network', cmap=nw_color_map, legend=False, title='PVR: CompCor+ - CompCor')

df['BASIC'].droplevel('Network').T.hvplot.box(width=2000)

# ***
# ## 4. Additional non-reported results
#
# ### Voxel-wise correlation of data across pre-processing pipelines
#
# **Create group-level maps**

command_file = open('./N08_PercentVarianceRemoved_CorrMaps.sh','w+')
for sc_a,sc_b in [('Reference','BASIC'),('Reference','BASICpp'),('Reference','COMPCOR'),('Reference','COMPCORpp')]:
    command = '3dMean -overwrite -prefix {DATA_DIR}/ALL/DROWSY_corr_{sc_b}_2_{sc_a}.nii.gz '.format(sc_a=sc_a,sc_b=sc_b, DATA_DIR=DATA_DIR) + \
              ' '.join([osp.join(DATA_DIR, item.split('_',1)[0], item.split('_',1)[1], '{run}_{sc_b}_2_{sc_a}.nii.gz'.format(run=item.split('_',1)[1], sc_a=sc_a,sc_b=sc_b)) for item in drowsy_scans]) + '\n'
    command_file.write(command)
command_file.close()

# Run the following code on a termninal
#
# ```bash
# # cd /data/SFIMJGC_HCP7T/hcp7t_fv_sleep/Notebooks
# sh ./N08_PercentVarianceRemoved_CorrMaps.sh
# ```
#
# **Plot group-level maps**

fig, axs = plt.subplots(2,2,figsize=(14,5))
fig.suptitle('Voxel-wise correlation of data after different regression schemes relative to bandpass + polort only', fontsize=16, ha='center', fontweight='bold')
map_1    = plot_epi(osp.join(DATA_DIR,'ALL','DROWSY_corr_BASIC_2_Reference.nii.gz'),     cmap='jet', vmin=0.6, vmax=1, axes=axs[0,0], title='Motion')
map_2    = plot_epi(osp.join(DATA_DIR,'ALL','DROWSY_corr_BASICpp_2_Reference.nii.gz'),   cmap='jet', vmin=0.6, vmax=1, axes=axs[0,1], title='Motion + Lagged iFV')
map_3    = plot_epi(osp.join(DATA_DIR,'ALL','DROWSY_corr_COMPCOR_2_Reference.nii.gz'),   cmap='jet', vmin=0.6, vmax=1, axes=axs[1,0], title='Motion + CompCor')
map_4    = plot_epi(osp.join(DATA_DIR,'ALL','DROWSY_corr_COMPCORpp_2_Reference.nii.gz'), cmap='jet', vmin=0.6, vmax=1, axes=axs[1,1], title='Motion + CompCor + Lagged iFV')
plt.close()

# **Distribution of time-series correlation across regression schemes**

df = pd.DataFrame(columns=['Basic_2_Reference','Basic+_2_Reference','CompCor_2_Reference','CompCor+_2_Reference'])
df['Basic_2_Reference']   = apply_mask('/data/SFIMJGC_HCP7T/HCP7T/ALL/DROWSY_corr_BASIC_2_Reference.nii.gz','/data/SFIMJGC_HCP7T/HCP7T/ALL/ALL_EPI_FBmask.er2.nii.gz')
df['Basic+_2_Reference']  = apply_mask('/data/SFIMJGC_HCP7T/HCP7T/ALL/DROWSY_corr_BASICpp_2_Reference.nii.gz','/data/SFIMJGC_HCP7T/HCP7T/ALL/ALL_EPI_FBmask.er2.nii.gz')
df['CompCor_2_Reference'] = apply_mask('/data/SFIMJGC_HCP7T/HCP7T/ALL/DROWSY_corr_COMPCOR_2_Reference.nii.gz','/data/SFIMJGC_HCP7T/HCP7T/ALL/ALL_EPI_FBmask.er2.nii.gz')
df['CompCor+_2_Reference'] = apply_mask('/data/SFIMJGC_HCP7T/HCP7T/ALL/DROWSY_corr_COMPCORpp_2_Reference.nii.gz','/data/SFIMJGC_HCP7T/HCP7T/ALL/ALL_EPI_FBmask.er2.nii.gz')

hist_plots = ((df.hvplot.scatter(x='Basic_2_Reference',y='Basic+_2_Reference',   xlabel='R(BASIC,Reference)', ylabel='R(BASIC+,Reference)', datashade=True, aspect='square', fontsize=fontsize_opts) * hv.Curve([[0, 0], [1, 1]]).opts(line_dash='dashed', color='black')) + \
(df.hvplot.scatter(x='Basic_2_Reference',y='CompCor_2_Reference',  xlabel='R(BASIC,Reference)', ylabel='R(CompCor,Reference)', datashade=True, aspect='square', fontsize=fontsize_opts) * hv.Curve([[0, 0], [1, 1]]).opts(line_dash='dashed', color='black')) + \
(df.hvplot.scatter(x='Basic_2_Reference',y='CompCor+_2_Reference', xlabel='R(BASIC,Reference)', ylabel='R(CompCor+,Reference)', datashade=True, aspect='square', fontsize=fontsize_opts) * hv.Curve([[0, 0], [1, 1]]).opts(line_dash='dashed', color='black'))).opts(toolbar=None, title='Relation to non-regression model decreases when we take into account lagged iFV')

combined_figure = pn.Row(fig, hist_plots)

combined_figure.save('./figures/Correlation_across_regression_schemes.png')

# ![](./figures/Correlation_across_regression_schemes.png)
