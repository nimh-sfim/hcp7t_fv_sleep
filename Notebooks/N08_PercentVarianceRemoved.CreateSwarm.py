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
# %matplotlib inline

fontsize_opts={'xlabel':14,'ylabel':14,'xticks':12,'yticks':12,'legend':12,'legend_title':12,'title':16}
# -

# ***
# ## 1. Generation of Swarm Jobs
# **Load list of scans to analize**

Manuscript_Runs = get_available_runs(when='final',type='all')
print('++ INFO: Number of scans: %d scans' % len(Manuscript_Runs))

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
os.system('echo "#swarm -f ./N08_PercentVarianceRemoved.SWARM.sh -g 4 -t 4 -b 30 --time=00:05:00 --partition quick,norm --logdir ./N08_PercentVarianceRemoved.logs')
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
#
# **Load list of scans labeled drowsy and awake**
#
# > **NOTE:** We only report results for drowsy scans

drowsy_scans    = get_available_runs(when='final', type='drowsy')
awake_scans     = get_available_runs(when='final', type='awake')
print('++ INFO: Number of Runs [Drowsy] = %d' % len(drowsy_scans))
print('++ INFO: Number of Runs [Awake]  = %d' % len(awake_scans))

# **Generate group-level (average) PRV maps for all regression schemes**

# %%time
for scenario in ['BASIC','BASICpp','COMPCOR','COMPCORpp']:
    command       = 'module load afni; 3dMean -overwrite -prefix {DATA_DIR}/ALL/DROWSY_PVR_{scenario}.nii.gz '.format(DATA_DIR=DATA_DIR, scenario=scenario) + \
                    ' '.join(['/data/SFIMJGC_HCP7T/HCP7T/{sbj}/{run}/{run}_PVR.{scenario}.nii.gz'.format(scenario=scenario, sbj=item.split('_',1)[0],run=item.split('_',1)[1]) for item in drowsy_scans])
    output        = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
    print(output.strip().decode())

# **Load voxel-wise PRV values per regression scheme into dataframe (to explore differences in distribution of values across the brain)**

mask_path = '/data/SFIMJGC_HCP7T/HCP7T/ALL/ROI.ALL.GM.mPP.mask.nii.gz'
files2labels={'BASIC':'Motion','BASICpp':'Motion + Lagged iFV','COMPCOR':'Motion + CompCor','COMPCORpp':'Motion + CompCor + Lagged iFV'}
df = pd.DataFrame(columns=['Motion','Motion + Lagged iFV','Motion + CompCor','Motion + CompCor + Lagged iFV'])
df.columns.name='Regressors:'
for scenario in ['BASIC','BASICpp','COMPCOR','COMPCORpp']:
    data_path = osp.join(DATA_DIR,'ALL','DROWSY_PVR_{scenario}.nii.gz'.format(scenario=scenario))
    df[files2labels[scenario]] = apply_mask(data_path,mask_path)

# **Create Histogram with distributions**

hist_plots = df.hvplot.kde(title='(E) % Variance Removed (PVR) in GM Ribbon - Drowsy Scans', fontsize=fontsize_opts, xlabel='% Variance Removed (PVR)').opts(legend_position='right', toolbar=None)

# **Plot Voxel-wise Maps**

fig, axs = plt.subplots(2,2,figsize=(14,5))
map_1 = plot_stat_map(osp.join(DATA_DIR,'ALL','DROWSY_PVR_BASIC.nii.gz'),     osp.join(DATA_DIR,'ALL','ALL_T1w_restore_brain.nii.gz'), cmap='jet', vmax=50, axes=axs[0,0], title='(A) Motion')
map_2 = plot_stat_map(osp.join(DATA_DIR,'ALL','DROWSY_PVR_BASICpp.nii.gz'),   osp.join(DATA_DIR,'ALL','ALL_T1w_restore_brain.nii.gz'), cmap='jet', vmax=50, axes=axs[0,1], title='(B) Motion + Lagged iFV')
map_3 = plot_stat_map(osp.join(DATA_DIR,'ALL','DROWSY_PVR_COMPCOR.nii.gz'),   osp.join(DATA_DIR,'ALL','ALL_T1w_restore_brain.nii.gz'), cmap='jet', vmax=50, axes=axs[1,0], title='(C) Motion + CompCor')
map_4 = plot_stat_map(osp.join(DATA_DIR,'ALL','DROWSY_PVR_COMPCORpp.nii.gz'), osp.join(DATA_DIR,'ALL','ALL_T1w_restore_brain.nii.gz'), cmap='jet', vmax=50, axes=axs[1,1], title='(D) Motion + CompCor + Lagged iFV')
fig.suptitle('% Variance Removed (PVR)  Maps per Regression Scenario', fontsize=20, ha='center')
plt.close()

combined_figure = pn.Row(fig,pn.Column(pn.pane.Markdown('#'),hist_plots))

combined_figure.save('./figures/PRV.png')

# ![](./figures/PRV.png)

# **Print median PRV per regression scheme**

df.median()

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
