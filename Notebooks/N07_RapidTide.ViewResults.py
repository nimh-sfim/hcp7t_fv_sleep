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
# This notebook runs rapidtide in all runs.
#
# 1. Create ouput folder
#
# 2. Runs rapidtide following the BASICnobpf pipeline. 
#
# 3. Removes a lot of unnecessary outputs from rapidtide.
#
# 4. Create Thresholded versions of the lag and corr maps
#
# # Primary Outputs
#
# * ```${DATA_DIR}/${SBJ}/${RUN}/${RUN}_BASICnobpf.rapidtide/${RUN}_BASICnobpf_corrout.nii.gz```:
# * ```${DATA_DIR}/${SBJ}/${RUN}/${RUN}_BASICnobpf.rapidtide/${RUN}_BASICnobpf_lagtimes.nii.gz```:
# * ```${DATA_DIR}/${SBJ}/${RUN}/${RUN}_BASICnobpf.rapidtide/${RUN}_BASICnobpf_p_lt_0p005_mask.nii.gz```:
# * ```${DATA_DIR}/${SBJ}/${RUN}/${RUN}_BASICnobpf.rapidtide/${RUN}_BASICnobpf_corrout.masked_0p005.nii.gz```:
# * ```${DATA_DIR}/${SBJ}/${RUN}/${RUN}_BASICnobpf.rapidtide/${RUN}_BASICnobpf_lagtimes.masked_0p005.nii.gz```:
#
#

# +
import os
import os.path as osp
import numpy as np
import pandas as pd
import subprocess

from nilearn.plotting import plot_stat_map
from nilearn.image    import load_img, index_img

from utils.variables import Resources_Dir, DATA_DIR
from utils.basics    import get_available_runs
ALL_DIR  = '/data/SFIMJGC_HCP7T/HCP7T/ALL/'

import hvplot.pandas
import panel as pn
import matplotlib.pyplot as plt

from nilearn.masking import apply_mask
from IPython.display import Markdown as md

fontsize_opts={'xlabel':14,'ylabel':14,'xticks':12,'yticks':12,'legend':12,'legend_title':12,'title':14}
# -

remove_HRa_scans = False
p_threshold      = '0p050'

# ***
# # 1. Load list of runs

# +
# %%time
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

# + [markdown] tags=[]
# ***
# # 4. Create Group-level Lag Maps (p<0.05)
#
# The steps involved in this calculation are:
#
# 1. Concatenate lagmaps for all subjects into a single file (we do this to be able to use 3dTstat in the next step)
# 2. Compute the nzmean and nzmedian across all subjects' lagmaps
# 3. Remove voxels that did not reach significance (p<0.05) for at least half the sample.
#
# > **Main Output:**  /data/SFIMJGC_HCP7T/HCP7T/ALL/ALL.rapidtide_BASICnobpf_p_lt_0p050.lagtimes_summary.nii.gz
# -

# ### 4.1 Create lists with the paths to all necessary files
#
# Here we gather files names into three dictionary objects:
#
# * masked_corrout: cross-correlation traces only for voxels that passed the threshold
#
# * masked_lagtimes: lagtime maps only for voxels that passed the threshold
#
# * masks: mask with voxels that passed the threshold

# +
# %%time
# Gather all mask and masked lagtime files for drwosy subjects for the different p-values
# =======================================================================================
masks, masked_lagtimes,masked_corrout = list(),list(),list()
for item in Drowsy_Runs:
    sbj,run = item.split('_',1)
    mask_path           = osp.join(DATA_DIR,sbj,run,run+'_BASICnobpf.rapidtide',run+'_BASICnobpf_p_lt_{pval}_mask.nii.gz'.format(pval=p_threshold))
    masked_lagtime_path = osp.join(DATA_DIR,sbj,run,run+'_BASICnobpf.rapidtide',run+'_BASICnobpf_lagtimes.masked_{pval}.nii.gz'.format(pval=p_threshold))
    masked_corrout_path = osp.join(DATA_DIR,sbj,run,run+'_BASICnobpf.rapidtide',run+'_BASICnobpf_corrout.masked_{pval}.nii.gz'.format(pval=p_threshold))
    if osp.exists(mask_path):
        masks.append(mask_path)
    else:
        print(' ++ WARNING: Missing mask [%s]' % mask_path)
    if osp.exists(masked_lagtime_path):
        masked_lagtimes.append(masked_lagtime_path)
    else:
        print('++ WARNING: Missing lagtime map [%s]' % masked_lagtime_path)
    if osp.exists(masked_corrout_path):
        masked_corrout.append(masked_corrout_path)
    else:
        print('++ WARNING: Missing corrout map [%s]' % masked_corrout_path)
    
print('++ INFO: Number of files gathered: (corrout=%d, lagtimes=%d, mask=%d)' % (len(masked_corrout),len(masked_lagtimes),len(masks)))
# -

# ### 4.2. Create group masks with count of how many runs passed the threshold in each voxel
#
# For each subject, rapidtide generated one file with a mask informing about which voxels have a significant cross-correlation result (e.g., SBJ_DIR/RUN_DIR/rfMRI_REST2_AP_BASICnobpf.rapidtide/rfMRI_REST2_AP_BASICnobpf_p_lt_0p050_mask.nii.gz). 
#
# We average those to create a single significance mask (**FINAL_0p050_mask_path**).

# %%time
FINAL_0p050_mask_path = osp.join(DATA_DIR,'ALL','ALL.rapidtide_BASICnobpf_p_lt_{pval}_mask.{ss}.nii.gz'.format(pval=p_threshold,ss=scan_selection))
command               = 'module load afni; 3dMean -overwrite -prefix {prefix} {input_files}; 3dcalc -overwrite -a {prefix} -expr "ispositive(a-0.4)" -prefix {prefix}'.format(prefix=FINAL_0p050_mask_path, input_files=' '.join(masks))
output                = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
print(output.strip().decode())
print('++ INFO: Final significance mask for rapidtide (p<0.05) --> %s' % FINAL_0p050_mask_path)

# %matplotlib inline
t1_bg    = load_img(osp.join(ALL_DIR,'ALL_T1w_restore_brain.nii.gz'))
mask_img = load_img(FINAL_0p050_mask_path)
plot_stat_map(mask_img,t1_bg,alpha=0.8, cmap='RdBu_r', title='% Scans with Rapidtide p < 0.05');

# ### 4.3 Create Voxel-wise Group-level Lag Map
#
# The steps involved in this calculation are:
#
# 1. Concatenate lagmaps for all subjects into a single file (we do this to be able to use 3dTstat in the next step)
# 2. Compute the nzmean and nzmedian across all subjects' lagmaps
# 3. Remove voxels that did not reach significance (p<0.05) for at least half the sample.

# + tags=[]
# %%time
FINAL_0p050_lagmap_path = osp.join(DATA_DIR,'ALL','ALL.rapidtide_BASICnobpf_p_lt_{pval}.lagtimes_summary.{ss}.nii.gz'.format(pval=p_threshold, ss=scan_selection))
aux_output1_path        = osp.join(DATA_DIR,'ALL','ALL.rapidtide_BASICnobpf_p_lt_{pval}_all.{ss}.nii.gz'.format(pval=p_threshold, ss=scan_selection))
aux_output2_path        = osp.join(DATA_DIR,'ALL','ALL.rapidtide_BASICnobpf_p_lt_{pval}.lagtimes_summary_nomask.{ss}.nii.gz'.format(pval=p_threshold, ss=scan_selection))
command                 = 'module load afni; \
                           echo "++ INFO: Combine all lag maps from all subjects into a single file"; \
                           echo "   ==============================================================="; \
                           3dTcat  -overwrite -prefix {output_1} {input_files}; \
                           echo "++ INFO: Compute basic statistics across subjects" ; \
                           echo "   =============================================="; \
                           3dTstat -overwrite -nzmean -nzmedian -mean -median -prefix {output_2} {output_1}; \
                           echo "++ INFO: Restrict group maps to voxels that were significant in at least half the sample"; \
                           echo "   ====================================================================================="; \
                           3dcalc  -overwrite -a {output_2} -b {mask} -expr "a*b" -prefix {output_3}; rm {output_1}; \
                           echo "++ INFO: Trick to avoid issue regarding 5th dimension in nilearn"; \
                           echo "   ============================================================="; \
                           3dTcat -overwrite -prefix {output_3} {output_3};'.format(output_1=aux_output1_path,
                                                                                                                                output_2   = aux_output2_path,
                                                                                                                                output_3   = FINAL_0p050_lagmap_path,
                                                                                                                                mask       = FINAL_0p050_mask_path,
                                                                                                                                input_files= ' '.join(masked_lagtimes))
output = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
print(output.strip().decode())
print('++ INFO: Final significance mask for rapidtide (p<0.05) --> %s' % FINAL_0p050_lagmap_path)
# -

lagmap_img = load_img(index_img(FINAL_0p050_lagmap_path,0))
plot_stat_map(lagmap_img,t1_bg,alpha=0.8, cmap='jet', title='Group Level Lag maps (Non-zero Mean across subjects)');

FINAL_0p050_lagmap_path = osp.join(DATA_DIR,'ALL','ALL.rapidtide_BASICnobpf_p_lt_{pval}.lagtimes_summary.{ss}.nii.gz'.format(pval='0p050',ss=scan_selection))
print('++ INFO: Final significance map for rapidtide (p<0.05)          --> %s' % FINAL_0p050_mask_path)
print('++ INFO: Final average lagtime map across all subjects (p<0.05) --> %s' % FINAL_0p050_lagmap_path)

text="""
Now, we can create images of delay maps using afni, as follows:

```bash
module load afni

cd /data/SFIMJGC_HCP7T/HCP7T/ALL

afni -DAFNI_IMAGE_LABEL_MODE=1 -DAFNI_IMAGE_LABEL_SIZE=4 \\
-com "OPEN_WINDOW A.axialimage mont=10x1:18" -com "CLOSE_WINDOW A.coronalimage" \\
-com "SWITCH_UNDERLAY T1w_restore_brain.MEAN.abox.nii.gz" \\
-com "SWITCH_OVERLAY ALL.rapidtide_BASICnobpf_p_lt_0p050.lagtimes_summary.all.nii.gz" \\
-com "SET_SUBBRICKS A 0 1 1" -com "SET_PBAR_ALL A.-99 9 GoogleTurbo" -com "SET_XHAIRS A.OFF" \\
-com "OPEN_WINDOW A.sagittalimage mont=7x1:24" \\
-com "SAVE_PNG A.axialimage /data/SFIMJGC_HCP7T/hcp7t_fv_sleep/Notebooks/figures/Revision1_Figure12_PanelA_Axial.{ss}.png" \\
-com "SAVE_PNG A.sagittalimage /data/SFIMJGC_HCP7T/hcp7t_fv_sleep/Notebooks/figures/Revision1_Figure12_PanelA_Sagittal.{ss}.png" -com "QUITT"
```
""".format(ss=scan_selection)
md("%s"%(text))

pn.Column(pn.pane.PNG('/data/SFIMJGC_HCP7T/hcp7t_fv_sleep/Notebooks/figures/Revision1_Figure12_PanelA_Axial.{ss}.png'.format(ss=scan_selection)),
          pn.pane.PNG('/data/SFIMJGC_HCP7T/hcp7t_fv_sleep/Notebooks/figures/Revision1_Figure12_PanelA_Sagittal.{ss}.png'.format(ss=scan_selection)))

# Above we generated 4 different types of average lag maps--namely mean, median, non-zero mean and non-zero median. In the paper we report the non-zero median case. To make that clear, we now generate a new
# file that only contains that map.

# +
command = "module load afni; \
               cd /data/SFIMJGC_HCP7T/HCP7T/ALL; \
               3dcalc -overwrite -a ALL.rapidtide_BASICnobpf_p_lt_{pval}.lagtimes_summary.{ss}.nii.gz[1] -expr 'a' -prefix Revision1_Figure12_LagMaps.{ss}.nii.gz".format(pval=pval,ss=scan_selection)

output = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
print(output.strip().decode())
# -

# ### Create Histogram of Delays across the whole brain

# Paths to the mask and lagtimes
hist_mask_path = osp.join(DATA_DIR,'ALL','ALL.rapidtide_BASICnobpf_p_lt_0p050_mask.{ss}.nii.gz'.format(ss=scan_selection))
hist_lags_path = osp.join(DATA_DIR,'ALL','Revision1_Figure12_LagMaps.{ss}.nii.gz'.format(ss=scan_selection))

# Load all lagtimes into a single dataframe (index = voxels)
lags_across_brain = pd.DataFrame(apply_mask(hist_lags_path,hist_mask_path), columns=['Lag [seconds]'])

fontsize_opts={'xlabel':18,'ylabel':18,'xticks':16,'yticks':16}
(lags_across_brain.hvplot.hist(normed=True, bins=30, c='gray', width=500, height=500) * lags_across_brain.hvplot.kde(fontsize=fontsize_opts, ylabel='Density [proportion of voxels]', c='gray')).opts(toolbar=None)

print('++ INFO: Lag Range in the brain (5th to 95th quantiles) = [%.2f sec, %.2f sec]' % (lags_across_brain.quantile(0.05), lags_across_brain.quantile(0.95)))
print('++ INFO: Lag Range in the brain (min to max values)     = [%.2f sec, %.2f sec]' % (lags_across_brain.min(), lags_across_brain.max()))

# ### Histograms for specific regions (supplementary figure 5)

# First, we bring the Eickhoff SB et al. atlas provided by AFNI into the same grid as our group results:
#
# ```bash
# module load afni
# # cd /data/SFIMJGC_HCP7T/HCPt7T/ALL
# 3dcopy /usr/local/apps/afni/current/linux_centos_7_64/MNI_caez_mpm_22+tlrc. MNI_caez_mpm_22.nii.gz
# 3dresample -input MNI_caez_mpm_22.nii.gz -rmode NN -master ALL_ROI.V4lt.mPP.nii.gz -prefix MNI_caez_mpm_22.nii.gz
# 3drefit -copytables /usr/local/apps/afni/current/linux_centos_7_64/MNI_caez_mpm_22+tlrc. MNI_caez_mpm_22.nii.gz
#
# 3dcopy /usr/local/apps/afni/current/linux_centos_7_64/MNI_caez_ml_18+tlrc. MNI_caez_ml_18.nii.gz
# 3dresample -input MNI_caez_ml_18.nii.gz -rmode NN -master ALL_ROI.V4lt.mPP.nii.gz -prefix MNI_caez_ml_18.nii.gz
# 3drefit -copytables /usr/local/apps/afni/current/linux_centos_7_64/MNI_caez_ml_18+tlrc. MNI_caez_ml_18.nii.gz
# ```
#
# Next we extract three ROIs of interest:
#
# ```bash
# 3dcalc -a MNI_caez_mpm_22.nii.gz -expr 'equals(a,181) + equals(a,210) + equals(a,213) + equals(a,218)' -prefix MNI_caez_mpm_22.Auditory.nii.gz
# 3dcalc -a MNI_caez_ml_18.nii.gz -expr 'equals(a,57) + equals(a,58)' -prefix MNI_caez_ml_18.Somatosensory.nii.gz
# 3dcalc -a MNI_caez_ml_18.nii.gz -expr 'equals(a,7) + equals(a,8) + equals(a,3) + equals(a,4) + within(a,11,16)' -prefix MNI_caez_ml_18.Frontal.nii.gz
# ```
#
# Finally, we write into a text file all the lag values (different from zero) for voxels in each of the ROIs:
#
# ```bash
# 3dmaskdump -nozero -noijk -mask MNI_caez_mpm_22.Auditory.nii.gz ALL.rapidtide_BASICnobpf_p_lt_0p050.lagtimes_summary.all.nii.gz[1] >> Lags_in_Auditory.1D
# 3dmaskdump -nozero -noijk -mask MNI_caez_ml_18.Somatosensory.nii.gz     ALL.rapidtide_BASICnobpf_p_lt_0p050.lagtimes_summary.all.nii.gz[1] >> Lags_in_Somatosensory.1D
# 3dmaskdump -nozero -noijk -mask MNI_caez_ml_18.Frontal.nii.gz   ALL.rapidtide_BASICnobpf_p_lt_0p050.lagtimes_summary.all.nii.gz[1] >> Lags_in_Frontal.1D
#
# ```
#

lags_ss = pd.DataFrame(np.loadtxt(osp.join(DATA_DIR,'ALL','Lags_in_Somatosensory.1D')))
lags_au = pd.DataFrame(np.loadtxt(osp.join(DATA_DIR,'ALL','Lags_in_Auditory.1D')))
lags_fr = pd.DataFrame(np.loadtxt(osp.join(DATA_DIR,'ALL','Lags_in_Frontal.1D')))

fontsize_opts={'xlabel':18,'ylabel':18,'xticks':16,'yticks':16,'legend':16,'legend_title':16,'title':16}
(lags_ss.hvplot.kde(label='Somatosensory Cortex', c='r', fontsize=fontsize_opts, ylabel='Densisty\n[proportion of voxels]', xlabel='Lags [seconds]') * \
lags_au.hvplot.kde(label='Auditory Cortex', c='orange') * \
lags_fr.hvplot.kde(label='Frontal Cortex', c='blue')).opts(toolbar=None)

# + [markdown] tags=[]
# ***
#
# # 5. Generation of Group-level Cross-Correlation traces
#
# For each lag, we compute the average across all subjects of the cross-correlation traces for that particular lag. For each voxel, we only use data from subjects for which that particular voxel was significant according to rapidtide.
# -

pval = '0p050'

# %%time
nvols=31
# First we need to do this on a sub-brick by sub-brick fashion, then we will put them together with 
# 3dTcat or 3dTbucket
# =================================================================================================
for pval in ['0p050']:
    print('++++++++++++ ----------------------- %s ---------------------- ++++++++++++' % (pval))
    for v in np.arange(nvols):
        print(v,end='..')
        input_files  = ' '.join([file+'[{v}]'.format(v=v) for file in masked_corrout])
        output1_path = osp.join(DATA_DIR,'ALL','rm.ALL.rapidtide_BASICnobpf_p_lt_{pval}_all.v{v}.{ss}.nii.gz'.format(pval=pval,v=str(v).zfill(2),ss=scan_selection))
        output2_path = osp.join(DATA_DIR,'ALL','rm.ALL.rapidtide_BASICnobpf_p_lt_{pval}.corrout_summary.v{v}.{ss}.nii.gz'.format(pval=pval,v=str(v).zfill(2),ss=scan_selection))
        mask_path    = osp.join(DATA_DIR,'ALL','ALL.rapidtide_BASICnobpf_p_lt_{pval}_mask.{ss}.nii.gz'.format(pval=pval,ss=scan_selection))
        command = 'module load afni;                                                            \
               3dTcat -overwrite -prefix {output_1} {input_files};                              \
               3dcalc -overwrite -a {output_1} -expr "atanh(a)" -prefix  {output_1};             \
               3dTstat -overwrite -nzmean -prefix {output_2} {output_1};                         \
               3dcalc -overwrite -a {output_2} -expr "tanh(a)" -prefix {output_2};               \
               3dcalc -overwrite \
                      -a {output_2} \
                      -b {mask} \
                      -expr "a*b" \
                      -prefix {output_2}; \
              rm {output_1}'.format(output_1=output1_path,
                                    output_2=output2_path,
                                    mask=mask_path,
                                    input_files=input_files)
        output = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
        print(output.strip().decode())

# We now put together the results from the previous cell into a single file with all the datapoints. This code sets the time origin to -15s (which is what the original corrout datasets had). As TR is 1s, this does not need fixing as of now --> it might if you take out the -O 1 from rapidtide and or the data has a different TR
#

# +
# %%time
if not osp.exists('/data/SFIMJGC_HCP7T/HCP7T/ALL/temp_lag_results'):
    os.mkdir('/data/SFIMJGC_HCP7T/HCP7T/ALL/temp_lag_results')
    print("++ INFO: Temporary folder created: /data/SFIMJGC_HCP7T/HCP7T/ALL/temp_lag_results")

command = 'module load afni; \
               cd /data/SFIMJGC_HCP7T/HCP7T/ALL; \
               3dTcat -overwrite -prefix ALL.rapidtide_BASICnobpf_p_lt_{pval}.corrout_summary.nzmean.{ss}.nii.gz   rm.ALL.rapidtide_BASICnobpf_p_lt_{pval}.corrout_summary.v??.{ss}.nii.gz[0]; \
               3drefit -Torg -15 ALL.rapidtide_BASICnobpf_p_lt_{pval}.corrout_summary.nzmean.{ss}.nii.gz; \
               mv rm.ALL.rapidtide_BASICnobpf_p_lt_{pval}.corrout_summary.v??.{ss}.nii.gz temp_lag_results'.format(pval=pval,ss=scan_selection)
output = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
print(output.strip().decode())
print('============================================================================')
print('++ INFO: Output file: ALL.rapidtide_BASICnobpf_p_lt_{pval}.corrout_summary.nzmean.{ss}.nii.gz'.format(pval=pval, ss=scan_selection))
# -

# ***
#
# # 6. Show cross-correlation traces for a few represntative voxels (those will be in a manuscript figure)
#

# +
ijk_locs = [(56,52,24),(64,55,54),(64,97,46),(48,27,49),(33,65,56),(42,62,80)]
for i,j,k in ijk_locs:
    command = "module load afni; \
               cd /data/SFIMJGC_HCP7T/HCP7T/ALL; \
               3dmaskdump -quiet -noijk -ibox {i} {j} {k} ALL.rapidtide_BASICnobpf_p_lt_0p050.corrout_summary.nzmean.{ss}.nii.gz | tr -s ' ' '\n' > LagProfile_I{i}_J{j}_K{k}.{ss}.1D".format(i=i,j=j,k=k, ss=scan_selection)
    output = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
    print(output.strip().decode())

command = 'module load afni; \
           cd /data/SFIMJGC_HCP7T/HCP7T/ALL; \
           3dROIstats -quiet -mask {mask} ALL.rapidtide_BASICnobpf_p_lt_{pval}.corrout_summary.nzmean.{ss}.nii.gz > ALL.rapidtide_BASICnobpf_p_lt_{pval}.corrout_summary.nzmean.{ss}.1D'.format(mask=FINAL_0p050_mask_path,pval=p_threshold, ss=scan_selection)
output = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
print(output.strip().decode())                                    
# -

location_labels = ['Whole Brain', '(1) 4th Vent.', '(2) Post. Lat. Vent.', '(3) Ant. Lat. Vent.', '(4) PVC',     '(5) PAC',     '(6) PMC']
location_prefix = ['WB',          'I56_J52_K24',   'I64_J55_K54',          'I64_J97_K46',         'I48_J27_K49', 'I33_J65_K56', 'I42_J62_K80']
n_plots         = len(location_prefix)
data_dict={}
data = None
for i in np.arange(n_plots):
    if i == 0:
        path = '/data/SFIMJGC_HCP7T/HCP7T/ALL/ALL.rapidtide_BASICnobpf_p_lt_0p050.corrout_summary.nzmean.{ss}.1D'.format(ss=scan_selection)
    else:
        path = '/data/SFIMJGC_HCP7T/HCP7T/ALL/LagProfile_{ii}.{ss}.1D'.format(ii=location_prefix[i],ss=scan_selection)
    df   = pd.DataFrame(np.loadtxt(path)).reset_index()
    df.columns=['Lag (s)','Cross-Correlation']
    df['Lag (s)'] = df['Lag (s)'] - 15
    df['Location'] = location_labels[i]
    if data is None:
        data = df
    else:
        data = pd.concat([data,df],axis=0)
    data_dict[location_labels[i]] = df


data.groupby('Location').min()

plots = None
for i in np.arange(n_plots):
    if plots is None:
        plots = data_dict[location_labels[i]].hvplot(x='Lag (s)', width=200, height=250, c='b', ylim=(-0.5,1), title=location_labels[i], grid=True).opts(toolbar=None)
    else:
        plots = plots + data_dict[location_labels[i]].hvplot(x='Lag (s)', width=200, height=250, c='k', title=location_labels[i], grid=True).opts(toolbar=None)

fig8_panelsBC = plots.cols(7).opts(toolbar=None)
fig8_panelsBC

lagmap_path       = osp.join(ALL_DIR,'Figure8_LagMaps.{ss}.nii.gz'.format(ss=scan_selection))
lagmap_img        = load_img(lagmap_path)

fig, axs = plt.subplots(2,1,figsize=(20,7))
fig8_panelAsag = plot_stat_map(lagmap_img,t1_bg,cmap='jet', vmax=9, display_mode='x', axes=axs[1], cut_coords=[-53,-35,-18,0,18,35,53])
fig8_panelAaxi = plot_stat_map(lagmap_img,t1_bg,cmap='jet', vmax=9, display_mode='z', axes=axs[0], cut_coords=[-45,-32,-19,-7,6,19,31,44,56,68])
fig.suptitle('(A) Temporal Lag Maps', fontsize=20, fontweight='bold')
plt.close()

fig8 = pn.Column(pn.pane.Matplotlib(fig),pn.pane.Markdown('# (B) Whole Brain      (C) Voxel-wise cross-correlations for representative locations indicated above'),pn.pane.HoloViews(fig8_panelsBC))

fig8.save('./figures/Fig08_RapidTide_results.{ss}.png'.format(ss=scan_selection))

text="![](./figures/Fig08_RapidTide_results.{ss}.png)".format(ss=scan_selection)
md("%s"%(text))
