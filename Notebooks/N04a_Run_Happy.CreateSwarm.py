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

# #### Load necessary libraries

import os.path as osp
import os
import numpy as np
import pandas as pd
import subprocess
import hvplot.pandas
import holoviews as hv
from utils.variables import Resources_Dir, DATA_DIR
from utils.basics import get_available_runs, alised_freq

# ***
#
# ## 1. Description of the potential problem (Figure 3)

fmri_fs = 1        # Sampling Frequency of the fMRI recordings [Hz]
card_min_f = 50/60 # Common lower cardiac frequency
card_max_f = 80/60 # Common higher cardiac frequency

n = 500
hrs_df = pd.DataFrame(columns=['Heart Rate','Aliased Heart Rate'])
for hr in np.linspace(start=card_min_f,stop=card_max_f,num=n):
    hrs_df = hrs_df.append({'Heart Rate':hr, 'Aliased Heart Rate':alised_freq(fmri_fs,hr)}, ignore_index=True)

hv.VLine(0.03).opts(line_color='k',line_dash='dashed', xlim=(-.1,2)) * \
hv.VLine(0.07).opts(line_color='k',line_dash='dashed') * \
hv.VLine(1).opts(line_color='k') * \
hrs_df.hvplot.hist(bins=100, alpha=0.5, normed=True).opts(legend_position='top_right') * \
hrs_df.hvplot.kde(alpha=.3).opts(xlabel='Frequency [Hz]', ylabel='Density', toolbar=None, fontsize={'xticks':18, 'yticks':18, 'xlabel':18, 'ylabel':18, 'legend':18}) 

# > Figure 3. Simulation of frequency aliasing for cardiac pulsations. The sampling frequency of the fMRI data is 1Hz (black continuous line). Our target fluctuations of interest sit in the vicinity of 0.05Hz, and we will attempt their detection by focusing our attention on the frequency range [0.03Hz - 0.05 Hz] (narrow band between the two vertical balck dashed lines). Typical cardiac rates range from 50 to 80 beats per minute while subjects are resting (blue histogram/distribution). Due to frequency aliasing, cardiac pulsations at those frequencies will appear at lower parts of the spectrum in the fMRI recordings. As the figure shows, given an Fs=1 Hz there is potential for those to overlap (red histogram/distribution) with the target frequency of our study. 

# ***
# ## 2. Extract Cardiac Traces from fMRI data using the "Happy" Package

# #### Load List of Scans to process

scan_list = get_available_runs(when='final',type='all')
print('++ INFO: Number of scans to process: %d' % len(scan_list))

# #### Create Log Dir for swarm jobs

if not osp.exists('./N04a_Run_Happy.logs'):
    print('++ INFO: Creating logging dir: N04a_Run_Happy.logs')
    os.mkdir('./N04a_Run_Happy.logs')

# #### Create Swarm File

os.system('echo "#swarm -f ./N04a_Run_Happy..SWARM.sh -g 128 -t 32  --partition quick,norm --logdir ./N04a_Run_Happy.logs" > ./N04a_Run_Happy.SWARM.sh')
for item in scan_list:
    sbj,run = item.split('_',1)
    os.system('echo "export SBJ={sbj} RUN={run}; sh ./N04a_Run_Happy.CreateSwarm.sh" >> ./N04a_Run_Happy.SWARM.sh'.format(sbj=sbj,run=run))

# #### Check all outputs were generated

for scanID in scan_list:
    sbj,run = scanID.split('_',1)
    output_path = osp.join(DATA_DIR,sbj,run,'{run}_orig.happy'.format(run=run),'{run}_orig.happy_desc-stdrescardfromfmri_timeseries.tsv'.format(run=run))
    if not osp.exists(output_path):
        print('++ WARNING: %s is missing' % output_path)

# ***
# ***
# # END OF NOTEBOOK
# ***
# ***


