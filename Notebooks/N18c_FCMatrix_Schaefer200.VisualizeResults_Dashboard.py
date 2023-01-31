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

# + [markdown] tags=[]
# # Description - Differences in FC across scan types
#
# This notebook performs the following operations:
#
# * Create Swarm jobs to generate FC matrices for all runs via AFNI 3dNetCor
# * Check all FC matrices were generated correctly
# * Generates average connectivity matrices per scan type for all different pipelines
# * Generates histograms of FC values across the brain for the different conditions
# * Starts a dashboard where we can explore those FC matrices
# * Prepares files for subsequent statistical analyses using NBS software in MATLAB
# -

# # Import Libraries

# +
import os
import os.path as osp
import numpy as np
import xarray as xr
import pandas as pd
import panel as pn
import holoviews as hv
import hvplot.pandas
import matplotlib.pyplot as plt
from utils.variables import SCRIPTS_DIR, Resources_Dir, DATA_DIR
from utils.basics import get_available_runs

from shutil import rmtree
from random import sample
hv.extension('bokeh')
# -

# # Gather Port Information for launching the dashboard

port_tunnel = int(os.environ['PORT2'])
print('++ INFO: Second Port available: %d' % port_tunnel)

# # Load Run Lists (all, drowsy, awake)

# +
# %%time
Manuscript_Runs,Awake_Runs,Drowsy_Runs = {},{},{}
scan_HR_info             = pd.read_csv(osp.join(Resources_Dir,'HR_scaninfo.csv'), index_col=0)
scan_HR_info             = scan_HR_info[(scan_HR_info['HR_aliased']< 0.03) | (scan_HR_info['HR_aliased']> 0.07)]
Manuscript_Runs['noHRa'] = list(scan_HR_info.index)
Awake_Runs['noHRa']      = list(scan_HR_info[scan_HR_info['Scan Type']=='Awake'].index)
Drowsy_Runs['noHRa']     = list(scan_HR_info[scan_HR_info['Scan Type']=='Drowsy'].index)

Manuscript_Runs['all'] = get_available_runs(when='final', type='all')
Awake_Runs['all']      = get_available_runs(when='final', type='awake')
Drowsy_Runs['all']     = get_available_runs(when='final', type='drowsy')
print('++ INFO [All]:   Number of Runs: Total = %d | Awake = %d | Drowsy = %d' % (len(Manuscript_Runs['all']), len(Awake_Runs['all']), len(Drowsy_Runs['all'])))
print('++ INFO [noHRa]: Number of Runs: Total = %d | Awake = %d | Drowsy = %d' % (len(Manuscript_Runs['noHRa']), len(Awake_Runs['noHRa']), len(Drowsy_Runs['noHRa'])))
# -

# ***
# # Load Run Group Information
#
# In the manuscript we will only report results for run groupings based on Eye Tracker data. For exploration purposes, we also generated FC matrices when scans have been separated according to their ranking based on PSDsleep or GS. Those alternative groupings were generated in Notebook N11

# +
ET_Groups  = {'noHRa':{'Awake':Awake_Runs['noHRa'],'Drowsy':Drowsy_Runs['noHRa']},
              'all':  {'Awake':Awake_Runs['all'],'Drowsy':Drowsy_Runs['all']}}
              
GS_Groups  = {'noHRa':{'Drowsy':np.loadtxt(osp.join(Resources_Dir,'Run_List_GS_Top100.V4lt_grp.noHRa.txt'),dtype=str),
                       'Awake':np.loadtxt(osp.join(Resources_Dir,'Run_List_GS_Bot100.V4lt_grp.noHRa.txt'),dtype=str)},
              'all':  {'Drowsy':np.loadtxt(osp.join(Resources_Dir,'Run_List_GS_Top100.V4lt_grp.all.txt'),dtype=str),
                       'Awake':np.loadtxt(osp.join(Resources_Dir,'Run_List_GS_Bot100.V4lt_grp.all.txt'),dtype=str)}}
              
PSD_Groups = {'noHRa':{'Drowsy':np.loadtxt(osp.join(Resources_Dir,'Run_List_PSD_Top100.V4lt_grp.noHRa.txt'),dtype=str),
                       'Awake':np.loadtxt(osp.join(Resources_Dir,'Run_List_PSD_Bot100.V4lt_grp.noHRa.txt'),dtype=str)},
              'all':  {'Drowsy':np.loadtxt(osp.join(Resources_Dir,'Run_List_PSD_Top100.V4lt_grp.all.txt'),dtype=str),
                       'Awake':np.loadtxt(osp.join(Resources_Dir,'Run_List_PSD_Bot100.V4lt_grp.all.txt'),dtype=str)}}

#PSD_Groups = {'Drowsy': [s for s in PSD_Groups['Drowsy'] if s in ET_Groups['Drowsy']],
#              'Awake': [s for s in PSD_Groups['Awake'] if s in ET_Groups['Awake']]}

print('++ [All] ET-Based Grouping     : Awake = %d, Drowsy = %d' %(len(ET_Groups['all']['Awake']),len(ET_Groups['all']['Drowsy'])))
print('++ [noHRa] ET-Based Grouping   : Awake = %d, Drowsy = %d' %(len(ET_Groups['noHRa']['Awake']),len(ET_Groups['noHRa']['Drowsy'])))
print('++')
print('++ [All] GS-Based Grouping     : Awake = %d, Drowsy = %d' %(len(GS_Groups['all']['Awake']),len(GS_Groups['all']['Drowsy'])))
print('++ [noHRa] GS-Based Grouping   : Awake = %d, Drowsy = %d' %(len(GS_Groups['noHRa']['Awake']),len(GS_Groups['noHRa']['Drowsy'])))
print('++')
print('++ [All] PSD-Based Grouping    : Awake = %d, Drowsy = %d' %(len(PSD_Groups['all']['Awake']),len(PSD_Groups['all']['Drowsy'])))
print('++ [noHRa] PSD-Based Grouping  : Awake = %d, Drowsy = %d' %(len(PSD_Groups['noHRa']['Awake']),len(PSD_Groups['noHRa']['Drowsy'])))

# + [markdown] tags=[]
# # Generate FC Matrices
#
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

# Get Network Names
net_names = roi_info_df['Network'].unique()
print('++ INFO: Network Names %s' % str(net_names))
# Get Positions of start and end of ROIs that belong to the different networks. We will use this info to set labels
# on matrix axes
net_ends = [0]
for network in net_names:
    net_ends.append(roi_info_df[(roi_info_df['Network']==network) & (roi_info_df['Network_Hemi']=='LH')].iloc[-1]['ROI_ID'])
for network in net_names:
    net_ends.append(roi_info_df[(roi_info_df['Network']==network) & (roi_info_df['Network_Hemi']=='RH')].iloc[-1]['ROI_ID'])
print('++ INFO: Network End      IDs: %s ' % str(net_ends))
net_meds = [int(i) for i in net_ends[0:-1] + np.diff(net_ends)/2]
print('++ INFO: Network Midpoint IDs: %s ' % str(net_meds))

network_labels = roi_info_df['Network'][net_meds]
network_labels_tuples = list(zip(network_labels.index,network_labels)) + [(int(i),' ') for i in net_ends]

nw_color_map = {'LH-Vis':'purple',      'RH-Vis':'purple',
                   'LH-SomMot':'lightblue'  ,'RH-SomMot':'lightblue',
                   'LH-DorsAttn':'green'    ,'RH-DorsAttn':'green',
                   'LH-SalVentAttn':'violet','RH-SalVentAttn':'violet',
                   'LH-Limbic':'yellow','RH-Limbic':'yellow',
                   'LH-Cont':'orange','RH-Cont':'orange',
                   'LH-Default':'red','RH-Default':'red'}
hm_color_map = {'LH':'grey','RH':'darkgrey'}

# Structures to plot colored segments on the X and Y axis accompanying each matrix
aux       = ['-'.join([item['Network_Hemi'],item['Network']]) for i,item in roi_info_df.iterrows()]
net_names = []
[net_names.append(x) for x in aux if x not in net_names];
net_data  = dict(start=net_ends[:-1],
                end=net_ends[1:],
                start_event=-4*np.ones(14),
                end_event=-4*np.ones(14),
                net_names=net_names)
hm_data  = dict(start=[0,97],end=[100,195],start_event=[-4,-4],end_event=[-4,-4],hm_names=['LH','RH'])

Nrois = roi_info_df.shape[0]
print('++ INFO: Number of ROIs: %d' % Nrois)

# + [markdown] tags=[]
# ### Load individual FC matrices in memory
# -

# %%time
Z_matrix = {}
for suffix in ['Reference', 'GSR', 'BASIC', 'BASICpp', 'Behzadi_COMPCOR', 'Behzadi_COMPCORpp']:
    cc_matrix_xr = xr.DataArray(dims=['Run','ROI_x','ROI_y'], coords={'Run':Manuscript_Runs['all'],'ROI_x':roi_info_df['ROI_Name'],'ROI_y':roi_info_df['ROI_Name']})
    # Load all the connectivity matrices for all subjects
    for i,item in enumerate(Manuscript_Runs['all']):
        sbj,run  = item.split('_',1)
        path     = osp.join(DATA_DIR,sbj,run,'{run}_{suffix}.Shaeffer2018_200Parcels_000.netcc'.format(run=run,suffix=suffix))
        aux_cc_r = pd.read_csv(path,sep='\t',comment='#', header=1)
        # Convert R to Fisher Z-score as we will be computing the mean in the next cell
        aux_cc_Z = aux_cc_r.apply(np.arctanh)
        np.fill_diagonal(aux_cc_Z.values,1)
        cc_matrix_xr.loc[item,:,:] = aux_cc_Z
    Z_matrix[suffix] = cc_matrix_xr
    del cc_matrix_xr

# + [markdown] tags=[]
# ### Compute average FC matrix per group
# -

# %%time
Mean_CCs     = {}
for suffix in ['Reference', 'GSR', 'BASIC', 'BASICpp', 'Behzadi_COMPCOR', 'Behzadi_COMPCORpp']:
    for group_criteria,groups in zip(['ET','PSD','GS'],[ET_Groups,PSD_Groups,GS_Groups]):
        for scan_selection in ['all','noHRa']:
            # Compute average FC matrix per run type
            Mean_CCs[(scan_selection,suffix,group_criteria,'Awake', 'Z')]  = pd.DataFrame(Z_matrix[suffix].loc[groups[scan_selection]['Awake'] ,:,:].mean(axis=0).values)
            Mean_CCs[(scan_selection,suffix,group_criteria,'Drowsy','Z')]  = pd.DataFrame(Z_matrix[suffix].loc[groups[scan_selection]['Drowsy'],:,:].mean(axis=0).values)
            Mean_CCs[(scan_selection,suffix,group_criteria,'Awake','R')]   = Mean_CCs[(scan_selection,suffix,group_criteria,'Awake','Z')].apply(np.tanh)
            Mean_CCs[(scan_selection,suffix,group_criteria,'Drowsy','R')]  = Mean_CCs[(scan_selection,suffix,group_criteria,'Drowsy','Z')].apply(np.tanh)
            Mean_CCs[(scan_selection,suffix,group_criteria,'Awake-Drowsy','Z')]             = Mean_CCs[(scan_selection,suffix,group_criteria,'Awake','Z')] - Mean_CCs[(scan_selection,suffix,group_criteria,'Drowsy','Z')]
            Mean_CCs[(scan_selection,suffix,group_criteria,'Awake-Drowsy','R')]             = Mean_CCs[(scan_selection,suffix,group_criteria,'Awake-Drowsy','Z')].apply(np.tanh)
            Mean_CCs[(scan_selection,suffix,group_criteria,'100*(Awake-Drowsy)/Awake','R')] = 100 * Mean_CCs[(scan_selection,suffix,group_criteria,'Awake-Drowsy','R')] / Mean_CCs[(scan_selection,suffix,group_criteria,'Awake','Z')].apply(np.tanh)

# + [markdown] tags=[]
# ### Generate Plotting Infrastructure

# +
preprocessing_labels={'Reference': 'mPP + Drifts + Bandpass',
                      'BASIC':     'mPP + Motion + Drifts + Bandpass',
                      'GSR':       ' BASIC + Global Signal Regression',
                      'BASICpp':   'mPP + Motion + Drifts + Bandpass + Lagged iFV',
                      'Behzadi_COMPCOR':'mPP + Motion + Drifts + Bandpass + 5PCAs from vents and WM',
                      'Behzadi_COMPCORpp':'mPP + Motion + Drifts + Bandpass + 5PCAs from vents and WM + Lagged iFV'}
select_preprocessing = pn.widgets.Select(name='Pre-processing', options=['Reference','GSR','BASIC', 'BASICpp','Behzadi_COMPCOR','Behzadi_COMPCORpp'], 
                                         value='Reference')

select_grouping      = pn.widgets.Select(name='Group Creation', options=['ET','GS','PSD'], value='ET')
select_scans         = pn.widgets.Select(name='Scan Selection', options=['all','noHRa'], value='all')


# -

@pn.depends(select_preprocessing)
def inform_preprocessing(suffix):
    return pn.pane.Markdown('###'+preprocessing_labels[suffix],width=1000)


@pn.depends(select_preprocessing, select_grouping,select_scans)
def plot_drowsyFC(suffix,grouping,scan_selection):
    drowsy_matrix = hv.Image(Mean_CCs[(scan_selection,suffix,grouping,'Drowsy','R')].values[::-1], bounds=(0,0,Nrois,Nrois)).opts(clim=(-0.75,0.75), cmap='RdBu_r', aspect='square', frame_width=300, 
                                                                                              title='', fontsize={'ticks':12, 'title':16}, 
                                                                                              xlabel='', ylabel='',
                                                                                              colorbar=True, xaxis=None, yaxis=None).opts(toolbar=None)
    net_segments_y = hv.Segments(net_data, 
                             [hv.Dimension('start_event',range=(-8,Nrois)), 
                              hv.Dimension('start',range=(-8,Nrois)),'end_event', 'end']
                             ,'net_names').opts(color='net_names',line_width=10, cmap=nw_color_map,show_legend=False)
    hm_segments_x = hv.Segments(hm_data, 
                             [hv.Dimension('start',range=(-8,Nrois)), 
                              hv.Dimension('start_event',range=(-8,Nrois)),'end', 'end_event']
                             ,'hm_names').opts(color='hm_names',line_width=10, cmap=hm_color_map,show_legend=False)
    return net_segments_y * hm_segments_x * drowsy_matrix


@pn.depends(select_preprocessing, select_grouping, select_scans)
def plot_awakeFC(suffix,grouping,scan_selection):
    awake_matrix = hv.Image(Mean_CCs[(scan_selection,suffix,grouping,'Awake','R')].values[::-1], bounds=(0,0,Nrois,Nrois)).opts(clim=(-0.75,0.75), cmap='RdBu_r', aspect='square', frame_width=300, 
                                                                                              title='', fontsize={'ticks':12, 'title':16}, 
                                                                                              xlabel='', ylabel='',
                                                                                              colorbar=True, xaxis=None, yaxis=None).opts(toolbar=None)
    net_segments_y = hv.Segments(net_data, 
                             [hv.Dimension('start_event',range=(-8,Nrois)), 
                              hv.Dimension('start',range=(-8,Nrois)),'end_event', 'end']
                             ,'net_names').opts(color='net_names',line_width=10, cmap=nw_color_map,show_legend=False)
    hm_segments_x = hv.Segments(hm_data, 
                             [hv.Dimension('start',range=(-8,Nrois)), 
                              hv.Dimension('start_event',range=(-8,Nrois)),'end', 'end_event']
                             ,'hm_names').opts(color='hm_names',line_width=10, cmap=hm_color_map,show_legend=False)
    
    return net_segments_y * hm_segments_x * awake_matrix


@pn.depends(select_preprocessing, select_grouping, select_scans)
def plot_diffFC(suffix,grouping,scan_selection):
    diff_matrix   = hv.Image(Mean_CCs[(scan_selection,suffix,grouping,'Awake-Drowsy','R')].values[::-1], bounds=(0,0,Nrois,Nrois)).opts(clim=(-0.20,0.20), cmap='PiYG_r', aspect='square', frame_width=300, 
                                                                                              title='Awake - Drowsy', fontsize={'ticks':12, 'title':16}, 
                                                                                              xlabel='', ylabel='',
                                                                                              yticks=network_labels_tuples, xticks=[(50,'Left'),(100,'|'),(150,'Right')], colorbar=True).opts(toolbar=None)
    return diff_matrix


@pn.depends(select_preprocessing, select_grouping, select_scans)
def plot_staticFC_scan_level(suffix,grouping,scan_selection):
    awake_matrix = hv.Image(Mean_CCs[(scan_selection,suffix,grouping,'Awake','R')].values[::-1], bounds=(0,0,200,200)).opts(clim=(-0.75,0.75), cmap='RdBu_r', aspect='square', frame_width=300, 
                                                                                              title='Average (Awake)', fontsize={'ticks':12, 'title':18}, 
                                                                                              yticks=network_labels_tuples, xticks=[(50,'Left'),(100,'|'),(150,'Right')], colorbar=True)
     
    drowsy_matrix = hv.Image(Mean_CCs[(scan_selection,suffix,grouping,'Drowsy','R')].values[::-1], bounds=(0,0,200,200)).opts(clim=(-0.75,0.75), cmap='RdBu_r', aspect='square', frame_width=300, 
                                                                                              title='Average (Drowsy)', fontsize={'ticks':12, 'title':18}, 
                                                                                              yticks=network_labels_tuples, xticks=[(50,'Left'),(100,'|'),(150,'Right')], colorbar=True)
     
    diff_matrix   = hv.Image(Mean_CCs[(scan_selection,suffix,grouping,'Awake-Drowsy','R')].values[::-1], bounds=(0,0,200,200)).opts(clim=(-0.20,0.20), cmap='PiYG_r', aspect='square', frame_width=300, 
                                                                                              title='Awake - Drowsy', fontsize={'ticks':12, 'title':18}, 
                                                                                              yticks=network_labels_tuples, xticks=[(50,'Left'),(100,'|'),(150,'Right')], colorbar=True)
    
    diff_pc_matrix = hv.Image(Mean_CCs[(scan_selection,suffix,grouping,'100*(Awake-Drowsy)/Awake','R')].values[::-1], bounds=(0,0,200,200)).opts(clim=(-200,200), cmap='PiYG_r', aspect='square', frame_width=300, 
                                                                                              title='Awake - Drowsy (%)', fontsize={'ticks':12, 'title':18}, 
                                                                                              yticks=network_labels_tuples, xticks=[(50,'Left'),(100,'|'),(150,'Right')], colorbar=True)
    #layout = pn.pane.HTML(suffix+'_'+grouping)
    layout = (awake_matrix + drowsy_matrix + diff_matrix + diff_pc_matrix).cols(4)
    return layout


@pn.depends(select_preprocessing, select_grouping, select_scans)
def plot_histograms(suffix,grouping,scan_selection):
    FC_Drowsy = Mean_CCs[(scan_selection,suffix,grouping,'Drowsy','R')].values
    FC_Drowsy = pd.DataFrame(FC_Drowsy[np.triu_indices(Nrois,1)], columns=['Drowsy'])
    FC_Awake  = Mean_CCs[(scan_selection,suffix,grouping,'Awake','R')].values
    FC_Awake  = pd.DataFrame(FC_Awake[np.triu_indices(Nrois,1)], columns=['Awake'])
    h = pd.concat([FC_Drowsy,FC_Awake],axis=1).hvplot.kde(legend=False,
                                                          width=300, height=350, 
                                                          xlim=(-.4,1), ylim=(0,3.5), xticks=[(-0.4,'-0.4'),(0,'0'),(0.5,'0.5'),(1.0,'1')],
                                                          xlabel='R', ylabel='Density', 
                                                          title='',
                                                          fontsize={'ticks':12, 'labels':12,'title':16},
                                                          color=['lightblue','orange']).opts(toolbar=None)
    return h


# ### Start Dashboard

data_dashboard = pn.Column(pn.Row(select_preprocessing,select_grouping,select_scans,inform_preprocessing),
                           pn.Row(plot_awakeFC,plot_drowsyFC,plot_histograms,plot_diffFC))

data_dashboard_server = data_dashboard.show(port=port_tunnel,open=False)

data_dashboard_server.stop()

# ***
#
# ## Supplementary Figure for the GSR pipeline
#
# During the last round of reviews we were asked to add results for the FC analyses using also global signal regression. These analyses are reported as an additional supplementary figure.
#
# The top panels of the figure are screenshots from the data_dashboard initiated above, which now will also include the option to show results for the GSR scenario.
#
# The code below generated the lower panel of such supplementary figure.

# Load and store in memory the outputs from NBS for all different preprocessing scenarios
Diff_matrix={} # This dictionary will hold the combiantion of both contrasts (AgtD as 1s and DgtA as -1s)
AgtD_matrix={} # This dictionary will hold the AgtD results
DgtA_matrix={} # This dictionary will hold the DgtA results
for pipeline,pipeline_label in zip(['Reference', 'GSR', 'BASIC', 'BASICpp', 'COMPCOR', 'COMPCORpp'],
                                   ['Smoothing', 'GSR', 'Basic', 'Basic+', 'CompCor', 'CompCor+']):
    AgtD_path = osp.join(Resources_Dir+'_NBS','NBS_ET_Results','NBS_ET_{pp}_AgtD.edge'.format(pp=pipeline))
    DgtA_path = osp.join(Resources_Dir+'_NBS','NBS_ET_Results','NBS_ET_{pp}_DgtA.edge'.format(pp=pipeline))
    AgtD_matrix[pipeline_label] = np.loadtxt(AgtD_path)[::-1]
    DgtA_matrix[pipeline_label] = np.loadtxt(DgtA_path)[::-1]
    Diff_matrix[pipeline_label] = AgtD_matrix[pipeline_label] - DgtA_matrix[pipeline_label]

layout = None
for pp in ['Smoothing','Basic', 'CompCor', 'CompCor+','GSR']:
    mat = hv.Image(Diff_matrix[pp], bounds=(0,0,Nrois,Nrois)).opts(clim=(-2,2), cmap=['lightblue','white','orange'], aspect='square', frame_width=400, 
                                                                                              title=pp, fontsize={'ticks':12, 'title':16}, 
                                                                                              xlabel='', ylabel='',
                                                                                              colorbar=False, xaxis=None, yaxis=None).opts(toolbar=None)
    net_segments_y = hv.Segments(net_data, 
                             [hv.Dimension('start_event',range=(-8,Nrois)), 
                              hv.Dimension('start',range=(-8,Nrois)),'end_event', 'end']
                             ,'net_names').opts(color='net_names',line_width=10, cmap=nw_color_map,show_legend=False)
    hm_segments_x = hv.Segments(hm_data, 
                             [hv.Dimension('start',range=(-8,Nrois)), 
                              hv.Dimension('start_event',range=(-8,Nrois)),'end', 'end_event']
                             ,'hm_names').opts(color='hm_names',line_width=10, cmap=hm_color_map,show_legend=False)
    if layout is None:
        layout = mat * net_segments_y * hm_segments_x
    else:
        layout = layout + (mat * net_segments_y * hm_segments_x)

layout.cols(5)

# Given the number of connections that show significant differences in the GSR scenario is too big, it is not easy to see specific patterns of change being discussed. For that reason we have to additional panels in the
# supplementary figure that show connections stronger for Awake than Drowsy within the DMN and whithin the different attention networks. The next two cells create files with this information that can subsequently be loaded
# into BrainNetView to generate those two sub-panels.

M = pd.DataFrame(AgtD_matrix['GSR']).copy()
M.index = roi_info_df['Node_ID']
M.columns = roi_info_df['Node_ID']
for i in M.index:
    for j in M.columns:
        if not('Default' in i) or not('Default' in j):
            M.loc[i,j] = 0
M.to_csv('/data/SFIMJGC_HCP7T/hcp7t_fv_sleep/Resources_NBS/NBS_ET_Results/GSR_DMN_Map.edge',sep=' ', header=None,index=None)

M = pd.DataFrame(AgtD_matrix['GSR']).copy()
M.index = roi_info_df['Node_ID']
M.columns = roi_info_df['Node_ID']
for i in M.index:
    for j in M.columns:
        if not('Attn' in i) or not('Attn' in j):
            M.loc[i,j] = 0
M.to_csv('/data/SFIMJGC_HCP7T/hcp7t_fv_sleep/Resources_NBS/NBS_ET_Results/GSR_Attn_Map.edge',sep=' ', header=None,index=None)

# ***
# ***
# ### END OF NOTEBOOK
# ***
# ***
