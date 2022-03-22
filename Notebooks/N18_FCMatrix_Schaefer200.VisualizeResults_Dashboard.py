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
hm_data  = dict(start=[0,100],end=[100,200],start_event=[-4,-4],end_event=[-4,-4],hm_names=['LH','RH'])

Nrois = roi_info_df.shape[0]
print('++ INFO: Number of ROIs: %d' % Nrois)

# + [markdown] tags=[]
# ### Load individual FC matrices in memory
# -

# %%time
Z_matrix = {}
for suffix in ['Reference', 'BASIC', 'BASICpp', 'Behzadi_COMPCOR', 'Behzadi_COMPCORpp']:
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
for suffix in ['Reference', 'BASIC', 'BASICpp', 'Behzadi_COMPCOR', 'Behzadi_COMPCORpp']:
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
                      'BASICpp':   'mPP + Motion + Drifts + Bandpass + Lagged iFV',
                      'Behzadi_COMPCOR':'mPP + Motion + Drifts + Bandpass + 5PCAs from vents and WM',
                      'Behzadi_COMPCORpp':'mPP + Motion + Drifts + Bandpass + 5PCAs from vents and WM + Lagged iFV'}
select_preprocessing = pn.widgets.Select(name='Pre-processing', options=['Reference','BASIC', 'BASICpp','Behzadi_COMPCOR','Behzadi_COMPCORpp'], 
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

# + [markdown] tags=[]
# ***
#
# # Statistical Analyses for FC using NBS
#
# To find statistical differences in FC between scan types, we will rely on the NBS software (https://www.nitrc.org/projects/nbs). 
#
# Results from those analyses will then be plotted as 3D network graphs using BrainNetViewer (https://www.nitrc.org/projects/bnv/)
#
# NBS needs FC matrices in a very particular format (and organized in a particular way). In addition, it needs desigh matrices describing which matrices belong to each group. The following cells will generate all necessary files
#
# ### Generate Design Matrices for NBS 
#
# Here we generate 3 matrices, one per scan gropings. As mentioned above, we only report resutls for scan groups based on ET data. The other two resutls were for explorative purposes.
# -

# Create the design matrices
# ==========================
ET_DMatrix  = np.vstack([np.tile(np.array([0,1]),(len(ET_Groups['Awake']),1)),np.tile(np.array([1,0]),(len(ET_Groups['Drowsy']),1))])
np.savetxt(osp.join(Resources_Dir,'NBS_ET_DesingMatrix.txt'),ET_DMatrix,delimiter=' ',fmt='%d')
GS_DMatrix  = np.vstack([np.tile(np.array([0,1]),(len(GS_Groups['Awake']),1)),np.tile(np.array([1,0]),(len(GS_Groups['Drowsy']),1))])
np.savetxt(osp.join(Resources_Dir,'NBS_GS_DesingMatrix.txt'),GS_DMatrix,delimiter=' ',fmt='%d')
PSD_DMatrix = np.vstack([np.tile(np.array([0,1]),(len(PSD_Groups['Awake']),1)),np.tile(np.array([1,0]),(len(PSD_Groups['Drowsy']),1))])
np.savetxt(osp.join(Resources_Dir,'NBS_PSD_DesingMatrix.txt'),PSD_DMatrix,delimiter=' ',fmt='%d')

# ### Make copies of FC matrices on Resources folder that follow NBS requirements for data organization
#
# Here we will:
#
# * Generate individual folders for each pre-processing pipeline and group selection method --> We will have a total of 12 folders (4 pre-processing pipelines X 3 scan groupings)
#
# * Inside each folder, we will make copies of the connectivity matrices. Those copied will be named simply as subject???.txt --> This allows automatic loading of files in NBS

# %%time
# Create files with Z-scored connectivity matrices for their use in NBS
# =====================================================================
for suffix in ['BASIC', 'Behzadi_COMPCOR', 'AFNI_COMPCOR', 'AFNI_COMPCORp']:
    cc_matrix_xr = xr.DataArray(dims=['Run','ROI_x','ROI_y'], coords={'Run':Manuscript_Runs,'ROI_x':roi_info_df['ROI_Name'],'ROI_y':roi_info_df['ROI_Name']})
    # Load all the connectivity matrices for all subjects
    for i,item in enumerate(Manuscript_Runs):
        sbj,run  = item.split('_',1)
        path     = osp.join(DATA_DIR,sbj,run,'{run}_{suffix}.Shaeffer2018_200Parcels_000.netcc'.format(run=run,suffix=suffix))
        aux_cc_r = pd.read_csv(path,sep='\t',comment='#', header=1)
        # Convert R to Fisher Z-score as we will be computing the mean in the next cell
        aux_cc_Z = aux_cc_r.apply(np.arctanh)
        np.fill_diagonal(aux_cc_Z.values,1)
        cc_matrix_xr.loc[item,:,:] = aux_cc_Z
    # Save files to disk
    for sbj_lists,target_dir_prefix in zip([ET_Groups,    PSD_Groups,    GS_Groups],
                                           ['NBS_ET_Data','NBS_PSD_Data','NBS_GS_Data']):
        target_dir = osp.join(Resources_Dir,target_dir_prefix+'_'+suffix)
        if osp.exists(target_dir):
            rmtree(target_dir)
        os.mkdir(target_dir)
        print("++ INFO: Working on %s" % target_dir)
        for r,item in enumerate(sbj_lists['Awake']):
            sbj,run   = item.split('_',1)
            dest_path = osp.join(Resources_Dir,target_dir,'subject{id}.txt'.format(id=str(r+1).zfill(3)))
            np.savetxt(dest_path,cc_matrix_xr.loc[item,:,:],delimiter=' ',fmt='%f')
        for s,item in enumerate(sbj_lists['Drowsy']):
            sbj,run   = item.split('_',1)
            dest_path = osp.join(Resources_Dir,target_dir,'subject{id}.txt'.format(id=str(r+1+s+1).zfill(3)))    
            np.savetxt(dest_path,cc_matrix_xr.loc[item,:,:],delimiter=' ',fmt='%f')
        del r,s,item
    del cc_matrix_xr

# ### Run Statistical Analyses in MATLAB / NBS
#
# These analyses were conducted using NBS v1.2 on MATLAB 2019a. 
#
# To run the analysis, do the following:
#
# 1. Connect to a spersist node via NoMachine or VNC
#
# 2. Open a terminal and enter the project folder
#
# ```bash
# # cd /data/SFIMJGC_HCP7T/hcp7T_fv_sleep/
# ```
#
# 3. Load the matlab module and start matlab
#
# ```bash
# module load matlab/2019a
# matlab
# ```
#
# 4. Add NBS to the MATLAB path.
#
# ```matlab
# addpath(genpath('/data/SFIMJGC_HCP7T/hcp7t_fv_sleep_extraSW/NBS1.2/'))
# ```
#
# 5. Start NBS
#
# ```matlab
# NBS
# ```
#
# 6. Configure NBS for a particular scenario
#
#     See detailed instructions for each case on the following cells
#     
# 7. Save the results to disk using the File --> Save Current

# #### ET-based Groups, Behzadi_COMPCOR, Awake > Drowsy
#
# * Design Matrix: ```/data/SFIMJGC_HCP7T/hcp7t_fv_sleep/Resources/NBS_ET_DesingMatrix.txt```
# * Constrast: [-1,1]
# * Statistical Test: T-test
# * Threshold: 3.1
# * Connectivity Matrices: ```/data/SFIMJGC_HCP7T/hcp7t_fv_sleep/Resources/NBS_ET_Data_Behzadi/COMPCOR/subject0001.txt```
# * Node Coordinates: ```/data/SFIMJGC_HCP7T/hcp7t_fv_sleep/Resources/NBS_Node_Coordinates.txt```
# * Node LAbesl: ```/data/SFIMJGC_HCP7T/hcp7t_fv_sleep/Resources/NBS_Node_Labels.txt```
# * Exchange Blocks: EMPTY
# * Permutations: 5000
# * Significance: 0.05
# * Method: Network-Based Statistics
# * Component Size: Extent
#
# Once analyses are completed, please save as ```/data/SFIMJGC_HCP7T/hcp7t_fv_sleep/Resources/NBS_ET_Results/NBS_ET_Behzadi_COMPCOR_AgtD.mat```
# ![](./images/NBS_ET_Bezhadi_COMCOR_AgtD.configuration.png)

# #### ET-based Groups, Behzadi_COMPCOR, Drowsy > Awake
#
# * Design Matrix: ```/data/SFIMJGC_HCP7T/hcp7t_fv_sleep/Resources/NBS_ET_DesingMatrix.txt```
# * Constrast: [1,-1]
# * Statistical Test: T-test
# * Threshold: 3.1
# * Connectivity Matrices: ```/data/SFIMJGC_HCP7T/hcp7t_fv_sleep/Resources/NBS_ET_Data_Behzadi/COMPCOR/subject0001.txt```
# * Node Coordinates: LEAVE EMPTY TO AVOID MEMORY ISSUES DURING FINAL PLOTTING
# * Node LAbesl: ```/data/SFIMJGC_HCP7T/hcp7t_fv_sleep/Resources/NBS_Node_Labels.txt```
# * Exchange Blocks: EMPTY
# * Permutations: 5000
# * Significance: 0.05
# * Method: Network-Based Statistics
# * Component Size: Extent
#
# Once analyses are completed, please save as ```/data/SFIMJGC_HCP7T/hcp7t_fv_sleep/Resources/NBS_ET_Results/NBS_ET_Behzadi_COMPCOR_DgtA.mat```
# ![](./images/NBS_ET_Bezhadi_COMCOR_DgtA.configuration.png)

# Similarly, equivalent results can be generated for the other pre-processing scenarios, by selecteding the corresponding folder in the "Connectivity Matrices" field. 

# ***
# # Draw Statistically Significant Differences in Connectivity using BrainNetViewer
#
# ### Install BrainNetViewer
#
# For this step, we will use the BrainNetViewer (https://www.nitrc.org/projects/bnv/) software that runs on MATLAB. 
#
# 1. Connect to biowulf spersist node using NoMachine or VNC
#
# 2. Download the MATLAB version of BrainNetViewer (Version 1.7 Release 20191031) into /data/SFIMJGC_HCP7T/hcp7t_fv_sleep_extraSW/
#
# 3. Unzip the downloaded file in /data/SFIMJGC_HCP7T/hcp7t_fv_sleep_extraSW/BrainNetViewer
#
# ```bash
# # cd /data/SFIMJGC_HCP7T/hcp7t_fv_sleep_extraSW/
# # mkdir BrainNetViewer
# # mv ~/Downloads/BrainNetViewer_20191031.zip /data/SFIMJGC_HCP7T/hcp7t_fv_sleep_extraSW/BrainNetViewer
# # cd /data/SFIMJGC_HCP7T/hcp7t_fv_sleep_extraSW/BrainNetViewer
# unzip BrainNetViewer_20191031.zip
# # rm BrainNetViewer_20191031.zip
# ```
#
# 4. Start MATLAB
#
# ```bash
# # cd /data/SFIMJGC_HCP7T/hcp7t_fv_sleep
# module load matlab/2019a
# matlab
# ```
#
# 5. Add the path to BrainNetViewer in MATLAB's command window
#
# ```matlab
# addpath('/data/SFIMJGC_HCP7T/hcp7t_fv_sleep_extraSW/BrainNetViewer')
# ```
#
# 6. Start BrainNetViewer
#
# ```matlab
# BrainNet
# ```
#

# ### Convert NBS Ouputs to BrainNetViewer format
#
# We have generated a small MATLAB script that will take as input the saved results from a given NBS analysis and will write out an ```.edge``` file to be loaded into BrainNetViewer.
#
# This script will also print to the screen the number of signficantly different connections for each scenario. We use that information when composing the manuscript figure.
#
# 1. Start MATLAB
#
# ```bash
# # cd /data/SFIMJGC_HCP7T/hcp7t_fv_sleep
# module load matlab/2019a
# matlab
# ```
#
# 2. Enter the Notbook folder on the MATLAB console
#
# ```matlab
# # cd /data/SFIMJGC_HCP7T/hcp7t_fv_sleep/Notebooks
# ```
#
# 3. Run ```N12_FCMatrix_NBS2BrainViewer.m``` on the MATLAB console
#
# ```matlab
# N12_FCMatrix_NBS2BrainViewer.m
# ```

# ### Plot Results
#
# 1. Start MATLAB
#
# ```bash
# # cd /data/SFIMJGC_HCP7T/hcp7t_fv_sleep
# module load matlab/2019a
# matlab
# ```
#
# 2. Add BrainNetViewer to the path
#
# ```matlab
# addpath('/data/SFIMJGC_HCP7T/hcp7t_fv_sleep_extraSW/BrainNetViewer')
# ```
#
# 3. Start BrainNetViewer
#
# ```matlab
# BrainNet
# ```
#
# 4. Select "File --> Load File"
#
#     * Surface File: ```/data/SFIMJGC_HCP7T/hcp7t_fv_sleep_extraSW/BrainNetViewer/Data/SurfTemplate/BrainMesh_ICBM152.nv```
#     * Data File (nodes): ```/data/SFIMJGC_HCP7T/hcp7t_fv_sleep/Resources/BrainNet_Nodes.node```
#     * Data File (edges): ```/data/SFIMJGC_HCP7T/hcp7t_fv_sleep/Resources/NBS_ET_Results/NBS_ET_Behzadi_COMPCOR_AgtD.edge``` (This one will vary depending on which results you want to plot)
#     * Mapping: Leave Empty
#     
# 5. Press OK
#
# 6. On the BrainNet_option dialog that just opened, click Load and select ```/data/SFIMJGC_HCP7T/hcp7t_fv_sleep/Resources/NBS_ET_Results/BrainNet_Options_SchaeferColors.mat```
#
# 7. Click Apply
#
# This should result in a figure similar to this (depending on which data you are loading)
#
# ![](./images/NBS_SampleResult.png)
