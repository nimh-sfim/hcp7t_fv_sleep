#!/bin/bash
set -e
source ./common_variables.sh
module load afni

echo "++ INFO: Atlas Folder = ${ATLAS_DIR}"
echo "++ INFO: Atlas Name   = ${ATLAS_NAME}"
echo "++ INFO: Data Folder  = ${DATA_DIR}"
echo "++ INFO: Subject Name = ${SBJ}"

# Refernece grid for minamally pre-processed data
mPP_GRID=`echo ${DATA_DIR}/ALL/mPP_Grid.nii.gz`
echo "++ INFO: Reference fMRI Grid File = ${mPP_GRID}"

# Original Location of the 200 Schaefer Atlas (200 ROIs)
ATLAS_PATH=`echo ${ATLAS_DIR}/${ATLAS_NAME}/${ATLAS_NAME}_order_FSLMNI152_2mm.nii.gz`
echo "++ INFO: Atlas Template File = ${ATLAS_PATH}"

# (1) Enter This subject directory
# ================================
echo "++ INFO: (1) Enter This subject directory"
echo "========================================="
cd /data/SFIMJGC_HCP7T/HCP7T/${SBJ}/
echo " + INFO: Working Folder `pwd`"

# We can expect three files to have been downloaded from XNAT Central (which are in MNI, but in the grid of the T1)
#  * GM_Ribbon.nii.gz: includes GM ribbon and WM (but with sub-cortical as part of it). No CSF or ventricles
#  * brainmask_fs.nii.gz: follows the edge of the skull-stripped T1 --> WM,GM and CSF with brain stem all the way tot he bottom.
#  * aparc.a2009s+aseg.nii.gz: output of FS parcellation
#  Those three files are 0.7mm3 in a 260x311x260 grid.

# (2) Ensure these 3 files have their space correctly set to MNI
# ==============================================================
# NOTE: That way, any derivative from them will have the space set to MNI
echo "++ INFO: (2) Ensure these 3 files have their space correctly set to MNI"
echo "======================================================================="
for prefix in GM_Ribbon aparc.a2009s+aseg brainmask_fs
do
    echo "3drefit -space MNI ${prefix}.nii.gz"
    3drefit -space MNI ${prefix}.nii.gz
done

# (3) Create all necessary masks in original gird (i.e., anatomical grid and resolution)
# ======================================================================================
# GM, WM, 4th Vent (V4), lateral ventricles (Vl)
echo "++ INFO: (3) Create all necessary masks in original gird (i.e., anatomical grid and resolution)"
echo "==============================================================================================="
3dcopy -overwrite brainmask_fs.nii.gz                                                                                                                        ROI.FB.nii.gz
3dcalc -overwrite -a GM_Ribbon.nii.gz         -expr 'equals(a,42)  + equals(a,3)'                                                                    -prefix ROI.GM.nii.gz 
3dcalc -overwrite -a aparc.a2009s+aseg.nii.gz -expr 'equals(a,15)'                                                                                   -prefix ROI.V4.nii.gz 
3dcalc -overwrite -a aparc.a2009s+aseg.nii.gz -expr 'equals(a,43)  + equals(a,4)'                                                                    -prefix ROI.Vl.nii.gz 
3dcalc -overwrite -a aparc.a2009s+aseg.nii.gz -expr 'equals(a,2)+equals(a,41)+equals(a,251)+equals(a,252)+equals(a,253)+equals(a,254)+equals(a,255)' -prefix ROI.WM.nii.gz

# (4) Resameple original masks to EPI grid prior to eroding
# =========================================================
echo "++ INFO: (4) Resameple original masks to EPI grid prior to eroding"
echo "=================================================================="
# The masks provided by XNAT central sometimes contain a few voxels with flat timeseries on the borders. To avoid these voxels contributing to any calculations, 
# we will restrict all masks to only contain voxels that fall within the automask in all runs
# Compute automask per run
available_runs=`find ./ -name 'rfMRI_REST?_??_mPP.nii.gz'`
for RUN_PATH in ${available_runs}
do
    RUN=`echo ${RUN_PATH} | awk -F '/' '{print $2}'`
    3dAutomask -overwrite -prefix ./${RUN}/${RUN}_mPP.automask.nii.gz ${RUN_PATH}
done
available_automasks=`find ./ -name 'rfMRI_REST?_??_mPP.automask.nii.gz' | tr -s '\n' ' '`
echo "++ INFO: Available automasks: ${available_automasks}"
3dMean -overwrite -prefix ROI.automask.nii.gz ${available_automasks}
3dcalc -overwrite -prefix ROI.automask.nii.gz -expr 'equals(a,1)' -a ROI.automask.nii.gz

# Moving forward we will use ROI.automask.nii.gz to restrict all tissue masks
for prefix in ROI.GM ROI.V4 ROI.Vl ROI.WM ROI.FB
do
    3dresample -overwrite -rmode NN -inset ${prefix}.nii.gz -master ${mPP_GRID} -prefix ${prefix}.mPP.nii.gz
    3dcalc -overwrite -a ${prefix}.mPP.nii.gz -b ROI.automask.nii.gz -expr 'a*b' -prefix ${prefix}.mPP.nii.gz
done

# (5) Erode the masks in original anat space
# ==========================================
echo "++ INFO: (5) Erode the masks in original anat space"
echo "==================================================="
for prefix in ROI.GM ROI.V4 ROI.Vl ROI.WM ROI.FB
do
    3dmask_tool -overwrite -dilate_inputs -1 -inputs ${prefix}.nii.gz -prefix ${prefix}_e.nii.gz
done

# (6) Bring these eroded masks into the same grid as the fMRI data
# ================================================================
echo "++ INFO: (6) Bring these eroded masks into the same grid as the fMRI data"
echo "========================================================================="
for prefix in ROI.GM ROI.V4 ROI.Vl ROI.WM ROI.FB
do
    echo " +       3dresample -overwrite -rmode NN -inset ${prefix}_e.nii.gz -master ${mPP_GRID} -prefix ${prefix}_e.mPP.nii.gz"
    3dresample -overwrite                  \
           -rmode NN                       \
           -inset   ${prefix}_e.nii.gz     \
           -master  ${mPP_GRID}            \
           -prefix  ${prefix}_e.mPP.nii.gz
    3dcalc -overwrite -a ${prefix}_e.mPP.nii.gz -b ROI.automask.nii.gz -expr 'a*b' -prefix ${prefix}_e.mPP.nii.gz
    rm ${prefix}_e.nii.gz
done 

# (7) Create resampled version of the Schaefer atlas into the FIX grid
# =====================================================================
echo "++ INFO: (7) Create resampled version of the Schaefer atlas into the FIX grid"
echo "============================================================================="
echo " +       3dresample -overwrite -inset ${ATLAS_PATH} -rmode NN -master  ${mPP_GRID} -prefix  Schaefer2018_200Parcels.mPP.nii.gz"
3dresample -overwrite                \
           -inset ${ATLAS_PATH}      \
           -rmode NN                 \
           -master  ${mPP_GRID}      \
           -prefix  Schaefer2018_200Parcels.mPP.nii.gz

# (8) Set the table label and correct space in the subject specific atlas file
# ============================================================================
echo "++ INFO: (8) Set the table label and correct space in the subject specific atlas file"
echo "====================================================================================="
echo " +       3drefit -space MNI -labeltable ${ATLAS_DIR}/${ATLAS_NAME}/${ATLAS_NAME}_order.niml.lt Schaefer2018_200Parcels.mPP.nii.gz"
3drefit -space MNI \
        -labeltable ${ATLAS_DIR}/${ATLAS_NAME}/${ATLAS_NAME}_order.niml.lt \
        Schaefer2018_200Parcels.mPP.nii.gz
        
# (9) We then constrain the Atlas to the GM ribbon of each particular subject
# ===========================================================================
echo "++ INFO: (9) We then constrain the Atlas to the GM ribbon of each particular subject"
echo "===================================================================================="
echo " +       3dcalc     -overwrite -a Schaefer2018_200Parcels.mPP.nii.gz -b ROI.GM.mPP.nii.gz -expr 'a*b' -prefix Schaefer2018_200Parcels.mPP.nii.gz"
3dcalc     -overwrite                                   \
           -a Schaefer2018_200Parcels.mPP.nii.gz \
           -b ROI.GM.mPP.nii.gz                         \
           -expr 'a*b'                                  \
           -prefix Schaefer2018_200Parcels.mPP.nii.gz 

echo "=================================="
echo "++ INFO: Script finished correctly"
echo "=================================="
