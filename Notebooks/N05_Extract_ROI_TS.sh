#!/bin/bash

# Author: Javier Gonzalez-Castillo 
# Date: August 2021
#
set -e
module load afni
source ./common_variables.sh

# (0) Prepare environment
# =======================
echo "++ INFO: (0) Prepare the computing environment"
echo "=============================================="
# Group-level Forth Ventricle ROI
ROI_FV_PATH=`echo ${DATA_DIR}/ALL/ALL_ROI.V4.mPP.nii.gz`
echo " + INFO: Group-level FV ROI: ${ROI_FV_PATH}"

# Enter scripts directory
# =======================
echo " + INFO: Entering Notebooks directory..."
cd ${SCRIPTS_DIR}/Notebooks/

# Unset DISPLAY variable so that we don't get an error about access to XDisplay 
# when submitting batch jobs in the cluster
# =============================================================================
echo " + INFO: Unsetting the DISPLAY environment variable"
unset DISPLAY

# Enter Run Folder
# ================
cd ${DATA_DIR}/${SBJ}/${RUN}/
WORKDIR=`pwd`
echo "++ INFO: Working Directory: ${WORKDIR}"

# (1) Create Signal Percent Change version
# ------------------------------------
echo "++ INFO: (1) Create SPC version of the minimally pre-processed data..."
echo " + ==================================================================="
3dTstat -overwrite -mean -mask ../ROI.FB.mPP.nii.gz -prefix ${RUN}_mPP.MEAN.nii.gz ${RUN}_mPP.nii.gz[${VOLS_DISCARD}..$]
3dcalc -overwrite \
       -a ${RUN}_mPP.nii.gz[${VOLS_DISCARD}..$] \
       -b ${RUN}_mPP.MEAN.nii.gz \
       -c ../ROI.FB.mPP.nii.gz \
       -expr  'c * min(200, a/b*100)*step(a)*step(b)' \
       -prefix ${RUN}_mPP.scale.nii.gz
nifti_tool -strip_extras -overwrite -infiles ${RUN}_mPP.scale.nii.gz

# (2) Extract Representative Timseries from the minimally pre-processed data in SPC units
# =======================================================================================
echo "++ INFO: (2) Extract Representative Timseries from the minimally pre-processed data in SPC units"
echo "================================================================================================"
for items in ${ROI_FV_PATH},'V4_grp' ../ROI.V4_e.mPP.nii.gz,'V4_e' ../ROI.Vl_e.mPP.nii.gz,'Vl_e' ../ROI.GM.mPP.nii.gz,'GM' ../ROI.FB.mPP.nii.gz,'FB' ../ROI.WM_e.mPP.nii.gz,'WM_e'
do 
   IFS="," ; set -- $items
   echo " + INFO: Mask = $1 | Suffix = $2 --> Output = ${RUN}_mPP.Signal.$2.1D"
   3dmaskave -quiet -mask $1 ${RUN}_mPP.scale.nii.gz   > rm.aux.$2.1D
   3dDetrend -prefix - -polort ${POLORT} rm.aux.$2.1D\'  > rm.det.aux.$2.1D
   cat rm.det.aux.$2.1D | tr -s ' ' '\n' | sed '/^$/d' > ${RUN}_mPP.Signal.$2.1D
   rm rm.aux.$2.1D rm.det.aux.$2.1D
done

echo "=================================="
echo "++ INFO: Script finished correctly"
echo "=================================="
