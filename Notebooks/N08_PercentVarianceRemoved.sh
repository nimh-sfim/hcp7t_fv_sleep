#!/bin/bash
# Author: Javier Gonzalez-Castillo
# Data: 3th March, 2022
# =========================================================================

set -e
source ./common_variables.sh
module load afni 
OMP_NUM_THREADS=16

# (0) Enter working environment
# =============================
cd ${DATA_DIR}/${SBJ}/${RUN}/
WDIR=`pwd`
echo "++ INFO: Working Directory [${WDIR}]"

# (1) Obtain basic information about the data (e.g., TR, Num Volumnes, etc)
# =========================================================================
nt=`3dinfo -nt ${RUN}_mPP.scale.nii.gz`
tr=`3dinfo -tr ${RUN}_mPP.nii.gz`
echo " +   NT = ${nt} acquisitions"
echo " +   TR = ${tr} seconds"
echo " +   Discard Volumes relative to mPP = ${VOLS_DISCARD} acquisitions"
echo " +   POLORT = ${POLORT} polynomials"
echo " +   Bluring FWHM = ${BLUR_FWHM} mm"

## for scenario in BASIC BASICpp COMPCOR COMPCORpp
## do
##   3dcalc -overwrite \
##          -a ${RUN}_Reference.VAR.nii.gz \
##          -b ${RUN}_${scenario}.VAR.nii.gz \
##          -m ${DATA_DIR}/ALL/ALL_EPI_FBmask.nii.gz \
##          -expr 'm*100*(a-b)/a' \
##          -prefix ${RUN}_PVR.${scenario}.nii.gz
## done

## # March 11, 2022: This part requires the final atlas to exists. We may want to move this script to after the connectivity analysis
## # For now we leave it here, so we can move forward with the review
## 
## for pipeline in BASIC COMPCOR
## do
##     3dcalc -overwrite -a ${RUN}_PVR.${pipeline}pp.nii.gz -b ${RUN}_PVR.${pipeline}.nii.gz -expr 'a-b' -prefix ${RUN}_PVR.${pipeline}pp_minus_${pipeline}.nii.gz
##     3dROIstats -quiet \
##                -mask ../../ALL/Schaefer2018_200Parcels_7Networks_order_mPP.GM_Ribbon.nii.gz \
##                ${RUN}_PVR.${pipeline}pp_minus_${pipeline}.nii.gz | tr -s '\t' '\n' > ${RUN}_PVR.${pipeline}pp_minus_${pipeline}.1D
## done

## 3dcalc -overwrite -a ${RUN}_PVR.COMPCOR.nii.gz -b ${RUN}_PVR.BASIC.nii.gz -expr 'a-b' -prefix ${RUN}_PVR.COMPCOR_minus_BASIC.nii.gz
3dROIstats -quiet \
           -mask ../../ALL/Schaefer2018_200Parcels_7Networks_order_mPP.GM_Ribbon.nii.gz \
           ${RUN}_PVR.COMPCOR_minus_BASIC.nii.gz | tr -s '\t' '\n' > ${RUN}_PVR.COMPCOR_minus_BASIC.1D
## 
3dROIstats -quiet \
           -mask ../../ALL/Schaefer2018_200Parcels_7Networks_order_mPP.GM_Ribbon.nii.gz \
           ${RUN}_PVR.COMPCORpp_minus_BASIC.nii.gz | tr -s '\t' '\n' > ${RUN}_PVR.COMPCORpp_minus_BASIC.1D
## 
##3dcalc -overwrite -a ${RUN}_PVR.COMPCORpp.nii.gz -b ${RUN}_PVR.BASIC.nii.gz -expr 'a-b' -prefix ${RUN}_PVR.COMPCORpp_minus_BASIC.nii.gz
echo "=================================="
echo "++ INFO: Script finished correctly"
echo "=================================="
