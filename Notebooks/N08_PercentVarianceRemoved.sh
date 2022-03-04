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

for scenario in BASIC BASICpp COMPCOR COMPCORpp
do
  3dcalc -overwrite \
         -a ${RUN}_Reference.VAR.nii.gz \
         -b ${RUN}_${scenario}.VAR.nii.gz \
         -m ${DATA_DIR}/ALL/ALL_EPI_FBmask.nii.gz \
         -expr 'm*100*(a-b)/a' \
         -prefix ${RUN}_PVR.${scenario}.nii.gz
done  
echo "=================================="
echo "++ INFO: Script finished correctly"
echo "=================================="
