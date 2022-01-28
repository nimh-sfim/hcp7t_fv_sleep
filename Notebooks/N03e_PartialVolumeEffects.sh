#!/bin/bash

module load afni
source ./common_variables.sh
set -e

PRJDIR='/data/SFIMJGC_HCP7T/HCP7T'

cd ${PRJDIR}/${SBJ}/${RUN}
pwd

echo "++ INFO: Generate top FV masks in original and MNI space"
echo "++ ====================================================="
3dcalc -overwrite -a ${RUN}_orig.mask.FV.manual.dil01.nii.gz -expr 'a*isnegative(k-25)' -prefix ${RUN}_orig.mask.iFV.manual.dil01.nii.gz

echo "++ INFO: Extract representative time-series for those ROIs"
echo "=========================================================="
3dmaskave -quiet -mask ${RUN}_orig.mask.iFV.manual.dil01.nii.gz ${RUN}_orig.discard.nii.gz > rm.${RUN}_orig.mask.iFV.manual.dil01.1D
3dDetrend -prefix - -polort ${POLORT} rm.${RUN}_orig.mask.iFV.manual.dil01.1D\' > rm2.${RUN}_orig.mask.iFV.manual.dil01.1D
cat rm2.${RUN}_orig.mask.iFV.manual.dil01.1D | tr -s ' ' '\n' | sed '/^$/d' > ${RUN}_orig.mask.iFV.manual.dil01.1D
rm rm.${RUN}_orig.mask.iFV.manual.dil01.1D rm2.${RUN}_orig.mask.iFV.manual.dil01.1D

3dTcorr1D -overwrite -prefix ${RUN}_orig.mask.iFV.manual.dil01.map.nii.gz -mask ${RUN}_orig.mask.iFV.manual.dil01.nii.gz ${RUN}_PA_orig.nii.gz[10..$] ${RUN}_orig.mask.iFV.manual.dil01.1D

echo "++ Script Finished correctly"
echo "============================"
