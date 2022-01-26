#!/bin/bash

module load afni

set -e

PRJDIR='/data/SFIMJGC_HCP7T/HCP7T'

cd ${PRJDIR}/${SBJ}/${RUN}
pwd

echo "++ INFO: Extract average TS for whole ROI"
echo "++ ======================================"
3dROIstats -nobriklab -mask ${RUN}_orig.mask.FV.manual.nii.gz ${RUN}_orig.discard.nii.gz > ${RUN}_orig.discard.FV.mean.csv
3dROIstats -nobriklab -mask ${RUN}_orig.mask.FV.manual.nii.gz ${RUN}_orig.tshift.nii.gz  > ${RUN}_orig.tshift.FV.mean.csv
3dROIstats -nobriklab -mask ${RUN}_orig.mask.FV.manual.nii.gz ${RUN}_orig.detrend.nii.gz  > ${RUN}_orig.detrend.FV.mean.csv

3dROIstats -nobriklab -mask ${RUN}_orig.mask.FV.manual.dil01.nii.gz ${RUN}_orig.discard.nii.gz > ${RUN}_orig.discard.FV.dil01.mean.csv
3dROIstats -nobriklab -mask ${RUN}_orig.mask.FV.manual.dil01.nii.gz ${RUN}_orig.tshift.nii.gz  > ${RUN}_orig.tshift.FV.dil01.mean.csv
3dROIstats -nobriklab -mask ${RUN}_orig.mask.FV.manual.dil01.nii.gz ${RUN}_orig.detrend.nii.gz  > ${RUN}_orig.detrend.FV.dil01.mean.csv

echo "++ INFO: Create additional version (k) of the manually corrected mask"
echo "++ ============================================================================"
3dcalc  -overwrite -a ${RUN}_orig.mask.FV.manual.nii.gz -expr 'k*a' -prefix ${RUN}_orig.mask.FV.manual.k.nii.gz
#3dmerge -overwrite -1rank -prefix ${RUN}_orig.mask.FV.manual.rank.nii.gz ${RUN}_orig.mask.FV.manual.k.nii.gz

3dcalc  -overwrite -a ${RUN}_orig.mask.FV.manual.dil01.nii.gz -expr 'k*a' -prefix ${RUN}_orig.mask.FV.manual.dil01.k.nii.gz
#3dmerge -overwrite -1rank -prefix ${RUN}_orig.mask.FV.manual.dil01.rank.nii.gz ${RUN}_orig.mask.FV.manual.dil01.k.nii.gz

echo "++ INFO: Extract slice-by-slice profile"
echo "++ ===================================="
3dROIstats -nobriklab -mask ${RUN}_orig.mask.FV.manual.k.nii.gz ${RUN}_orig.discard.nii.gz  > ${RUN}_orig.discard.FV.k.csv
3dROIstats -nobriklab -mask ${RUN}_orig.mask.FV.manual.k.nii.gz ${RUN}_orig.tshift.nii.gz   > ${RUN}_orig.tshift.FV.k.csv
3dROIstats -nobriklab -mask ${RUN}_orig.mask.FV.manual.k.nii.gz ${RUN}_orig.detrend.nii.gz  > ${RUN}_orig.detrend.FV.k.csv

3dROIstats -nobriklab -mask ${RUN}_orig.mask.FV.manual.dil01.k.nii.gz ${RUN}_orig.discard.nii.gz  > ${RUN}_orig.discard.FV.dil01.k.csv
3dROIstats -nobriklab -mask ${RUN}_orig.mask.FV.manual.dil01.k.nii.gz ${RUN}_orig.tshift.nii.gz   > ${RUN}_orig.tshift.FV.dil01.k.csv
3dROIstats -nobriklab -mask ${RUN}_orig.mask.FV.manual.dil01.k.nii.gz ${RUN}_orig.detrend.nii.gz  > ${RUN}_orig.detrend.FV.dil01.k.csv

echo "++ INFO: Script finished correctly"
echo "++ ==============================="