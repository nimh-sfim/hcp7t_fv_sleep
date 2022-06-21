#!/bin/bash

set -e

PRJDIR='/data/SFIMJGC_HCP7T/HCP7T'

cd ${PRJDIR}/${SBJ}/${RUN}

echo "++ INFO: Extract average TS for whole ROI"
echo "++ ======================================"
3dROIstats -nobriklab -mask ${RUN}_orig.mask.FV.manual.nii.gz ${RUN}_orig.discard.nii.gz > ${RUN}_orig.discard.FV.mean.csv
3dROIstats -nobriklab -mask ${RUN}_orig.mask.FV.manual.nii.gz ${RUN}_orig.tshift.nii.gz  > ${RUN}_orig.tshift.FV.mean.csv
3dROIstats -nobriklab -mask ${RUN}_orig.mask.FV.manual.nii.gz ${RUN}_orig.detrend.nii.gz  > ${RUN}_orig.detrend.FV.mean.csv

echo "++ INFO: Create additional versions (k and rank) of the manually corrected mask"
echo "++ ============================================================================"
3dcalc  -overwrite -a ${RUN}_orig.mask.FV.manual.nii.gz -expr 'k*a' -prefix ${RUN}_orig.mask.FV.manual.k.nii.gz
3dmerge -overwrite -1rank -prefix ${RUN}_orig.mask.FV.manual.rank.nii.gz ${RUN}_orig.mask.FV.manual.k.nii.gz

echo "++ INFO: Extract slice-by-slice profile"
echo "++ ===================================="
3dROIstats -nobriklab -mask ${RUN}_orig.mask.FV.manual.k.nii.gz ${RUN}_orig.discard.nii.gz  > ${RUN}_orig.discard.FV.k.csv
3dROIstats -nobriklab -mask ${RUN}_orig.mask.FV.manual.k.nii.gz ${RUN}_orig.tshift.nii.gz   > ${RUN}_orig.tshift.FV.k.csv
3dROIstats -nobriklab -mask ${RUN}_orig.mask.FV.manual.k.nii.gz ${RUN}_orig.detrend.nii.gz  > ${RUN}_orig.detrend.FV.k.csv

#3dROIstats -nobriklab -mask ${RUN}_orig.mask.FV.manual.rank.nii.gz ${RUN}_orig.discard.nii.gz > ${RUN}_orig.discard.FV.rank.csv
#3dROIstats -nobriklab -mask ${RUN}_orig.mask.FV.manual.rank.nii.gz ${RUN}_orig.tshift.nii.gz  > ${RUN}_orig.tshift.FV.rank.csv
#3dROIstats -nobriklab -mask ${RUN}_orig.mask.FV.manual.rank.nii.gz ${RUN}_orig.detrend.nii.gz  > ${RUN}_orig.detrend.FV.rank.csv

# Jan 22, 2022 - Javier: This was for testing some ideas, but did not make it into the final manuscript
#echo "++ INFO: Run melodic on the different scenarios"
#echo "++ ============================================"
#for kind in discard tshift detrend
#do
#  echo "++ INFO: Working on scenario: ${kind}"
#  echo "++ ----------------------------------"
#  # Remove prior dir
#  if [ -d melodic.${kind} ]; then
#     rm -rf ./melodic.${kind}
#  fi
#  # Run melodic
#  melodic -i ${RUN}_orig.${kind}.nii.gz \
#          -o ./melodic.${kind} \
#          --Oall \
#          -m ${RUN}_orig.mask.FV.manual.nii.gz \
#          -v --nobet --dimest=aic --tr=1 --report
#  # Create links to mean
#  ln -s ${PRJDIR}/${SBJ}/${RUN}/${RUN}_orig.MEAN.nii.gz ./melodic.${kind}/${RUN}_orig.MEAN.nii.gz
#  ln -s ${PRJDIR}/${SBJ}/${RUN}/${RUN}_orig.STDV.nii.gz ./melodic.${kind}/${RUN}_orig.STDV.nii.gz
#done

echo "++ INFO: Script finished correctly"
echo "++ ==============================="
