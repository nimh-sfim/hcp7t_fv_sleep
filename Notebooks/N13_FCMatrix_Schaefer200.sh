#!/bin/bash
set -e
module load afni

cd /data/SFIMJGC_HCP7T/HCP7T/${SBJ}/${RUN}/

echo "++ Working Dir: `pwd`"
# Compute Connectivity Matrix (without regressing the signal)
# ===========================================================

for SUFFIX in BASIC Behzadi_COMPCOR AFNI_COMPCOR AFNI_COMPCORp
do
    echo "++ INFO: Extracting Full Brain Connectivity Matrix for ${RUN}_${SUFFIX}.nii.gz"
    3dNetCorr -overwrite \
              -mask ../ROI.FB.mPP.nii.gz \
              -inset ${RUN}_${SUFFIX}.nii.gz \
              -in_rois ../../ALL/Schaefer2018_200Parcels_7Networks_order_mPP.GM_Ribbon.nii.gz \
              -prefix ${RUN}_${SUFFIX}.Shaeffer2018_200Parcels 
              
   rm ${RUN}_${SUFFIX}.Shaeffer2018_200Parcels_000.niml.dset
   rm ${RUN}_${SUFFIX}.Shaeffer2018_200Parcels_000.roidat
done
#              -mask ../ROI.FB.mPP.nii.gz \

echo "================================== EDN OF SCRIPT =================================="
