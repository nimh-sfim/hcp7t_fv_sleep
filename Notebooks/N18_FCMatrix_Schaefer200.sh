#!/bin/bash
set -e
module load afni

cd /data/SFIMJGC_HCP7T/HCP7T/${SBJ}/${RUN}/

echo "++ Working Dir: `pwd`"
# Compute Connectivity Matrix (without regressing the signal)
# ===========================================================
for REGRESSION in Reference BASIC GSR BASICpp Behzadi_COMPCOR Behzadi_COMPCORpp
do
    echo "++ INFO: Extracting Full Brain Connectivity Matrix for ${RUN}_${REGRESSION}.nii.gz"
    3dNetCorr -overwrite \
              -mask ../ROI.FB.mPP.nii.gz \
              -inset ${RUN}_${REGRESSION}.nii.gz \
              -in_rois ../../ALL/Schaefer2018_200Parcels_7Networks_order_mPP.GM_Ribbon.nii.gz \
              -prefix ${RUN}_${REGRESSION}.Shaeffer2018_200Parcels 
              
   rm ${RUN}_${REGRESSION}.Shaeffer2018_200Parcels_000.niml.dset
   rm ${RUN}_${REGRESSION}.Shaeffer2018_200Parcels_000.roidat
done

echo "================================== EDN OF SCRIPT =================================="
