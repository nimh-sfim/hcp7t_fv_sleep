#!/bin/bash
set -e

module load afni
             
cd /data/SFIMJGC_HCP7T/HCP7T/${SBJ}/${RUN}
pwd

echo "++ INFO: Create Tshift and Discard version of the original data"
echo "++ ============================================================"
3dcalc -overwrite -a ${RUN}_orig.nii.gz[10..$] -expr 'a' -prefix ${RUN}_orig.discard.nii.gz
3dTshift -overwrite -prefix ${RUN}_orig.tshift.nii.gz ${RUN}_orig.discard.nii.gz

echo "++ INFO: Create Detrended version of the data"
echo "++ =========================================="
3dDetrend -overwrite -polort 2 -prefix rm.${RUN}_orig.detrend.nii.gz ${RUN}_orig.tshift.nii.gz
3dTstat -overwrite -mean -prefix rm.${RUN}_orig.detrend.MEAN.nii.gz  ${RUN}_orig.tshift.nii.gz
3dcalc -overwrite -a rm.${RUN}_orig.detrend.nii.gz -b rm.${RUN}_orig.detrend.MEAN.nii.gz -expr 'a+b' -prefix ${RUN}_orig.detrend.nii.gz
rm rm.${RUN}_orig.detrend.MEAN.nii.gz rm.${RUN}_orig.detrend.nii.gz


echo "++ INFO: Compute MEAN and STDEV maps"
echo "++ ================================="
3dTstat    -overwrite -mean  -prefix ${RUN}_orig.MEAN.nii.gz ${RUN}_orig.nii.gz
3dTstat    -overwrite -stdev -prefix ${RUN}_orig.STDV.nii.gz ${RUN}_orig.nii.gz

echo "++ INFO: Compute brain mask"
echo "++ ========================"
3dAutomask -overwrite        -prefix ${RUN}_orig.mask.FB_auto.nii.gz ${RUN}_orig.MEAN.nii.gz
             
echo "++ INFO: Compute Threshold for Sigma"
echo "++ ================================="
export STATS=(`3dROIstats -quiet -sigma -mask ${RUN}_orig.mask.FB_auto.nii.gz ${RUN}_orig.STDV.nii.gz`)
echo "++ INFO: STATS=${STATS[@]}"
echo "++ ----------------------------"
export STDEV_THR=`echo "${STATS[0]} + ${STATS[1]}" | bc -l`
echo "++ INFO: STDEV_THR=${STDEV_THR}"
echo "++ ----------------------------"

echo "++ INFO: Generate automatic version of CSF region mask"
echo "++ ==================================================="
3dcalc -overwrite -a ${RUN}_orig.STDV.nii.gz \
       -expr "ispositive(a-${STDEV_THR})" \
       -prefix ${RUN}_orig.mask.FV.auto_highSTDEV.nii.gz

echo "++ INFO: Generating Segmentation based on 3dSeg"
echo "++ ============================================"
if [ -d Segsy ]; then 
   rm -rf Segsy
fi

3dSeg -anat ${RUN}_orig.MEAN.nii.gz -mask AUTO -classes 'CSF ; GM ; WM' -bias_classes 'GM ; WM' -bias_fwhm 25 -mixfrac UNI -main_N 5 -blur_meth BFT

3dcalc -overwrite -a Segsy/Classes+orig. -expr 'equals(a,3)' -prefix ${RUN}_orig.mask.FV.auto_3dseg.nii.gz
             
echo "++ INFO: Generate automatic version of CSF region mask"
echo "++ ==================================================="
3dcalc -overwrite -a ${RUN}_orig.mask.FV.auto_highSTDEV.nii.gz \
                  -b ${RUN}_orig.mask.FV.auto_3dseg.nii.gz     \
                  -expr 'step(a+b)' \
                  -prefix ${RUN}_orig.mask.FV.auto_union.nii.gz 
                  
3dcalc -overwrite -a ${RUN}_orig.mask.FV.auto_union.nii.gz     \
       -expr 'a*within(i,60,70)*within(k,0,42)*within(j,30,64)' \
       -prefix ${RUN}_orig.mask.FV.auto_union.nii.gz

3dClusterize -overwrite -nosum -1Dformat \
             -inset ${RUN}_orig.mask.FV.auto_union.nii.gz \
             -idat 0 -ithr 0 -NN 1 -clust_nvox 50 -bisided -0.0001 0.0001 \
             -pref_map ${RUN}_orig.mask.FV.auto_union.nii.gz
             
3dcalc -overwrite -a ${RUN}_orig.mask.FV.auto_union.nii.gz \
       -expr 'equals(a,1)' \
       -prefix ${RUN}_orig.mask.FV.manual.nii.gz

echo "++ INFO: Erode automatic version of CSF region mask"
echo "++ ==================================================="
3dmask_tool -overwrite -inputs ${RUN}_orig.mask.FV.manual.nii.gz -dilate_inputs -1 -prefix ${RUN}_orig.mask.FV.manual.dil01.nii.gz
3dROIstats -nzvoxels -mask ${RUN}_orig.mask.FV.manual.dil01.nii.gz ${RUN}_orig.mask.FV.manual.dil01.nii.gz

echo "++ INFO: Script finished correctly"
echo "++ ==============================="
