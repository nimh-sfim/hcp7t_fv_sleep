set -e
DATA_DIR='/data/SFIMJGC_HCP7T/HCP7T'
RUN_DIR=`echo ${DATA_DIR}/${SBJ}/${RUN}`

cd ${RUN_DIR}
pwd

#echo "++ INFO: Insert Slice Timing information in original fMRI files"
#3drefit -Tslices '*0.001' `cat /data/SFIMJGC_HCP7T/HCP7T/ALL/ALL_orig.SliceTimings.info` ${RUN}_orig.nii.gz

#echo "++ INFO: Time-shift correction"
#3dTshift -overwrite -tzero 0 -prefix ${RUN}_orig.tshift.nii.gz ${RUN}_orig.nii.gz[10..$]

#echo "++ INFO: Extract TS per slice: ORIG + FULL ROI"
#3dROIstats -nobriklab -mask ${RUN}_orig.FVmask.nii.gz      ${RUN}_orig.nii.gz > ${RUN}_orig.FVmask.bySlice.csv

#echo "++ INFO: Extract TS per slice: ORIG + VENTRAL PORTION ROI"
#3dROIstats -nobriklab -mask ${RUN}_orig.FVmask.tube.nii.gz ${RUN}_orig.nii.gz > ${RUN}_orig.FVmask.tube.bySlice.csv

#echo "++ INFO: Extract TS per slice: TSHIFTED + VENTRAL PORTION ROI"
#3dROIstats -nobriklab -mask ${RUN}_orig.FVmask.tube.nii.gz ${RUN}_orig.tshift.nii.gz > ${RUN}_orig.tshift.FVmask.tube.bySlice.csv

#echo "++ INFO: Extract TS per slice: TSHIFTED + FULL ROI"
#3dROIstats -nobriklab -mask ${RUN}_orig.FVmask.nii.gz      ${RUN}_orig.tshift.nii.gz > ${RUN}_orig.tshift.FVmask.bySlice.csv

#echo "++ INFO: Extract TS per slice: ORIG + FULL ROI (extended)"
#3dROIstats -nobriklab -mask ${RUN}_orig.FVmask2.nii.gz      ${RUN}_orig.nii.gz        > ${RUN}_orig.FVmask2.bySlice.csv

#echo "++ INFO: Extract TS per slice: TSHIFTED + FULL ROI (extended)"
#3dROIstats -nobriklab -mask ${RUN}_orig.FVmask2.nii.gz      ${RUN}_orig.tshift.nii.gz > ${RUN}_orig.tshift.FVmask2.bySlice.csv

echo "++ INFO: Extract TS per slice: ORIG + FULL ROI (extended)"
3dROIstats -nobriklab -mask ${RUN}_orig.FVmask3.nii.gz      ${RUN}_orig.nii.gz        > ${RUN}_orig.FVmask3.bySlice.csv

echo "++ INFO: Extract TS per slice: TSHIFTED + FULL ROI (extended)"
3dROIstats -nobriklab -mask ${RUN}_orig.FVmask3.nii.gz      ${RUN}_orig.tshift.nii.gz > ${RUN}_orig.tshift.FVmask3.bySlice.csv

echo "++ INFO: Script Finished correctly"
