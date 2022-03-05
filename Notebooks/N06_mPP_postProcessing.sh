#!/bin/bash
# Author: Javier Gonzalez-Castillo
# Data: 3th March, 2022
#
# Notes:
# ======
# In this project, we will pre-process data in different ways. The main pipelines are:
# 
# (1) mPP (minimally-preprocessed data). This is what we download from HCP. Has all spatial transformations, but no regression, scaling or bluring
# 
# (2) BASIC: blur (4mm), scaling, regression (motion + 1st der mot + legendre 5 + bpf 0.01 - 0.1 Hz)
#
# (3) BASIC: blur (4mm), scaling, regression (motion + 1st der mot + legendre 5) --> Only for rapidtide (so it does it's own filtering)
#
# (4) BEHZADI COMPCOR: blur (4mm), scaling, regression (motion + 1st der mot + legendre 5 + PCAs CSF & WM + bpf 0.01 - 0.1Hz).
#
# ===================================================================================================================================================

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

## # (2) Spatially Smooth mPP data
## # =============================
## echo "++ INFO: (2) Spatially Smooth mPP data"
## echo "++ ==================================="
## if [ ! -e ./${RUN}_mPP.blur.nii.gz ]; then
##    3dBlurInMask -overwrite                                     \
##                 -mask ../ROI.FB.mPP.nii.gz                     \
##                 -FWHM ${BLUR_FWHM}                             \
##                 -input ./${RUN}_mPP.nii.gz[${VOLS_DISCARD}..$] \
##                 -prefix ./${RUN}_mPP.blur.nii.gz
## 
##    nifti_tool -strip_extras -overwrite -infiles ${RUN}_mPP.blur.nii.gz
## else
##    echo "++ WARNING: Blur file already exists. Not generating it again."
## fi
## 
## # (3) Create SPC version of the blurred data
## # ==========================================
## echo "++ INFO: (3) Create SPC version of the blurred data"
## echo "++ ================================================"
## if [ ! -e ${RUN}_mPP.blur.scale.nii.gz ]; then
##    3dTstat -overwrite -mean -mask ../ROI.FB.mPP.nii.gz -prefix ${RUN}_mPP.MEAN.nii.gz ${RUN}_mPP.blur.nii.gz
##    3dcalc -overwrite                                  \
##        -a ${RUN}_mPP.blur.nii.gz                      \
##        -b ${RUN}_mPP.MEAN.nii.gz                      \
##        -c ../ROI.FB.mPP.nii.gz                        \
##        -expr  'c * min(200, a/b*100)*step(a)*step(b)' \
##        -prefix ${RUN}_mPP.blur.scale.nii.gz
## 
##    nifti_tool -strip_extras -overwrite -infiles ${RUN}_mPP.blur.scale.nii.gz
## else
##    echo "++ WARNING: Blur.Scale file already exists. Not generating it again."
## fi
## 
# (4) Extract Representative Timseries for scale.blur 
# -----------------------------------------------------
echo "++ INFO: (4) Extract Representative Timseries"
echo "==============================================="
for items in ${ROI_FV_PATH},'V4_grp' ${ROI_FVlt_PATH},'V4lt_grp' ${ROI_FVut_PATH},'V4ut_grp' ../ROI.V4_e.mPP.nii.gz,'V4_e' ../ROI.Vl_e.mPP.nii.gz,'Vl_e' ../ROI.GM.mPP.nii.gz,'GM' ../ROI.FB.mPP.nii.gz,'FB' ../ROI.WM_e.mPP.nii.gz,'WM_e'
do 
   IFS="," ; set -- $items
   echo " + INFO: Mask = $1 | Suffix = $2 --> Output = ${RUN}_BASIC.Signal.$2.1D"
   3dmaskave -quiet -mask $1 ${RUN}_mPP.blur.scale.nii.gz   > rm.aux.$2.1D
   3dDetrend -prefix - -polort ${POLORT} rm.aux.$2.1D\'  > rm.det.aux.$2.1D
   cat rm.det.aux.$2.1D | tr -s ' ' '\n' | sed '/^$/d' > ${RUN}_mPP.blur.scale.Signal.$2.1D
   rm rm.aux.$2.1D rm.det.aux.$2.1D
   1d_tool.py -overwrite -infile ${RUN}_mPP.blur.scale.Signal.$2.1D -derivative -write ${RUN}_mPP.blur.scale.Signal.$2.der.1D
done

# ================================================================================
# =============================== PART 1: QA Metrics =============================
# ================================================================================

# (5) Compute DVARS (SRMS) in minimally pre-processed data
# ====================================================
echo "++ INFO: (5) Computing DVARS (SRMS) in minimally pre-processed dataset"
echo "++ ==================================================================="
3dTto1D -overwrite                                   \
        -input ${RUN}_mPP.nii.gz[${VOLS_DISCARD}..$] \
        -mask ../ROI.FB.mPP.nii.gz                   \
        -method srms                                 \
        -prefix ${RUN}_Movement_SRMS.1D
echo "++ Output: ${RUN}_Movement_SRMS.1D"

tail -n ${nt} ${RUN}_Movement_Regressors.txt    >  ${RUN}_Movement_Regressors.discard10.txt
tail -n ${nt} ${RUN}_Movement_Regressors_dt.txt >  ${RUN}_Movement_Regressors_dt.discard10.txt

# =================================================================================
# =============================== PART 2: Generate Regressors of interest =========
# =================================================================================

# (6) Generate Bandpass filtering regressors
# ======================================
echo "++ INFO: (6) Generating bandpass filtering regressors"
echo "++ =================================================="
1dBport -nodata ${nt} ${tr} -band 0.01 0.1   -invert -nozero > ${RUN}_Bandpass_Regressors.txt

# (7) Generate CompCorr regressors (Behzadi Style)
# ================================================
echo "++ INFO: (7) Generating Behzadi-style CompCorr regressors"
echo "++ ======================================================"
# 7.1 We extract PCA from detrended dataset (to match later regression)
3dTproject -overwrite \
            -mask ../ROI.FB.mPP.nii.gz \
            -polort ${POLORT} \
            -prefix rm.det_pcin_${RUN}.nii.gz \
            -input ${RUN}_mPP.nii.gz[${VOLS_DISCARD}..$]

# 7.2 Extract PCA from detrended dataset (to match later regression)
3dpc -overwrite                                        \
     -mask ../ROI.compcorr.mPP.nii.gz                      \
     -pcsave 5                                         \
     -prefix ${RUN}_mPP_Behzadi_CompCorr_Regressors \
     rm.det_pcin_${RUN}.nii.gz

# 7.3 Remove redundant files
rm ${RUN}_mPP_Behzadi_CompCorr_Regressors+tlrc.????
rm ${RUN}_mPP_Behzadi_CompCorr_Regressors0?.1D
rm ${RUN}_mPP_Behzadi_CompCorr_Regressors_eig.1D
echo " + OUPUT: Behzadi CompCorr Regressors File ${RUN}_Behzadi_CompCorr_Regressors_vec.1D"
rm rm.det_pcin_${RUN}.nii.gz

# ======================================================================================== #
# ================ Run different pre-processing pipelines (post mPP) ===================== #
# ======================================================================================== #

# (8) Create Reference for comparison of denoising strategies
# ===========================================================
echo "++ INFO: (8) Reference Pre-processing"
echo "++ =================================="
3dTproject -overwrite                                          \
           -mask   ../ROI.FB.mPP.nii.gz                        \
           -input  ${RUN}_mPP.blur.scale.nii.gz                \
           -polort ${POLORT}                                   \
           -ort    ${RUN}_Bandpass_Regressors.txt              \
           -prefix ${RUN}_Reference.nii.gz
echo " + OUTPUT from Basic Pipeline: ${RUN}_Reference.nii.gz"

# (8.1) Extract Representative Timeseries for Reference
echo "++ INFO: (8.1) Extract Representative Timseries for Reference"
echo "============================================================="
for items in ${ROI_FV_PATH},'V4_grp' ${ROI_FVlt_PATH},'V4lt_grp' ${ROI_FVut_PATH},'V4ut_grp' ../ROI.V4_e.mPP.nii.gz,'V4_e' ../ROI.Vl_e.mPP.nii.gz,'Vl_e' ../ROI.GM.mPP.nii.gz,'GM' ../ROI.FB.mPP.nii.gz,'FB' ../ROI.WM_e.mPP.nii.gz,'WM_e'
do 
   IFS="," ; set -- $items
   echo " + INFO: Mask = $1 | Suffix = $2 --> Output = ${RUN}_BASIC.Signal.$2.1D"
   3dmaskave -quiet -mask $1 ${RUN}_Reference.nii.gz   > rm.aux.$2.1D
   3dDetrend -prefix - -polort ${POLORT} rm.aux.$2.1D\'  > rm.det.aux.$2.1D
   cat rm.det.aux.$2.1D | tr -s ' ' '\n' | sed '/^$/d' > ${RUN}_Reference.Signal.$2.1D
   rm rm.aux.$2.1D rm.det.aux.$2.1D
   1d_tool.py -overwrite -infile ${RUN}_Reference.Signal.$2.1D -derivative -write ${RUN}_Reference.Signal.$2.der.1D
done

# (8.2) Compute variance for Reference
echo "++ INFO (8.2) Compute variance for output of Reference pipeline"
echo "==============================================================="
3dTstat -overwrite -stdev -prefix ${RUN}_Reference.VAR.nii.gz ${RUN}_Reference.nii.gz
3dcalc  -overwrite -a ${RUN}_Reference.VAR.nii.gz -expr 'a*a' -prefix ${RUN}_Reference.VAR.nii.gz 

# (9) Basic Pre-processing 
# ========================
echo "++ INFO: (9) Basic Pre-processing"
echo "++ =============================="
3dTproject -overwrite                                          \
           -mask   ../ROI.FB.mPP.nii.gz                        \
           -input  ${RUN}_mPP.blur.scale.nii.gz                \
           -polort ${POLORT}                                   \
           -ort    ${RUN}_Movement_Regressors_dt.discard10.txt \
           -ort    ${RUN}_Bandpass_Regressors.txt              \
           -prefix ${RUN}_BASIC.nii.gz
echo " + OUTPUT from Basic Pipeline: ${RUN}_BASIC.nii.gz"

# (9.1) Extract Representative Timseries for BASIC
echo "++ INFO: (9.1) Extract Representative Timseries"
echo "==============================================="
for items in ${ROI_FV_PATH},'V4_grp' ${ROI_FVlt_PATH},'V4lt_grp' ${ROI_FVut_PATH},'V4ut_grp' ../ROI.V4_e.mPP.nii.gz,'V4_e' ../ROI.Vl_e.mPP.nii.gz,'Vl_e' ../ROI.GM.mPP.nii.gz,'GM' ../ROI.FB.mPP.nii.gz,'FB' ../ROI.WM_e.mPP.nii.gz,'WM_e'
do 
   IFS="," ; set -- $items
   echo " + INFO: Mask = $1 | Suffix = $2 --> Output = ${RUN}_BASIC.Signal.$2.1D"
   3dmaskave -quiet -mask $1 ${RUN}_BASIC.nii.gz   > rm.aux.$2.1D
   3dDetrend -prefix - -polort ${POLORT} rm.aux.$2.1D\'  > rm.det.aux.$2.1D
   cat rm.det.aux.$2.1D | tr -s ' ' '\n' | sed '/^$/d' > ${RUN}_BASIC.Signal.$2.1D
   rm rm.aux.$2.1D rm.det.aux.$2.1D
   1d_tool.py -overwrite -infile ${RUN}_BASIC.Signal.$2.1D -derivative -write ${RUN}_BASIC.Signal.$2.der.1D
done

# (9.2) Compute variance for BASIC
echo "++ INFO (9.2) Compute variance for output of BASIC pipeline"
echo "==========================================================="
3dTstat -overwrite -stdev -prefix ${RUN}_BASIC.VAR.nii.gz ${RUN}_BASIC.nii.gz
3dcalc  -overwrite -a ${RUN}_BASIC.VAR.nii.gz -expr 'a*a' -prefix ${RUN}_BASIC.VAR.nii.gz 

# (10) Basic Pre-processing - No Filtering
# =========================================
echo "++ INFO: (10) Basic Pre-processing (no filtering - input to rapidtide)"
echo "++ =================================================================="
3dTproject -overwrite                                          \
           -mask   ../ROI.FB.mPP.nii.gz                        \
           -input  ${RUN}_mPP.blur.scale.nii.gz                \
           -polort ${POLORT}                                   \
           -ort    ${RUN}_Movement_Regressors_dt.discard10.txt \
           -prefix ${RUN}_BASICnobpf.nii.gz
echo " + OUTPUT from Basic Pipeline (no filtering): ${RUN}_BASICnobpf.nii.gz"

# (11) Behzadi CompCor Pipeline
# =============================
echo "++ INFO: (11) Behzadi CompCorr Pre-processing"
echo "++ =========================================="
3dTproject -overwrite                                           \
           -mask   ../ROI.FB.mPP.nii.gz                         \
           -input  ${RUN}_mPP.blur.scale.nii.gz                 \
           -polort ${POLORT}                                    \
           -ort    ${RUN}_Movement_Regressors_dt.discard10.txt  \
           -ort    ${RUN}_Bandpass_Regressors.txt               \
           -ort    ${RUN}_mPP_Behzadi_CompCorr_Regressors_vec.1D        \
           -prefix ${RUN}_Behzadi_COMPCOR.nii.gz
echo " + OUTPUT from Behzadi CompCor: ${RUN}_Behzadi_COMPCOR.nii.gz"

# (11.1) Extract Representative Timseries for BASIC
echo "++ INFO: (11.1) Extract Representative Timseries"
echo "================================================"
for items in ${ROI_FV_PATH},'V4_grp' ${ROI_FVlt_PATH},'V4lt_grp' ${ROI_FVut_PATH},'V4ut_grp' ../ROI.V4_e.mPP.nii.gz,'V4_e' ../ROI.Vl_e.mPP.nii.gz,'Vl_e' ../ROI.GM.mPP.nii.gz,'GM' ../ROI.FB.mPP.nii.gz,'FB' ../ROI.WM_e.mPP.nii.gz,'WM_e'
do 
  IFS="," ; set -- $items
   echo " + INFO: Mask = $1 | Suffix = $2 --> Output = ${RUN}_Behzadi_COMPCOR.Signal.$2.1D"
   3dmaskave -quiet -mask $1 ${RUN}_Behzadi_COMPCOR.nii.gz   > rm.aux.$2.1D
   3dDetrend -prefix - -polort ${POLORT} rm.aux.$2.1D\'  > rm.det.aux.$2.1D
   cat rm.det.aux.$2.1D | tr -s ' ' '\n' | sed '/^$/d' > ${RUN}_Behzadi_COMPCOR.Signal.$2.1D
   rm rm.aux.$2.1D rm.det.aux.$2.1D
   1d_tool.py -overwrite -infile ${RUN}_Behzadi_COMPCOR.Signal.$2.1D -derivative -write ${RUN}_Behzadi_COMPCOR.Signal.$2.der.1D
done

# 11.2 Compute variance for COMPCOR
echo "++ INFO (11.2) Compute variance for output of COMPCOR pipeline"
echo "=============================================================="
3dTstat -overwrite -stdev -prefix ${RUN}_COMPCOR.VAR.nii.gz ${RUN}_Behzadi_COMPCOR.nii.gz
3dcalc  -overwrite -a ${RUN}_COMPCOR.VAR.nii.gz -expr 'a*a' -prefix ${RUN}_COMPCOR.VAR.nii.gz 

echo "++ INFO: (12) Maps of correlation between the different pre-processing"
echo "++ ==================================================================="
3dTcorrelate -overwrite -prefix ${RUN}_BASIC_2_Reference.nii.gz   ${RUN}_BASIC.nii.gz ${RUN}_Reference.nii.gz
3dTcorrelate -overwrite -prefix ${RUN}_COMPCOR_2_Reference.nii.gz ${RUN}_Behzadi_COMPCOR.nii.gz ${RUN}_Reference.nii.gz

echo "=================================="
echo "++ INFO: Script finished correctly"
echo "=================================="
