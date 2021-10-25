#!/bin/bash
# Author: Javier Gonzalez-Castillo
# Data: 24th August, 2021
#
# Notes:
# ======
# In this project, we will pre-process data in different ways. The main pipelines are:
# 
# (1) mPP (minimally-preprocessed data). This is what we download from HCP. Has all spatial transformations, but no regression, scaling or bluring
# 
# (2) BASIC: blur (4mm), scaling, regression (motion + 1st der mot + legendre 5 + bpf 0.01 - 0.1 Hz)
#
# (3) AFNI COMPCOR: blur (4mm), scaling, regression (motion + 1st der mot + legendre 5 + PCAs lat vents + bpf 0.01 - 0.1Hz)
#
# (4) AFNI COMPCOR+: blur (4mm), scaling, regression (motion + 1st der mot + legendre 5 + PCAs lat vents + V4 signal + bpf 0.01 - 0.1Hz)
#
# (5) BEHZADI COMPCOR: blur (4mm), scaling, regression (motion + 1st der mot + legendre 5 + PCAs CSF & WM + bpf 0.01 - 0.1Hz).
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

# (2) Spatially Smooth mPP data
# =============================
echo "++ INFO: (2) Spatially Smooth mPP data"
echo "++ ==================================="
if [ ! -e ./${RUN}_mPP.blur.nii.gz ]; then
   3dBlurInMask -overwrite                                     \
                -mask ../ROI.FB.mPP.nii.gz                     \
                -FWHM ${BLUR_FWHM}                             \
                -input ./${RUN}_mPP.nii.gz[${VOLS_DISCARD}..$] \
                -prefix ./${RUN}_mPP.blur.nii.gz

   nifti_tool -strip_extras -overwrite -infiles ${RUN}_mPP.blur.nii.gz
else
   echo "++ WARNING: Blur file already exists. Not generating it again."
fi

# (3) Create SPC version of the blurred data
# ==========================================
echo "++ INFO: (3) Create SPC version of the blurred data"
echo "++ ================================================"
if [ ! -e ${RUN}_mPP.blur.scale.nii.gz ]; then
   3dTstat -overwrite -mean -mask ../ROI.FB.mPP.nii.gz -prefix ${RUN}_mPP.MEAN.nii.gz ${RUN}_mPP.blur.nii.gz
   3dcalc -overwrite                                  \
       -a ${RUN}_mPP.blur.nii.gz                      \
       -b ${RUN}_mPP.MEAN.nii.gz                      \
       -c ../ROI.FB.mPP.nii.gz                        \
       -expr  'c * min(200, a/b*100)*step(a)*step(b)' \
       -prefix ${RUN}_mPP.blur.scale.nii.gz

   nifti_tool -strip_extras -overwrite -infiles ${RUN}_mPP.blur.scale.nii.gz
else
   echo "++ WARNING: Blur.Scale file already exists. Not generating it again."
fi

# ================================================================================
# =============================== PART 1: QA Metrics =============================
# ================================================================================

# (4) Compute DVARS (SRMS) in minimally pre-processed data
# ====================================================
echo "++ INFO: (4) Computing DVARS (SRMS) in minimally pre-processed dataset"
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

# (5) Generate Bandpass filtering regressors
# ======================================
echo "++ INFO: (5) Generating bandpass filtering regressors"
echo "++ =================================================="
1dBport -nodata ${nt} ${tr} -band 0.01 0.1   -invert -nozero > ${RUN}_Bandpass_Regressors.txt

# (6) Generate CompCorr regressors (AFNI Style)
# =============================================
# In AFNI, Compcor is implemented as the top 3 PCA components present in the lateral ventricles
# We did this in the original submission, but one reviewer brough to our attention than conventional
# Compcorr uses 5 PCA components and is less restrictive in terms of tissue definition (CSF + WM).
# We keep these analyses so that we can compare the effects of both methods

echo "++ INFO: (6) Generating AFNI-style CompCorr regressors"
echo "++ ==================================================="
# 6.1 We extract PCA from detrended dataset (to match later regression)
3dTproject -overwrite \
           -mask ../ROI.FB.mPP.nii.gz \
           -polort ${POLORT} \
           -prefix rm.det_pcin_${RUN}.nii.gz \
           -input ${RUN}_mPP.nii.gz[${VOLS_DISCARD}..$]

# 6.2 Extract top 3 PCAs (those will be the regressors for CompCorr
3dpc -overwrite \
     -mask ../ROI.Vl_e.mPP.nii.gz \
     -pcsave 3 \
     -prefix ${RUN}_mPP_AFNI_CompCorr_Regressors \
     rm.det_pcin_${RUN}.nii.gz

# 6.3 Remove redundant files
rm ${RUN}_mPP_AFNI_CompCorr_Regressors+tlrc.????
rm ${RUN}_mPP_AFNI_CompCorr_Regressors0?.1D
rm ${RUN}_mPP_AFNI_CompCorr_Regressors_eig.1D
echo " + OUPUT: AFNI CompCorr Regressors File ${RUN}_AFNI_CompCorr_Regressors_vec.1D"

echo "++ INFO: (7) Generating Behzadi-style CompCorr regressors"
echo "++ ======================================================"
# (7) Generate CompCorr regressors (Behzadi Style)
# ================================================
# 7.1 Extract PCA from detrended dataset (to match later regression)
3dpc -overwrite                                        \
     -mask ../ROI.compcorr.mPP.nii.gz                      \
     -pcsave 5                                         \
     -prefix ${RUN}_mPP_Behzadi_CompCorr_Regressors \
     rm.det_pcin_${RUN}.nii.gz

# 7.2 Remove redundant files
rm ${RUN}_mPP_Behzadi_CompCorr_Regressors+tlrc.????
rm ${RUN}_mPP_Behzadi_CompCorr_Regressors0?.1D
rm ${RUN}_mPP_Behzadi_CompCorr_Regressors_eig.1D
echo " + OUPUT: Behzadi CompCorr Regressors File ${RUN}_Behzadi_CompCorr_Regressors_vec.1D"
rm rm.det_pcin_${RUN}.nii.gz

# ======================================================================================== #
# ================ Run different pre-processing pipelines (post mPP) ===================== #
# ======================================================================================== #
# (8) Basic Pre-processing 
# ========================
echo "++ INFO: (8) Basic Pre-processing"
echo "++ =============================="
3dTproject -overwrite                                          \
           -mask   ../ROI.FB.mPP.nii.gz                        \
           -input  ${RUN}_mPP.blur.scale.nii.gz                \
           -polort ${POLORT}                                   \
           -ort    ${RUN}_Movement_Regressors_dt.discard10.txt \
           -ort    ${RUN}_Bandpass_Regressors.txt              \
           -prefix ${RUN}_BASIC.nii.gz
echo " + OUTPUT from Basic Pipeline: ${RUN}_BASIC.nii.gz"

# (9) Basic Pre-processing - No Filtering
# =========================================
echo "++ INFO: (9) Basic Pre-processing (no filtering - input to rapidtide)"
echo "++ =================================================================="
3dTproject -overwrite                                          \
           -mask   ../ROI.FB.mPP.nii.gz                        \
           -input  ${RUN}_mPP.blur.scale.nii.gz                \
           -polort ${POLORT}                                   \
           -ort    ${RUN}_Movement_Regressors_dt.discard10.txt \
           -prefix ${RUN}_BASICnobpf.nii.gz
echo " + OUTPUT from Basic Pipeline (no filtering): ${RUN}_BASICnobpf.nii.gz"

# (10) AFNI CompCor Pipeline
# =========================
echo "++ INFO: (10) AFNI CompCorr Pre-processing"
echo "++ ======================================="
3dTproject -overwrite                                           \
           -mask   ../ROI.FB.mPP.nii.gz                         \
           -input  ${RUN}_mPP.blur.scale.nii.gz                 \
           -polort ${POLORT}                                    \
           -ort    ${RUN}_Movement_Regressors_dt.discard10.txt  \
           -ort    ${RUN}_Bandpass_Regressors.txt               \
           -ort    ${RUN}_mPP_AFNI_CompCorr_Regressors_vec.1D        \
           -prefix ${RUN}_AFNI_COMPCOR.nii.gz
echo " + OUTPUT from AFNI CompCor: ${RUN}_AFNI_COMPCOR.nii.gz"

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

# (12) AFNI CompCor+ (Most likelly to be deprecated during the reviews)
# ====================================================================
echo "++ INFO: (12) AFNI CompCorr+ Pre-processing"
echo "++ ========================================"
3dTproject -overwrite                                          \
           -mask   ../ROI.FB.mPP.nii.gz                        \
           -input  ${RUN}_mPP.blur.scale.nii.gz                \
           -polort ${POLORT}                                   \
           -ort    ${RUN}_Movement_Regressors_dt.discard10.txt \
           -ort    ${RUN}_Bandpass_Regressors.txt              \
           -ort    ${RUN}_mPP_AFNI_CompCorr_Regressors_vec.1D  \
           -ort    ${RUN}_mPP.Signal.V4_grp.1D                 \
           -prefix ${RUN}_AFNI_COMPCORp.nii.gz
echo " + OUTPUT from AFNI CompCor+: ${RUN}_COMPCORp.nii.gz"

echo "=================================="
echo "++ INFO: Script finished correctly"
echo "=================================="
