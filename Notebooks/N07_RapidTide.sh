#!/bin/bash/sh
# Author: Javier Gonzalez-Castillo
# Date: August 25th, 2021
# Notes:
#
# RapidTide is run for two different purposes: 
#     (1) Obtain delay maps for maximal correlation, 
#     (2) perform dynamic (e.g. lag-aware) removal of V4 signal. 
#
# For (1) we use unipolar, while for (2) we use bipolar.
#
# For this reason, there will be 2 different attempts at running rapidTide as follows:
#
# (a) A UNI attempt after BASICnobpf. We do this prior to removal of any ventricular signal, becuase
#     the 0.05 Hz flucutations are present in all ventricles and if we do it after COMPCOR, then part of 
#     the signal of interest would be gone.
# 
# (b) A BIPOLAR attempts after COMPCORnopbpf, which will generate the COMPCORpp pipeline
#
# =========================================================================================================

set -e

source ./common_variables.sh
module load afni

# Enter scripts directory
# =======================
echo "++ Entering Notebooks directory..."
cd ${SCRIPTS_DIR}

# Activate miniconda
# ==================
echo "++ Activating miniconda"
. /data/SFIMJGC_HCP7T/Apps/miniconda38/etc/profile.d/conda.sh

# Activate vigilance environment
# ==============================
echo "++ Activating rapidtide environment"
conda activate hcp7t_fv_sleep_rapidtide_env

# Unset DISPLAY variable so that we don't get an error about access to XDisplay
echo "++ Unsetting the DISPLAY environment variable"
unset DISPLAY

cd ${DATA_DIR}/${SBJ}/${RUN}/
WORKDIR=`pwd`
echo "++ Working Directory: ${WORKDIR}"

echo "++ INFO: Create directory for rapidtide results"
echo "++ ============================================"
if [ -d ${RUN}_BASICnobpf.rapidtide ]; then
     rm -rf ${RUN}_BASICnobpf.rapidtide
fi
mkdir ${RUN}_BASICnobpf.rapidtide

# =================================================================================== #
# ================== RAPIDTIDE ====================================================== #
# =================================================================================== #

# (1) Run rapidtide
# -------------
echo "++ INFO: (1) Run Rapidtide"
echo "++ ======================="
pwd
rapidtide2x_legacy  ./${RUN}_BASICnobpf.nii.gz \
           ./${RUN}_BASICnobpf.rapidtide/${RUN}_BASICnobpf \
           -F 0.01,0.1 --multiproc --nprocs=32 \
           -r -17,15 \
           --passes=1 -O 1 --regressor=${RUN}_mPP.Signal.V4lt_grp.1D

# (1.1) Remove many ouputs of rapidtide that we do not need (for space)
echo "++ INFO: (1.1) Remove unncessary outputs from RapidTide2"
echo "++ ==============================================="
for output_suffix in corrdistdata_pass1.txt datatoremove.nii.gz fitcoff.nii.gz fitNorm.nii.gz fitR2.nii.gz fitR.nii.gz formattedcommandline.txt globallaghist_pass1_centerofmass.txt globallaghist_pass1_peak.txt globallaghist_pass1.txt laghist_centerofmass.txt laghist.txt lagmask.nii.gz lagsigma.nii.gz mean.nii.gz memusage.csv MTT.nii.gz nullcorrelationhist_pass1_centerofmass.txt nullcorrelationhist_pass1_peak.txt nullcorrelationhist_pass1.txt options.txt p_lt_0p001_thresh.txt p_lt_0p005_thresh.txt p_lt_0p010_thresh.txt p_lt_0p050_thresh.txt referenceautocorr_pass1.txt reference_fmrires.txt reference_origres_prefilt.txt reference_origres.txt reference_resampres.txt Rhist_centerofmass.txt Rhist_peak.txt Rhist.txt runtimings.txt sigfit.txt strengthhist_centerofmass.txt strengthhist_peak.txt strengthhist.txt widthhist_centerofmass.txt widthhist_peak.txt widthhist.txt
do
   for scenario in BASICnobpf
   do
       if [ -e ./${RUN}_${scenario}.rapidtide/${RUN}_${scenario}_${output_suffix} ]; then 
            rm ./${RUN}_${scenario}.rapidtide/${RUN}_${scenario}_${output_suffix}; fi
   done
done

# (1.2) Correct the headers for the remainder files
echo "++ INFO: (1.2) Correct the headers for the remainder files"
echo "++ ======================================================="
for output_suffix in corrout filtereddata gaussout lagregressor lagstrengths lagtimes p_lt_0p001_mask p_lt_0p005_mask p_lt_0p010_mask p_lt_0p050_mask R2
do
  for scenario in BASICnobpf
   do
     pwd
     echo "nifti_tool -strip_extras -overwrite -infiles ./${RUN}_${scenario}.rapidtide/${RUN}_${scenario}_${output_suffix}.nii.gz"
     nifti_tool -strip_extras -overwrite -infiles ./${RUN}_${scenario}.rapidtide/${RUN}_${scenario}_${output_suffix}.nii.gz
   done
done

# (1.3) Create extra-thresholded versions of outputs for lag maps
echo "++ INFO: (1.3) Create extra-thresholded versions of outputs for lag maps" 
echo "++ ====================================================================="
cd ./${RUN}_BASICnobpf.rapidtide/
for pval in 0p010 0p050 0p001 0p005
do
  3dcalc -overwrite                                                   \
          -a ${RUN}_BASICnobpf_lagtimes.nii.gz          \
          -b ${RUN}_BASICnobpf_p_lt_${pval}_mask.nii.gz \
          -expr 'a*b'                                                 \
          -prefix ${RUN}_BASICnobpf_lagtimes.masked_${pval}.nii.gz
  3dcalc -overwrite                                                  \
          -a ${RUN}_BASICnobpf_corrout.nii.gz           \
          -b ${RUN}_BASICnobpf_p_lt_${pval}_mask.nii.gz \
          -expr 'a*b'                                                 \
          -prefix ${RUN}_BASICnobpf_corrout.masked_${pval}.nii.gz
done
cd ..

# =============================================================================================== #
# ====================== PREPROCESSING PIPELINES THAT REQUIRE RAPIDTIDE OUTPUTS ================= #
# =============================================================================================== #

# (2) Behzadi CompCor Pipeline + Rapidtide Regressor
# ==================================================
echo "++ INFO: (2) Behzadi CompCorr Pre-processing + RapidTide Voxel-wise Regressor"
echo "++ =========================================================================="
3dTproject -overwrite                                           \
           -mask   ../ROI.FB.mPP.nii.gz                         \
           -input  ${RUN}_mPP.blur.scale.nii.gz                 \
           -polort ${POLORT}                                    \
           -ort    ${RUN}_Movement_Regressors_dt.discard10.txt  \
           -ort    ${RUN}_Bandpass_Regressors.txt               \
           -ort    ${RUN}_mPP_Behzadi_CompCorr_Regressors_vec.1D  \
           -dsort  ./${RUN}_BASICnobpf.rapidtide/${RUN}_BASICnobpf_lagregressor.nii.gz \
           -prefix ${RUN}_Behzadi_COMPCORpp.nii.gz
echo " + OUTPUT from Behzadi CompCor: ${RUN}_Behzadi_COMPCORpp.nii.gz"

# (2.1) Extract Representative Timseries for Behzadi_COMPCORpp
echo "++ INFO: (2.1) Extract Representative Timseries"
echo "==============================================="
for items in ${ROI_FV_PATH},'V4_grp' ${ROI_FVlt_PATH},'V4lt_grp' ${ROI_FVut_PATH},'V4ut_grp' ../ROI.V4_e.mPP.nii.gz,'V4_e' ../ROI.Vl_e.mPP.nii.gz,'Vl_e' ../ROI.GM.mPP.nii.gz,'GM' ../ROI.FB.mPP.nii.gz,'FB' ../ROI.WM_e.mPP.nii.gz,'WM_e'
do
   IFS="," ; set -- $items
   echo " + INFO: Mask = $1 | Suffix = $2 --> Output = ${RUN}_Behzadi_COMPCORpp.Signal.$2.1D"
   3dmaskave -quiet -mask $1 ${RUN}_Behzadi_COMPCORpp.nii.gz   > rm.aux.$2.1D
   3dDetrend -prefix - -polort ${POLORT} rm.aux.$2.1D\'  > rm.det.aux.$2.1D
   cat rm.det.aux.$2.1D | tr -s ' ' '\n' | sed '/^$/d' > ${RUN}_Behzadi_COMPCORpp.Signal.$2.1D
   rm rm.aux.$2.1D rm.det.aux.$2.1D
   1d_tool.py -overwrite -infile ${RUN}_Behzadi_COMPCORpp.Signal.$2.1D -derivative -write ${RUN}_Behzadi_COMPCORpp.Signal.$2.der.1D
done

# (2.2) Generate variance maps
echo "++ (2.2) INFO: Compute variance for output of COMPCORpp pipeline"
echo "================================================================"
3dTstat -overwrite -stdev -prefix ${RUN}_COMPCORpp.VAR.nii.gz ${RUN}_Behzadi_COMPCORpp.nii.gz
3dcalc  -overwrite -a ${RUN}_COMPCORpp.VAR.nii.gz -expr 'a*a' -prefix ${RUN}_COMPCORpp.VAR.nii.gz 

# (3) BASIC Pipeline + Rapidtide
# ==============================
echo "++ INFO: (3) Basic Pre-processing + RapidTide Voxel-wise Regressor"
echo "++ ==============================================================="
3dTproject -overwrite                                          \
           -mask   ../ROI.FB.mPP.nii.gz                        \
           -input  ${RUN}_mPP.blur.scale.nii.gz                \
           -polort ${POLORT}                                   \
           -ort    ${RUN}_Movement_Regressors_dt.discard10.txt \
           -ort    ${RUN}_Bandpass_Regressors.txt              \
           -dsort  ./${RUN}_BASICnobpf.rapidtide/${RUN}_BASICnobpf_lagregressor.nii.gz \
           -prefix ${RUN}_BASICpp.nii.gz
echo " + OUTPUT from Basic Pipeline: ${RUN}_BASICpp.nii.gz"

# (3.1) Extract Representative Timseries for BASICpp
echo "++ INFO: (3.1) Extract Representative Timseries"
echo "==============================================="
for items in ${ROI_FV_PATH},'V4_grp' ${ROI_FVlt_PATH},'V4lt_grp' ${ROI_FVut_PATH},'V4ut_grp' ../ROI.V4_e.mPP.nii.gz,'V4_e' ../ROI.Vl_e.mPP.nii.gz,'Vl_e' ../ROI.GM.mPP.nii.gz,'GM' ../ROI.FB.mPP.nii.gz,'FB' ../ROI.WM_e.mPP.nii.gz,'WM_e'
do
   IFS="," ; set -- $items
   echo " + INFO: Mask = $1 | Suffix = $2 --> Output = ${RUN}_BASICpp.Signal.$2.1D"
   3dmaskave -quiet -mask $1 ${RUN}_BASICpp.nii.gz   > rm.aux.$2.1D
   3dDetrend -prefix - -polort ${POLORT} rm.aux.$2.1D\'  > rm.det.aux.$2.1D
   cat rm.det.aux.$2.1D | tr -s ' ' '\n' | sed '/^$/d' > ${RUN}_BASICpp.Signal.$2.1D
   rm rm.aux.$2.1D rm.det.aux.$2.1D
   1d_tool.py -overwrite -infile ${RUN}_BASICpp.Signal.$2.1D -derivative -write ${RUN}_BASICpp.Signal.$2.der.1D
done

# (3.2) Generate variance maps
echo "++ INFO: (3.2) Compute variance for output of BASICpp pipeline"
echo "=============================================================="
3dTstat -overwrite -stdev -prefix ${RUN}_BASICpp.VAR.nii.gz ${RUN}_BASICpp.nii.gz
3dcalc  -overwrite -a ${RUN}_BASICpp.VAR.nii.gz -expr 'a*a' -prefix ${RUN}_BASICpp.VAR.nii.gz 


# (4) Create voxel-wise correlation maps between different pre-processed datasets
# ===============================================================================
echo "++ INFO: (4) Maps of correlation between the different pre-processing"
echo "++ ==================================================================="
3dTcorrelate -overwrite -prefix ${RUN}_BASICpp_2_Reference.nii.gz   ${RUN}_BASICpp.nii.gz ${RUN}_Reference.nii.gz
3dTcorrelate -overwrite -prefix ${RUN}_COMPCORpp_2_Reference.nii.gz ${RUN}_Behzadi_COMPCORpp.nii.gz ${RUN}_Reference.nii.gz

# (5) Compute the variance explained by lagged and single regressor
# =================================================================
export OMP_NUM_THREADS=16
echo "++ INFO: (5) Compute the variance explained by the voxel-wise lagged regressor"
echo "++ ==========================================================================="
for pipeline in Reference BASIC BASICpp Behzadi_COMPCOR Behzadi_COMPCORpp
do
    3dTcorrelate -overwrite -prefix ${RUN}_${pipeline}.R2_lagreg.nii.gz -pearson ${RUN}_${pipeline}.nii.gz ${RUN}_BASICnobpf.rapidtide/${RUN}_BASICnobpf_lagregressor.nii.gz
    3dcalc -overwrite -a            ${RUN}_${pipeline}.R2_lagreg.nii.gz -expr 'a*a' -prefix ${RUN}_${pipeline}.R2_lagreg.nii.gz

    1dBandpass 0.01 0.1 ${RUN}_mPP.Signal.V4lt_grp.1D > ${RUN}_mPP.det_bpf.Signal.V4lt_grp.1D
    3dTcorr1D -pearson -overwrite -mask ../ROI.FB.mPP.nii.gz -prefix ${RUN}_${pipeline}.R2_V4lt_grp.nii.gz ${RUN}_${pipeline}.nii.gz ${RUN}_mPP.det_bpf.Signal.V4lt_grp.1D
    3dcalc -overwrite -a       ${RUN}_${pipeline}.R2_V4lt_grp.nii.gz -expr 'a*a' -prefix ${RUN}_${pipeline}.R2_V4lt_grp.nii.gz
done
echo "=================================="
echo "++ INFO: Script finished correctly"
echo "=================================="
