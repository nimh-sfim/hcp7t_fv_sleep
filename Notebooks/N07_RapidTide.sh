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
# Run rapidtide
# -------------
echo "++ INFO: Run Rapidtide"
echo "++ ==================="
pwd
rapidtide2x_legacy  ./${RUN}_BASICnobpf.nii.gz \
           ./${RUN}_BASICnobpf.rapidtide/${RUN}_BASICnobpf \
           -F 0.01,0.1 --multiproc --nprocs=32 \
           -r -17,15 \
           --passes=1 -O 1 --regressor=${RUN}_mPP.Signal.V4_grp.1D

# Remove many ouputs of rapidtide that we do not need (for space)
# ===============================================================
echo "++ INFO: Remove unncessary outputs from RapidTide2"
echo "++ ==============================================="
for output_suffix in corrdistdata_pass1.txt datatoremove.nii.gz fitcoff.nii.gz fitNorm.nii.gz fitR2.nii.gz fitR.nii.gz formattedcommandline.txt globallaghist_pass1_centerofmass.txt globallaghist_pass1_peak.txt globallaghist_pass1.txt laghist_centerofmass.txt laghist.txt lagmask.nii.gz lagsigma.nii.gz mean.nii.gz memusage.csv MTT.nii.gz nullcorrelationhist_pass1_centerofmass.txt nullcorrelationhist_pass1_peak.txt nullcorrelationhist_pass1.txt options.txt p_lt_0p001_thresh.txt p_lt_0p005_thresh.txt p_lt_0p010_thresh.txt p_lt_0p050_thresh.txt referenceautocorr_pass1.txt reference_fmrires.txt reference_origres_prefilt.txt reference_origres.txt reference_resampres.txt Rhist_centerofmass.txt Rhist_peak.txt Rhist.txt runtimings.txt sigfit.txt strengthhist_centerofmass.txt strengthhist_peak.txt strengthhist.txt widthhist_centerofmass.txt widthhist_peak.txt widthhist.txt
do
   for scenario in BASICnobpf
   do
       if [ -e ./${RUN}_${scenario}.rapidtide/${RUN}_${scenario}_${output_suffix} ]; then 
            rm ./${RUN}_${scenario}.rapidtide/${RUN}_${scenario}_${output_suffix}; fi
   done
done

# Correct the headers for the remainder files
# ===========================================
echo "++ INFO: Correct the headers for the remainder files"
echo "++ ================================================="
for output_suffix in corrout filtereddata gaussout lagregressor lagstrengths lagtimes p_lt_0p001_mask p_lt_0p005_mask p_lt_0p010_mask p_lt_0p050_mask R2
do
  for scenario in BASICnobpf
   do
     pwd
     echo "nifti_tool -strip_extras -overwrite -infiles ./${RUN}_${scenario}.rapidtide/${RUN}_${scenario}_${output_suffix}.nii.gz"
     nifti_tool -strip_extras -overwrite -infiles ./${RUN}_${scenario}.rapidtide/${RUN}_${scenario}_${output_suffix}.nii.gz
   done
done

# Create extra-thresholded versions of outputs for lag maps
# =========================================================
echo "++ INFO: Create extra-thresholded versions of outputs for lag maps" 
echo "++ ==============================================================="
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

echo "=================================="
echo "++ INFO: Script finished correctly"
echo "=================================="
