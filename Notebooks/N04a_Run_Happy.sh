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
conda activate hcp7t_fv_sleep_rapidtide_keras_env

# Unset DISPLAY variable so that we don't get an error about access to XDisplay
echo "++ Unsetting the DISPLAY environment variable"
unset DISPLAY

cd ${DATA_DIR}/${SBJ}/${RUN}/
WORKDIR=`pwd`
echo "++ Working Directory: ${WORKDIR}"

echo "++ INFO: Create directory for happy results"
echo "++ ========================================"
if [ -d ${RUN}_orig.happy ]; then
     rm -rf ${RUN}_orig.happy
fi
mkdir ${RUN}_orig.happy

# Run rapidtide
# -------------
echo "++ INFO: Run Happy"
echo "++ ==============="
pwd

export OMP_NUM_THREADS=32

happy --temporalglm \
      ./${RUN}_orig.nii.gz \
      ../../ALL/ALL_orig.SliceTimings.info.forHAPPY \
      ./${RUN}_orig.happy
      
# Move outputs of happy to results folder
# =======================================
mv ./${RUN}_orig.happy_* ./${RUN}_orig.happy

# Organize Outputs
cd ${DATA_DIR}/${SBJ}/${RUN}/${RUN}_orig.happy
gunzip ${RUN}_orig.happy_*.tsv.gz

echo "=================================="
echo "++ INFO: Script finished correctly"
echo "=================================="