# Enter scripts directory
echo "++ Entering Notebooks directory..."
cd /data/SFIMJGC_HCP7T/hcp7t_fv_sleep/Notebooks/

# Activate miniconda
echo "++ Activating miniconda"
. /data/SFIMJGC_HCP7T/Apps/miniconda38/etc/profile.d/conda.sh

# Activate vigilance environment
echo "++ Activating vigilance environment"
conda activate hcp7t_fv_sleep_env
python_path=`which python`
echo "++ --> Python path is ${python_path}"
# Unset DISPLAY variable so that we don't get an error about access to XDisplay
echo "++ Unsetting the DISPLAY environment variable"
unset DISPLAY

# Call python program
echo "++ Calling Python program: N10_mPP_Spectrogram.py -s ${SBJ} -d ${RUN} -r ${REGION} -w ${DATADIR}"
python ./N10_mPP_Spectrogram.py \
    -s ${SBJ} \
    -d ${RUN} \
    -r ${REGION} \
    -w ${DATADIR} 
