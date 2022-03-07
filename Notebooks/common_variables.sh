# Please configure these variables accorinding to your wokring environment
export DATA_DIR=/data/SFIMJGC_HCP7T/HCP7T/
export SCRIPTS_DIR=/data/SFIMJGC_HCP7T/hcp7t_fv_sleep/
export ATLAS_DIR=/data/SFIMJGC_HCP7T/HCP7T/Atlases/


# Do not touch these variables if trying to reproduce original results
export ATLAS_NAME=Schaefer2018_200Parcels_7Networks
export POLORT=5
export VOLS_DISCARD=10
export BLUR_FWHM=4
export ROI_FV_PATH=`echo ${DATA_DIR}/ALL/ALL_ROI.V4.mPP.nii.gz`
export ROI_FVlt_PATH=`echo ${DATA_DIR}/ALL/ALL_ROI.V4lt.mPP.nii.gz`
export ROI_FVut_PATH=`echo ${DATA_DIR}/ALL/ALL_ROI.V4ut.mPP.nii.gz`

