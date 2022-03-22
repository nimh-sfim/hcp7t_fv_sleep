#!/bin/bash

# Author: Javier Gonzalez-Castillo
# Date last modification: March 6th, 2022

set -e

DATADIR=/data/SFIMJGC_HCP7T/HCP7T
ROI_PATH=`echo ${DATADIR}/ALL/ALL.CSFmask.FIXgrid.fs4vent.consensus.nii.gz`

echo "++ Enter the scan directory..."
cd ${DATADIR}/${SBJ}/${RUN}
WDIR=`pwd`
echo " + Working Directory              --> $WDIR"

# First we extract the portion of the time-series associated with this segment. 
# Becuase the 4th vent TS was obtained from the ${REGRESSION} (this is the same across all pre-processing cases)
TS_PATH=`echo rm.${TYPE}.${RUN}_${REGRESSION}.${UUID}.1D`
echo " + ROI Timeseries for the segment --> $TS_PATH"
echo "++ INFO: Extracting mean representative time series for the 4th ventricle for a given segment"
echo "         1d_tool.py -overwrite -infile ${RUN}_${REGRESSION}.Signal.${REGION}.1D -select_rows ${VOLS} -write ${TS_PATH}"
1d_tool.py -overwrite -infile ${RUN}_${REGRESSION}.Signal.${REGION}.1D -select_rows ${VOLS} -write ${TS_PATH}
echo "++ -----------------------------------------------------------------------------------------"

CM_PATH=`echo rm.${TYPE}.${RUN}_${REGRESSION}.${UUID}.cmap.nii`
CM_Z_PATH=`echo rm.${TYPE}.${RUN}_${REGRESSION}.${UUID}.Zcmap.nii`
echo " + CorrMap for the segment        --> $CM_PATH"
echo "++ INFO: Computing correlation map for the segment..."
echo " +       3dTcorr1D -overwrite -mask ${DATADIR}/${SBJ}/ROI.FB.mPP.nii.gz -prefix ${CM_PATH} ${RUN}_${REGRESSION}.nii.gz${VOLS} ${TS_PATH}"
3dTcorr1D -overwrite -mask ${DATADIR}/${SBJ}/ROI.FB.mPP.nii.gz -prefix ${CM_PATH} ${RUN}_${REGRESSION}.nii.gz${VOLS} ${TS_PATH}
 
echo "++ INFO: Convert to Z-scores via the Fisher Transformation"
echo " +       3dcalc -overwrite -a ${CM_PATH} -m ${DATADIR}/${SBJ}/ROI.FB.mPP.nii.gz -expr 'm*atanh(a)' -prefix ${CM_Z_PATH}"
3dcalc -overwrite -a ${CM_PATH} -m ${DATADIR}/${SBJ}/ROI.FB.mPP.nii.gz -expr 'm*atanh(a)' -prefix ${CM_Z_PATH}

# March 6th, 2022: May be able to delete this.
echo "++ INFO: Extracting average R per ROI (7 Networks)"
3dROIstats -nomeanout \
             -quiet \
             -nzmean \
             -mask ${DATADIR}/ALL/Schaefer2018_200Parcels_7Networks_order_mPP.GM_Ribbon.nii.gz \
             ${CM_Z_PATH} \
             > rm.${TYPE}.${RUN}_${REGRESSION}.${UUID}.cmap.Schaefer2018_7Nws_200.1D

echo "++ -----------------------------------------------------------------------------------------"
echo "++ ---------------------- Script finished correctly ----------------------------------------"
echo "++ -----------------------------------------------------------------------------------------"

rm ${TS_PATH}
