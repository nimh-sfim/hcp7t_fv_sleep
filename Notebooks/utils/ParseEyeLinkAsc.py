# ParseEyeLinkAsc.py
# - Reads in .asc data files from EyeLink and produces pandas dataframes for further analysis
#
# Created 7/31/18-8/15/18 by DJ.
# Updated 7/4/19 by DJ - detects and handles monocular sample data.
#
# New note: 08/16/2021 by JGC
# This file has been modified by Javier Gonzalez-Castillo for the purpose of this particular project
# * Added support for non-verobose mode so that less informational messages are printed on the screen.
# * Uses python library re for working with regular expressions
# * Control for some additional error when finding empty files
#
# If interested in the original version of this library, please go to https://github.com/djangraw/ParseEyeLinkAscFiles

def ParseEyeLinkAsc(elFilename, verbose=False):
    # dfRec,dfMsg,dfFix,dfSacc,dfBlink,dfSamples = ParseEyeLinkAsc(elFilename)
    # -Reads in data files from EyeLink .asc file and produces readable dataframes for further analysis.
    #
    # INPUTS:
    # -elFilename is a string indicating an EyeLink data file from an AX-CPT task in the current path.
    #
    # OUTPUTS:
    # -dfRec contains information about recording periods (often trials)
    # -dfMsg contains information about messages (usually sent from stimulus software)
    # -dfFix contains information about fixations
    # -dfSacc contains information about saccades
    # -dfBlink contains information about blinks
    # -dfSamples contains information about individual samples
    #
    # Created 7/31/18-8/15/18 by DJ.
    # Updated 11/12/18 by DJ - switched from "trials" to "recording periods" for experiments with continuous recording
    
    # Import packages
    import numpy as np
    import pandas as pd
    import time
    import re

    # ===== READ IN FILES ===== #
    # Read in EyeLink file
    if verbose:
      print('Reading in EyeLink file %s...'%elFilename)
    t = time.time()
    f = open(elFilename,'r')
    fileTxt0 = f.read().splitlines(True) # split into lines
    #fileTxt0 = filter(None, fileTxt0) #  remove emptys, commented by JGC
    fileTxt0 = np.array(fileTxt0) # concert to np array for simpler indexing
    f.close()
    # END of JAVIER's modification
    if verbose:
      print('Done! Took %f seconds.'%(time.time()-t))

    # Separate lines into samples and messages
    if verbose:
      print('Sorting lines...')
    nLines = len(fileTxt0)
    lineType = np.array(['OTHER']*nLines,dtype='object')
    iStartRec = None
    t = time.time()
    for iLine in range(nLines):
        if len(fileTxt0[iLine])<3:
            lineType[iLine] = 'EMPTY'
        elif fileTxt0[iLine].startswith('*') or fileTxt0[iLine].startswith('>>>>>'):
            lineType[iLine] = 'COMMENT'
        elif re.match('[ \t]', fileTxt0[iLine]): #elif option added by JGC
            lineType[iLine] = 'MSG_EXTRA'
        elif fileTxt0[iLine].split()[0][0].isdigit() or fileTxt0[iLine].split()[0].startswith('-'):
            lineType[iLine] = 'SAMPLE'
        else:
            lineType[iLine] = fileTxt0[iLine].split()[0]
        if '!CAL' in fileTxt0[iLine]: # TODO: Find more general way of determining if recording has started
            iStartRec = iLine+1
    if verbose:
      print('Done! Took %f seconds.'%(time.time()-t))
    
    
    
    # ===== PARSE EYELINK FILE ===== #
    t = time.time()
    # Trials
    if verbose:
      print('Parsing recording markers...')
    iNotStart = np.nonzero(lineType!='START')[0]
    dfRecStart = pd.read_csv(elFilename,skiprows=iNotStart,header=None,delim_whitespace=True,usecols=[1])
    dfRecStart.columns = ['tStart']
    iNotEnd = np.nonzero(lineType!='END')[0]
    dfRecEnd = pd.read_csv(elFilename,skiprows=iNotEnd,header=None,delim_whitespace=True,usecols=[1,5,6])
    dfRecEnd.columns = ['tEnd','xRes','yRes']
    # combine trial info
    dfRec = pd.concat([dfRecStart,dfRecEnd],axis=1)
    nRec = dfRec.shape[0]
    if verbose:
      print('%d recording periods found.'%nRec)

    # Import Messages
    NumMSG = np.count_nonzero(lineType=='MSG')
    if NumMSG > 0:
        if verbose:
          print('Parsing stimulus messages... [%d messages]'% NumMSG)
        t = time.time()
        iMsg = np.nonzero(lineType=='MSG')[0]
        # set up
        tMsg = []
        txtMsg = []
        t = time.time()
        for i in range(len(iMsg)):
            # separate MSG prefix and timestamp from rest of message
            info = fileTxt0[iMsg[i]].split()
            # extract info
            tMsg.append(int(info[1]))
            txtMsg.append(' '.join(info[2:]))
        # Convert dict to dataframe
        dfMsg = pd.DataFrame({'time':tMsg, 'text':txtMsg})
        if verbose:
          print('Done! Took %f seconds.'%(time.time()-t))
    else:
        dfMsg = None
        if verbose:
          print('No messages available --> dfMsg = None')
    
    # Import Fixations
    NumFix = np.count_nonzero(lineType=='EFIX')
    if NumFix > 0:
        if verbose:
          print('Parsing fixations... [%d fixations]' % NumFix)
        t = time.time()
        iNotEfix = np.nonzero(lineType!='EFIX')[0]
        dfFix = pd.read_csv(elFilename,skiprows=iNotEfix,header=None,delim_whitespace=True,usecols=range(1,8))
        dfFix.columns = ['eye','tStart','tEnd','duration','xAvg','yAvg','pupilAvg']
        nFix = dfFix.shape[0]
        if verbose:
          print('Done! Took %f seconds.'%(time.time()-t))
    else:
        dfFix = None
        if verbose:
          print('No fixation information available --> dfFix = None')
    
    # Saccades
    NumSac = np.count_nonzero(lineType=='ESACC')
    if NumSac >0:
        if verbose:
          print('Parsing saccades... [%d saccades]' % NumSac)
        t = time.time()
        iNotEsacc = np.nonzero(lineType!='ESACC')[0]
        dfSacc = pd.read_csv(elFilename,skiprows=iNotEsacc,header=None,delim_whitespace=True,usecols=range(1,11))
        dfSacc.columns = ['eye','tStart','tEnd','duration','xStart','yStart','xEnd','yEnd','ampDeg','vPeak']
        if verbose:
          print('Done! Took %f seconds.'%(time.time()-t))
    else:
        dfSacc = None
        if verbose:
          print('No saccades information available --> dfSacc = None')
    
    # Blinks
    NumBlinks = np.count_nonzero(lineType=='EBLINK')
    if NumBlinks > 0:
        if verbose:
          print('Parsing blinks... [%d blinks]' % NumBlinks)
        iNotEblink = np.nonzero(lineType!='EBLINK')[0]
        dfBlink = pd.read_csv(elFilename,skiprows=iNotEblink,header=None,delim_whitespace=True,usecols=range(1,5))
        dfBlink.columns = ['eye','tStart','tEnd','duration']
        if verbose:
          print('Done! Took %f seconds.'%(time.time()-t))
    else:
        dfBlink = None
        if verbose:
          print('No blinks information available --> dfBlink = None')
    
    # determine sample columns based on eyes recorded in file
    if dfFix is None:
        eye = 'L'
        eyesInFile = 'L'
        if verbose:
          print('monocular data ASSUMED (%c eye).'%eye)
        cols = ['tSample', '%cX'%eye, '%cY'%eye, '%cPupil'%eye]
    else:
        eyesInFile = np.unique(dfFix.eye)
        if eyesInFile.size==2:
            if verbose:
              print('binocular data detected.')
            cols = ['tSample', 'LX', 'LY', 'LPupil', 'RX', 'RY', 'RPupil']
        else:
            eye = eyesInFile[0]
            if verbose:
              print('monocular data detected (%c eye).'%eye)
            cols = ['tSample', '%cX'%eye, '%cY'%eye, '%cPupil'%eye]
    # Import samples    
    if verbose:
      print('Parsing samples...')
    t = time.time()
    if iStartRec is None:
        iNotSample = np.nonzero( lineType!='SAMPLE')[0]
    else:
        iNotSample = np.nonzero( np.logical_or(lineType!='SAMPLE', np.arange(nLines)<iStartRec))[0]
    dfSamples = pd.read_csv(elFilename,skiprows=iNotSample,header=None,delim_whitespace=True,
                            usecols=range(0,len(cols)))
    dfSamples.columns = cols
    # Convert values to numbers
    for eye in ['L','R']:
        if eye in eyesInFile:
            dfSamples['%cX'%eye] = pd.to_numeric(dfSamples['%cX'%eye],errors='coerce')
            dfSamples['%cY'%eye] = pd.to_numeric(dfSamples['%cY'%eye],errors='coerce')
            dfSamples['%cPupil'%eye] = pd.to_numeric(dfSamples['%cPupil'%eye],errors='coerce')
        else:
            dfSamples['%cX'%eye] = np.nan
            dfSamples['%cY'%eye] = np.nan
            dfSamples['%cPupil'%eye] = np.nan
            
    if verbose:
      print('Done! Took %.1f seconds.'%(time.time()-t))
    
    # Return new compilation dataframe
    return dfRec,dfMsg,dfFix,dfSacc,dfBlink,dfSamples
