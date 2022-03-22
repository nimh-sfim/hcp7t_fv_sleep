from scipy.signal import get_window, spectrogram
from .basics import get_window_index
import pandas as pd
from matplotlib.cm import get_cmap
from matplotlib.colors import rgb2hex

def generate_cmap_hex(cmap_name,num_colors):
    cmap     = get_cmap(cmap_name,num_colors)
    cmap_hex = []
    for i in range(cmap.N):
        rgba = cmap(i)
        cmap_hex.append(rgb2hex(rgba))
    return cmap_hex
   
def get_spectrogram(ts):
    WIN_LENGTH, WIN_OVERLAP, NFFT, SCALING, DETREND, FS = 60,59, 128, 'density', 'constant', 1
    win_idx        = get_window_index(nacqs=ts.shape[0],tr=int(1/FS), win_dur=WIN_LENGTH)
    f,t,Sxx        = spectrogram(ts,FS,window=get_window(('tukey',0.25),WIN_LENGTH), noverlap=WIN_OVERLAP, scaling=SCALING, nfft=NFFT, detrend=DETREND, mode='psd')
    spectrogram_df = pd.DataFrame(Sxx,index=f,columns=win_idx)
    return spectrogram_df
   
