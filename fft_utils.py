import sys, os
from random import randint, sample
import json
import numpy as np
import pandas as pd
from scipy import signal

import matplotlib.pyplot as plt
import seaborn as sns
import torch

# >>> fft related functions >>>
def calc_fft_peak(sig, clip_rate=0.5):
    """Return the fft of given input signal.
    """
    # calc DFT for 'signal'
    fft_sig = np.fft.fft(sig)
    #fft_sig = np.abs(fft_sig)  # norm
    
    # clip part of sequence
    clip_i = int(len(fft_sig)*clip_rate)
    fft_sig = fft_sig[:clip_i]
    # find peak frequency for FFT_signal, return idx
    peak_idx = signal.find_peaks(np.abs(fft_sig))[0]
    #print('Find peak idx: ', peak_idx)
    
    return fft_sig, peak_idx

def calc_fft_diff(sig, sig_ref, peak_idx=None):
    """Return the relative difference of both signal.
    """
    #print(sig, sig_ref) # debug
    fft_ref, _peak_idx = calc_fft_peak(sig_ref)
    fft_sig, _ = calc_fft_peak(sig)
    if peak_idx is None:
        peak_idx = _peak_idx[:3]

    diff = np.abs( (fft_sig[peak_idx] - fft_ref[peak_idx]) / np.abs(fft_ref[peak_idx]))
    return diff
# <<< fft related functions


# >>> numerical functions >>>
def g(x):   
    """ 
    :g Dirichlet boundary / ground truth weak solution
    """
    return ((0.5 - torch.abs(x - 0.5))**2)[:, 0]

def g_ecl(x):   
    """ 
    :u* Dirichlet boundary / ground truth function; consider only one direction: x_0 (approximately isotropy)
    """
    return torch.sin( (np.pi * x[:, 0]**2) / 2.0 )
# <<< numerical functions <<<

# >>> plot functions >>>
def plot_line(data, peak_idx, save_to='example.png'):
    ax, fig = plt.subplots()
    plt.plot(data)
    # marker for peaks
    plt.scatter(peak_idx, data[peak_idx], color='red')
    # text
    plt.xlabel('Frequnecy Index')
    plt.ylabel('Intensity')
    plt.savefig(save_to)
    print('Write .png file to:  ', save_to)

def plot_both_line(data1, data2, save_to='fit_curve.png'):
    ax, fig = plt.subplots()
    plt.plot(data1)
    plt.plot(data2)
    plt.savefig(save_to)

def plot_spectrum(data_list, peak_index):
    vticks = peak_index             
    ax, fig = plt.subplots()
    x = np.stack(data_list, axis=1)  # row: freq index, col: epoch
    df_x = pd.DataFrame(x, index=vticks)
    ax = sns.heatmap(df_x, cmap='icefire')
    plt.xlabel('Epoch')
    plt.ylabel('Frequency Index')

    plt.savefig('heatmap.png')
    plt.close()
# <<< plot functions <<< 

if __name__ == "__main__":
    x = torch.Tensor(np.linspace(-1,1,100).reshape(100,1))
    gx =  g_ecl(x)
    #uux = uu(x)
    fft_ux, peak_idx = calc_fft_peak(gx, 0.3)
    fft_ux = np.abs(fft_ux)
    print(fft_ux[[6,5,1]])
     # marker for peaks
    plot_line(fft_ux, peak_idx=[6,5,1])
    
    #print(calc_fft_diff(ux, uux))
    #plot_both_line(ux, uux)
    #np.fft.fft2()
   

    # To be used
    #ax, fig = plt.subplots()
    #x = [np.linspace(0,1,100), 
    #            np.linspace(1,2,100), 
    #            np.linspace(2,3,100)]

    #plot_spectrum(x)
    