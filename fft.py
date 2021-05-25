import sys, os
from random import randint, sample
import json
import numpy as np
import pandas as pd
from scipy import signal

import matplotlib.pyplot as plt
import seaborn as sns
import torch

def calc_fft_peak(sig, clip=0.5):
    """Return the fft of given input signal.
    """
    # calc DFT for 'signal'
    fft_sig = np.fft.fft(sig)
    fft_sig = np.abs(fft_sig)
    # clip part of sequence
    clip_i = int(len(fft_sig)*0.5)
    fft_sig = fft_sig[:clip_i]
    # find peak frequency for FFT_signal, return idx
    peak_idx = signal.find_peaks(fft_sig)[0]
    print('Find peak idx: ', peak_idx)
    
    return fft_sig, peak_idx

def calc_fft_diff(sig, sig_ref):
    """Return the difference of both signal.
    """
    fft_ref, peak_idx = calc_fft_peak(sig_ref)
    fft_sig, _ = calc_fft_peak(sig)

    diff = np.abs(fft_sig[peak_idx] - fft_ref[peak_idx])
    return diff


# numerical functions 
def g0(x):
    """
    :g_0 a component of grount truth solution
    """
    return np.sin(x) + np.sin(4. * x) / 4. - np.sin(8. * x) / 8. + np.sin(24. * x) / 36.
    

def u(x):
    """ 
    :u Dirichlet boundary / ground truth weak solution
    """
    _g0_m1 = g0(-1)
    _g0_1 = g0(1)
    c0 = -(_g0_m1 + _g0_1) / 2.
    c1 = (_g0_m1 - _g0_1) / 2.
    return g0(x) + c1 * x + c0  # broadcast


def uu(x):
    """ 
    :u Dirichlet boundary / ground truth weak solution
    """
    _g0_m1 = g0(-1)
    _g0_1 = g0(1)
    c0 = -(_g0_m1 + _g0_1) / 2.
    c1 = (_g0_m1 - _g0_1) / 2.
    return g0(x)/2. + c1 * x + c0  # broadcast

def plot_line(data, peak_idx, save_to='example.png'):
    ax, fig = plt.subplots()
    plt.plot(data)
    # marker for peaks
    plt.scatter(peak_idx, data[peak_idx], color='red')
    plt.savefig(save_to)
    print('Write .png file to:  ', save_to)

def plot_both_line(data1, data2, save_to='fit_curve.png'):
    ax, fig = plt.subplots()
    plt.plot(data1)
    plt.plot(data2)
    plt.savefig(save_to)

if __name__ == "__main__":
    x = np.linspace(-1,1,100)
    ux =  u(x)
    uux = uu(x)
    #fft_ux, peak_idx = calc_fft_peak(ux)
    print(calc_fft_diff(ux, uux))
    plot_both_line(ux, uux)
    # marker for peaks
    #plot_line(fft_ux, peak_idx)

    # To be used
    # ax, fig = plt.subplots()
    # x = np.stack([[1,2,3], [1,2,3], [1,2,3]],axis=1)  # row: freq index, col: epoch
    # sns.heatmap(x, ax=ax)
    # plt.savefig('heatmap.png')
    