import sys, os
from random import randint
import json
from tqdm import tqdm
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import torch

def sample_xr_xb(Nr, Nb, dim, lbd, rbd):
    xr = float(rbd - lbd) * torch.rand(Nr, dim) + float(lbd)
    # Sample along the boundary
    xb = float(rbd - lbd) * torch.rand(Nr, dim) + float(lbd)
    for i in range(Nb):
        j = randint(0, dim - 1)
        val = float(rbd - lbd) * randint(0,1) + float(lbd)
        xb[i, j] = val
    return xr, xb

def sample_xr_xb_xa(Nr, Nb, Na, dim, lbd, rbd, t0, T, N):
    xr = float(rbd - lbd) * torch.rand(Nr, dim) + float(lbd)
    # Sample along the boundary
    xb = float(rbd - lbd) * torch.rand(Nr, dim) + float(lbd)
    for i in range(Nb):
        j = randint(0, dim - 1)
        val = float(rbd - lbd) * randint(0,1) + float(lbd)
        xb[i, j] = val
    
    # extend another dimension: time
    tr_t0 = torch.ones(Nr,1) * t0
    tr_T = torch.ones(Nr,1) * T
    tb = torch.rand(Nb, 1)
    trange = torch.FloatTensor([t0 + float(T-t0) * i / N for i in range(N)])
    tr = torch.cat([trange for _ in range(Nr)])

    xr_t0 = torch.cat([xr, tr_t0], dim=1)
    xr_T = torch.cat([xr, tr_T], dim=1)
    xr = torch.cat([xr for _ in range(N)], dim=0)   # repeat for N times
    xr = torch.cat([xr, tr], dim=1)

    xb = torch.cat([xb,tb], dim=1)

    xa = float(rbd - lbd) * torch.rand(Na, dim+1) + float(lbd)
    xa[:, -1] = 0.

    return xr, xr_t0, xr_T, xb, xa

def plot_func(u_true, u_param, dim):
    u_param.eval()
    h_axis = 100
    v_axis = 100
    x = torch.zeros(h_axis * v_axis, dim)

    for i in range(h_axis):
        for j in range(v_axis):
            x[i * h_axis + j][0] = float(j) / v_axis    # x0 (h)
            x[i * h_axis + j][1] = 1. - float(i) / h_axis    # x1 (v)
    _u_true = u_true(x)     # 10000 * 1
    _u_param = u_param(x)   # 10000 * 1
    dist = torch.abs(_u_true - _u_param)
    grid = dist.view(h_axis, v_axis).numpy()
    ax = sns.heatmap(grid)
    fig = ax.get_figure()
    fig.savefig('diff.png')



def plot_moving(data, desc, log_y=True):
    if log_y:
        plt.axes(yscale = "log")
    plt.plot(data)
    plt.ylabel('Moving {} each training episodes'.format(desc))
    plt.xlabel('Iters')
    plt.savefig('moving_{}.png'.format(desc))
    plt.close()