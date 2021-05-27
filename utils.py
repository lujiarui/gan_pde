import sys, os
from random import randint, sample
import json
from tqdm import tqdm
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import torch

def sample_xr_xb(Nr, Nb, dim, lbd, rbd):
    xr = float(rbd - lbd) * torch.rand(Nr, dim) + float(lbd)
    # Sample along the boundary
    xb = float(rbd - lbd) * torch.rand(Nb, dim) + float(lbd)
    for i in range(Nb):
        j = randint(0, dim - 1)
        val = float(rbd - lbd) * randint(0,1) + float(lbd)
        xb[i, j] = val
    return xr, xb

def sample_xr_xb_xa(Nr, Nb, Na, dim, lbd, rbd, t0, T, N):
    xr = float(rbd - lbd) * torch.rand(Nr, dim) + float(lbd)
    # Sample along the boundary
    xb = float(rbd - lbd) * torch.rand(Nb, dim) + float(lbd)
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

def sample_lin(low, high, dim):
    lin = torch.linspace(low, high)   # 100 pts by default
    if dim == 1:
        grid_size = 100
        x = torch.zeros(grid_size, 1)
        for i in range(100):
            x[i][0] = lin[i] 
    else:   # only show the first two dimension
        h_axis, v_axis = 100, 100
        x = torch.zeros(h_axis * v_axis, dim)
        for i in range(h_axis * v_axis):
            x[i][0] = lin[i%h_axis] 
            x[i][1] = lin[i//v_axis]
    return x

def plot_func(u_true, u_param, dim, low, high, dirs='.'):
    
    x = sample_lin(low, high, dim)
    u_param.eval()
    u_param.to(torch.device('cpu'))
    _u_true = u_true(x).squeeze()     # 10000 * 1
    _u_param = u_param(x).squeeze().detach()   # 10000 * 1
    dist = torch.abs(_u_true - _u_param)

    x = x.squeeze().numpy()
    
    if dim == 1:    # single-variable function plot
        fig, ax = plt.subplots()
        ax.set_xlim([low, high])
        # true function plot
        u_true_grid = _u_true.numpy()
        df_true = pd.DataFrame({'x':x, 'y':u_true_grid})
        # parameterized function plot
        u_param_grid = _u_param.numpy()
        df_param = pd.DataFrame({'x':x, 'y':u_param_grid})
        
        sns.lineplot(data=df_true, x='x', y='y', ax=ax, label='True value')
        sns.lineplot(data=df_param, x='x', y='y', ax=ax, label='DNN Fit value')
        fig.savefig(os.path.join(dirs, 'u_true_param.png'))

    else:    
        h_axis, v_axis = 100, 100
        ticks = torch.linspace(low, high).tolist()  # num = 100 by default
        xticks = [str(round(value, 2)) for value in ticks]
        yticks = [str(round(high+low-value, 2)) for value in ticks]
        
        grid = dist.view(h_axis, v_axis).numpy()
        data = pd.DataFrame(grid, columns=xticks, index=yticks)
        fig, ax = plt.subplots()
        sns.heatmap(data, ax=ax)
        # text
        plt.xlabel('x1')
        plt.ylabel('x2')
        fig.savefig(os.path.join(dirs,'diff.png'))


        grid = _u_true.view(h_axis, v_axis).numpy()
        data = pd.DataFrame(grid, columns=xticks, index=yticks)
        fig, ax = plt.subplots()
        sns.heatmap(data, ax=ax)
        # text
        plt.xlabel('x1')
        plt.ylabel('x2')
        fig.savefig(os.path.join(dirs,'u_true.png'))

        grid = _u_param.view(h_axis, v_axis).numpy()
        data = pd.DataFrame(grid, columns=xticks, index=yticks)
        fig, ax = plt.subplots()
        sns.heatmap(data, ax=ax)
        # text
        plt.xlabel('x1')
        plt.ylabel('x2')
        fig.savefig(os.path.join(dirs,'u_param.png'))

    


def plot_moving(data, desc, log_y=True):
    if log_y:
        plt.axes(yscale = "log")
    plt.plot(data)
    # text for figure
    plt.title('Moving {} each training episodes'.format(desc))
    plt.ylabel('Moving {}'.format(desc))
    plt.xlabel('Epoch')

    plt.savefig('moving_{}.png'.format(desc))
    plt.close()


