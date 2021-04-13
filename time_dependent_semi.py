"""[4.2.6-2] Diffusion-reaction PDEs
Implemented with Pytorch.(torch version >= 1.8.1)

* Variable interpretation:

- x: torch.Tensor, (Number of points, dimension)
- 
"""
import sys, os
from copy import deepcopy
import random
from random import randint
import json
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import grad
import matplotlib.pyplot as plt

from train import train

# >>> global params definition (same as 4.2.3)>>> 
PARAMS = {
    'name': 'ParabolicEqnInvTime',
    't0': 0,
    'T': 1.,
    'N': 10,
    'dim': 10,
    'left boundary': -1,
    'right boundary': 1,
    'K_primal': 2,
    'K_adv': 1,
    'lr_primal': 0.015,
    'lr_adv': 0.04,
    'Nr': None,
    'Nb': None,
    'Na': None,
    'alpha': None,
    'gamma': None,
    'use elu': False,
    'n_iter': 20000,
}
# update 
PARAMS['Nr'] = PARAMS['dim'] * 4000
PARAMS['Nb'] = PARAMS['dim'] * PARAMS['dim'] * 40
PARAMS['Na'] = PARAMS['Nb']
PARAMS['alpha'] = PARAMS['Nb'] * 10000
PARAMS['alpha'] = PARAMS['Nb'] * 10000

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# <<< global params definition <<<

torch.set_default_tensor_type(torch.DoubleTensor)
random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
np.random.seed(0)


# >>> Numerical function definition >>>
def h(x):
    """Initial constraint (also a coefficient in g)
    """
    return 2.0 * torch.sin(np.pi / 2.0 * x[:, 0]) * torch.cos(np.pi / 2.0 * x[:, 1])

def g(x, _h):   
    """ 
    :u* Dirichlet boundary / ground truth function
    -1 indicates the dimension of 'time'
    """
    return _h * torch.exp(-1. * x[:, -1])

def f(x, _g): 
    """
    R.H.S of PDE
    """
    return (np.pi * np.pi - 2.0) * _g / 2.0 - _g * _g / torch.cos(np.pi / 2.0 * x[:, 1])

def loss_all(xr: torch.Tensor, xr_t0: torch.Tensor, xr_T: torch.Tensor, xb: torch.Tensor, u_theta, phi_eta, alpha, gamma, device):
    """
    Args:
        torch.Tensor: (Nr x d)
        torch.Tensor: (Nb x d)
        Network instance
        Network instance
        alpha: weight constant
    Returns:
        torch.Tensor: (1 x 1)
    """
    # Calculate derivative w.r.t. to x
    xr = Variable(xr, requires_grad=True)
    xr_t0 = Variable(xr_t0, requires_grad=True)
    xr_T = Variable(xr_T, requires_grad=True)
    
    # Calculate derivative w.r.t. to x[x1, x2, ...]
    _out_u_theta = torch.sum(u_theta(xr))
    _grad_u_theta = grad(_out_u_theta, xr, create_graph=True)[0]

    _out_phi_eta = torch.sum(phi_eta(xr))
    _grad_phi_eta = grad(_out_phi_eta, xr, create_graph=True)[0]


    # feed forward
    _phi_eta = phi_eta(xr)              # comp. graph => loss
    _u_theta_bdry = u_theta(xb)        # comp. graph => loss

    # <<< PDE-specific: calculate for I <<<
    # do something
    # <<< PDE-specific: calculate for I <<<

    loss_int = 2 * torch.log(I_sum) - torch.log( torch.sum(_phi_eta * _phi_eta) )
    loss_bdry = torch.sum( (_u_theta_bdry - g(xb))**2 ) / xb.shape[0]
    
    return loss_int + loss_bdry * alpha + loss_init * gamma
# <<< Numerical function definition <<<


if __name__ == '__main__':
    train(params=PARAMS, 
            g=g, 
            loss_func=loss_all, 
            device=DEVICE,
            requires_time=True)

