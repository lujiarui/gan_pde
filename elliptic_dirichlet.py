"""[4.2.3] High-dimensional nonlinear elliptic PDEs 
Implemented with Pytorch. (torch version >= 1.8.1)

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
from torch.autograd import grad, Variable
import matplotlib.pyplot as plt

from train import train

# >>> global params definition >>> 
PARAMS = {
    'name': 'EllipticDirichlet',
    'dim': 25,
    'left boundary': -1,
    'right boundary': 1,
    'K_primal': 1,
    'K_adv': 1,
    'lr_primal': 0.015,
    'lr_adv': 0.04,
    'Nr': None,
    'Nb': None,
    'alpha': None,
    'use elu': False,
    'n_iter': 20000,
}
# update 
PARAMS['Nr'] = PARAMS['dim'] * 4000
PARAMS['Nb'] = PARAMS['dim'] * PARAMS['dim'] * 40
PARAMS['alpha'] = PARAMS['Nb'] * 10000

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# <<< global params definition <<<

torch.set_default_tensor_type(torch.DoubleTensor)
random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
np.random.seed(0)


# >>> Numerical function definition >>>
def a(x: torch.Tensor):
    """
    :1+|x|^2 by definition
    Sum along the dimension axis
    """
    a0 = x**2
    return 1 + torch.sum(a0, dim=1)

def p0(x: torch.Tensor):
    """
    :\rho_0^2 by definition
    Only the first two dimensions contributes
    """
    return np.pi * x[:, 0]**2 + x[:, 1]**2  / 2.0

def p1(x: torch.Tensor):
    """
    :\rho_1^2 by definition
    Only the first two dimensions contributes
    """
    return ((np.pi)**2 * x[:, 0]**2 + x[:, 1]**2)  / 4.0

def f(x):
    """
    R.H.S of PDE
    """
    _p0 = p0(x)
    _p1 = p1(x)
    _a = a(x)
    _cos = torch.cos(_p0)
    return 4.0 * _p1 * _a * torch.sin(_p0) - \
            (4.0 * _p0 + (np.pi + 1) * _a) * _cos + \
                2.0 * _p1 * _cos**2

def g(x):   
    """ 
    :u* Dirichlet boundary / ground truth function
    """
    return torch.sin( (np.pi * x[:, 0]**2 + x[:, 1]**2) / 2.0 )

def loss_all(xr: torch.Tensor, xb: torch.Tensor, u_theta, phi_eta, alpha, device):
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
    
    # Calculate derivative w.r.t. to x[x1, x2, ...]
    _out_u_theta = torch.sum(u_theta(xr).squeeze())
    _grad_u_theta = grad(_out_u_theta, xr, create_graph=True)[0]

    _out_phi_eta = torch.sum(phi_eta(xr).squeeze())
    _grad_phi_eta = grad(_out_phi_eta, xr, create_graph=True)[0]

    # feed forward
    _phi_eta = phi_eta(xr).squeeze()              # comp. graph => loss
    _u_theta_bdry = u_theta(xb).squeeze()        # comp. graph => loss
    
    # >>> PDE-specific: calculate for I >>>
    t1 = torch.sum(_grad_u_theta * _grad_phi_eta , dim=1) * a(xr)  # norm, sum along dimension
    t2 = torch.sum(_grad_u_theta * _grad_u_theta, dim=1) / 2.0
    I = (t2 - f(xr)) * _phi_eta - t1
    # <<< PDE-specific: calculate for I <<<

    loss_int = 2 * torch.log(I.norm()) - torch.log( _phi_eta.norm()**2 )
    loss_bdry = (_u_theta_bdry - g(xb)).norm()**2  / xb.shape[0]
    
    return loss_int + loss_bdry * alpha
# <<< Numerical function definition <<<


if __name__ == '__main__':
    train(params=PARAMS, 
            g=g, 
            loss_func=loss_all, 
            device=DEVICE)

