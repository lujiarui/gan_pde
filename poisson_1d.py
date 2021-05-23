"""[4.2.1] Poisson equation only admits weak solution
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
    'name': 'Poisson_1d',
    'dim': 1,
    'left boundary': -1,
    'right boundary': 1,
    'K_primal': 1,
    'K_adv': 1,
    'lr_primal': 0.015,
    'lr_adv': 0.04,
    'Nr': 10000,
    'Nb': 400,
    'alpha': None,
    'use elu': True,
    'n_iter': 10000,
}
PARAMS['alpha'] = PARAMS['Nb'] * 10000
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# <<< global params definition <<<

torch.set_default_tensor_type(torch.DoubleTensor)
random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
np.random.seed(0)


# >>> Numerical function definition >>>
def f(x):
    """
    R.H.S of PDE
    """
    return torch.sin(x) + 4*torch.sin(4*x) - 8*torch.sin(8*x) + 16*torch.sin(24*x)

def g0(x):
    """
    :g_0 a component of grount truth solution
    """
    return torch.sin(x) + torch.sin(4*x)/4 - torch.sin(8*x)/8 + torch.sin(24*x)/36
    

def g(x):   
    """ 
    :g Dirichlet boundary / ground truth weak solution
    """
    _m1 = torch.Tensor([-1.])
    _1 = torch.Tensor([1.])
    _g0_m1 = g0(_m1)
    _g0_1 = g0(_1)
    c0 = -(_g0_m1 + _g0_1) / 2
    c1 = (_g0_m1 - _g0_1) / 2
    return g0(x) + c1 * x + c0

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
    
    # >>> PDE-specific: calculate for I (integrand) >>>
    t1 = torch.sum(_grad_u_theta * _grad_phi_eta , dim=1)    # norm, sum along dimension
    I = t1 - f(xr) * _phi_eta
    # <<< PDE-specific: calculate for I (integrand) <<<

    loss_int = 2 * torch.log(I.norm()) - torch.log( _phi_eta.norm()**2 )
    loss_bdry = (_u_theta_bdry - g(xb)).norm()**2  / xb.shape[0]
    
    return loss_int + loss_bdry * alpha
# <<< Numerical function definition <<<



if __name__ == '__main__':
    print('Use device: ', DEVICE)
    train(params=PARAMS, 
            g=g, 
            loss_func=loss_all, 
            device=DEVICE,
        #    valid=True,
            model_path='./WAN_Poisson_1d_2.pt'
        )

