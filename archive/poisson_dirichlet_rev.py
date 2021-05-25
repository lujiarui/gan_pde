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
    'name': 'PoissonDirichlet',
    'dim': 2,
    'left boundary': 0,
    'right boundary': 1,
    'K_primal': 1,
    'K_adv': 1,
    'lr_primal': 0.015,
    'lr_adv': 0.04,
    'Nr': 10000,
    'Nb': 400,
    'alpha': None,
    'use elu': True,
    'n_iter': 100000,
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
    return -2.

def g(x):   
    """ 
    :g Dirichlet boundary / ground truth weak solution
    """
    return ((0.5 - torch.abs(x - 0.5))**2)[:, 0]

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
    _grad_u_theta = grad(_out_u_theta, xr, create_graph=True)[0]    # N x d
    _grad_2_u_theta = []
    for idx in range(xr.shape[0]):
        d2u_theta = grad(_grad_u_theta[idx][0], xr[idx][0], create_graph=True)[0]
        d2u_theta += grad(_grad_u_theta[idx][1], xr[idx][1], create_graph=True)[0]
        _grad_2_u_theta.append(d2u_theta)
    
    _grad_2_u_theta = torch.cat(_grad_2_u_theta, dim=0)
    #print(_grad_2_u_theta.shape)

    # feed forward
    _phi_eta = phi_eta(xr).squeeze()              # comp. graph => loss
    _u_theta_bdry = u_theta(xb).squeeze()        # comp. graph => loss
    
    #print(_phi_eta.shape)
    # >>> PDE-specific: calculate for I (integrand) >>>
    t1 = _grad_2_u_theta * _phi_eta    # norm, sum along dimension
    I = t1 - f(xr) * _phi_eta
    # <<< PDE-specific: calculate for I (integrand) <<<

    loss_int = 2 * torch.log(I.norm()) - torch.log( _phi_eta.norm()**2 )
    loss_bdry = (_u_theta_bdry - g(xb)).norm()**2  / xb.shape[0]
    
    return loss_int + loss_bdry * alpha
# <<< Numerical function definition <<<



if __name__ == '__main__':
    train(params=PARAMS, 
            g=g, 
            loss_func=loss_all, 
            device=DEVICE,
        #    valid=True,
        #    model_path='./poisson/WAN_PoissonDirichlet_2.pt'
        )

