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
    'lr_primal': 0.0015,
    'lr_adv': 0.004,
    'Nr': 10000,  # 10000
    'Nb': 400,
    'alpha': None,
    'use elu': True,
    'n_iter': 3000,
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
    _u_theta = u_theta(xr).squeeze()
    _out_u_theta = torch.sum(_u_theta)
    _grad_u_theta = grad(_out_u_theta, xr, create_graph=True)[0]

    # feed forward
    _phi_eta = phi_eta(xr).squeeze()              # comp. graph => loss
    _u_theta_bdry = u_theta(xb).squeeze()        # comp. graph => loss

    _out_phi_eta = torch.sum(_phi_eta)
    _grad_phi_eta = grad(_out_phi_eta, xr, create_graph=True)[0]

    # >>> PDE-specific: calculate for I (integrand) >>>
    t1_list = []
    for i in range(xr.shape[1]):
        for j in range(xr.shape[1]):
            t1_list.append(_grad_u_theta[:, i] * _grad_phi_eta[:, j])

    I = sum(t1_list) - f(xr) * _phi_eta
    # <<< PDE-specific: calculate for I (integrand) <<<

    loss_int = 2 * (torch.log(I.norm()) - torch.log(_phi_eta.norm()) )
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

