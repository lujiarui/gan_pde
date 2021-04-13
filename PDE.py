from copy import deepcopy
import torch

class Poisson_Dirichlet:
    """4.2.1 Closure of Poisson equation with Dirichlet boundary.
    """
    def __init__(self):
        self.params = {
            'dim': 2,
            'left boundary': 0,
            'right boundary': 1,
            'K_primal': 2,
            'K_adv': 1,
            'lr_primal': 0.015,
            'lr_adv': 0.04,
            'Nr': 10000,
            'Nb': 400,
            'alpha': None,
            'use elu': True,
            'n_iter': 100000,
        }
        self.params['alpha'] = self.params['Nb'] * 10000
    def loss(self, xr: torch.Tensor, xb: torch.Tensor, u_th, phi_n):
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
        x_copy = deepcopy(xr).requires_grad_(True)
        x_copy = x_copy.to(device)
        _out_u_th = u_th(x_copy).squeeze()       # need val only, (N,1)
        _out_u_th.backward( torch.ones(xr.shape[0]).to(device) )   # each point contribute equally
        _grad_u_th = x_copy.grad
        _out_phi_n = phi_n(x_copy).squeeze()     # need val only, (N,1)
        _out_phi_n.backward( torch.ones(xr.shape[0]).to(device) )   # each point contribute equally
        _grad_phi_n = x_copy.grad
        # feed forward
        _u_th = u_th(xr)             # comp. graph => loss
        _phi_n = phi_n(xr)           # comp. graph => loss
        _u_th_bdry = u_th(xb)        # comp. graph => loss
        # >>> PDE-specific: calculate for I >>>
        t1 = torch.sum(_grad_u_th * _grad_phi_n , dim=1) # norm, sum along dimension
        I = t1 - f(xr) * _phi_n 
        # <<< PDE-specific: calculate for I <<<
        I_sum = torch.sum(I)    # sum along samples
        loss_int = 2 * torch.log(I_sum) - torch.log( torch.sum(_phi_n * _phi_n) )
        loss_bdry = torch.sum( (_u_th_bdry - g(xb))**2 ) / xb.shape[0]
        return loss_int + loss_bdry * self.params['alpha']
    
    def f(self, x):
    def g(self, x):



class Elliptic_Dirichlet:
    """4.2.3 Closure of Elliptic equation with Dirichlet boundary.
    """
    def __init__(self):
        self.params = {
            'dim': 10,
            'left boundary': -1,
            'right boundary': 1,
            'K_primal': 2,
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
        self.params['Nr'] = self.params['dim'] * 4000
        self.params['Nb'] = self.params['dim'] * self.params['dim'] * 40
        self.params['alpha'] = self.params['Nb'] * 10000
    def loss(self, xr: torch.Tensor, xb: torch.Tensor, u_th, phi_n):
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
        x_copy = deepcopy(xr).requires_grad_(True)
        x_copy = x_copy.to(device)
        
        _out_u_th = u_th(x_copy).squeeze()       # need val only, (N,1)
        _out_u_th.backward( torch.ones(xr.shape[0]).to(device) )   # each point contribute equally
        _grad_u_th = x_copy.grad
        
        _out_phi_n = phi_n(x_copy).squeeze()     # need val only, (N,1)
        _out_phi_n.backward( torch.ones(xr.shape[0]).to(device) )   # each point contribute equally
        _grad_phi_n = x_copy.grad
        
        # feed forward
        _u_th = u_th(xr)             # comp. graph => loss
        _phi_n = phi_n(xr)           # comp. graph => loss
        _u_th_bdry = u_th(xb)        # comp. graph => loss
        
        # >>> PDE-specific: calculate for I >>>
        t1 = torch.sum(_grad_u_th * _grad_phi_n , dim=1) * a(xr)  # norm, sum along dimension
        t2 = torch.sum(_grad_u_th * _grad_u_th, dim=1) / 2.0
        I = (t2 - f(xr)) * _phi_n - t1
        # <<< PDE-specific: calculate for I <<<

        I_sum = torch.sum(I)    # sum along samples
        loss_int = 2 * torch.log(I_sum) - torch.log( torch.sum(_phi_n * _phi_n) )
        loss_bdry = torch.sum( (_u_th_bdry - g(xb))**2 ) / xb.shape[0]
        
        return loss_int + loss_bdry * self.params['alpha']

