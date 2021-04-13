from tqdm import tqdm
import torch
import torch.optim as optim

from utils import sample_xr_xb, sample_xr_xb_xa, plot_func, plot_moving
from nn import PrimalNet, AdverNet


def train(params, g, loss_func, device, log_every=100, requires_time=False):
    primal_net = PrimalNet(in_dim=params['dim'], use_elu=params['use elu']).to(device)
    adv_net = AdverNet(in_dim=params['dim']).to(device)
    primal_net.train()
    adv_net.train()

    opt_primal = optim.Adam(params=primal_net.parameters(), 
                                lr=params['lr_primal'])    # min
    opt_adv = optim.Adam(params=adv_net.parameters(), 
                                lr=params['lr_adv'])     # max
    opt_adv.param_groups[0]['lr'] *= -1.     # maxmize task: negative lr
    
    moving_loss = []
    moving_err  = []
    
    for step in tqdm(range(params['n_iter'])):
        # do something
        opt_primal.zero_grad()
        opt_adv.zero_grad()

        
        if requires_time:
            xr, xr_t0, xr_T, xb, xa = sample_xr_xb_xa(params['Nr'], params['Nb'], params['Na'], 
                                params['dim'], params['left boundary'], params['right boundary'],
                                params['t0'], params['T'], params['N'])
            xr = xr.to(device)
            xr_t0 = xr_t0.to(device)
            xr_T = xr_T.to(device)
            xb = xb.to(device)
            xa = xa.to(device)
            loss = loss_func(xr, xr_t0, xr_T, xb, xa, primal_net, adv_net, params['alpha'], params['gamma'], device)

        else:
            xr, xb = sample_xr_xb(params['Nr'], params['Nb'], params['dim'], 
                                params['left boundary'], params['right boundary'])
            xr = xr.to(device)
            xb = xb.to(device)
            loss = loss_func(xr, xb, primal_net, adv_net, params['alpha'], device)
            
        moving_loss.append(loss.item()) # log loss
        loss.backward()
        for _ in range(params['K_primal']):
            opt_primal.step()
        for _ in range(params['K_adv']):
            opt_adv.step()
        
        err = torch.sum(torch.abs(primal_net(xr).detach() - g(xr))) + \
                torch.sum(torch.abs(primal_net(xb).detach() - g(xb)))
        moving_err.append(err.item())
        
        if step % log_every == 0:
            print('\nTrain | GAN loss: {} | Error: {}\n'.format(loss.item(), err.item()))

    print('Training terminates normally.')
    
    plot_moving(moving_err, 'error')
    plot_moving(moving_loss, 'loss')

    torch.save({
            'primal net': primal_net.state_dict(),
            'adversial net': adv_net.state_dict(),
            },'WAN_{}_{}.pt'.format(params['name'], params['dim']))
    
    return