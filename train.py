import os
from tqdm import tqdm
import torch
import torch.optim as optim

from utils import sample_xr_xb, sample_xr_xb_xa, sample_lin, plot_func, plot_moving
from nn import PrimalNet, AdverNet
import fft_utils


def train(params, g, loss_func, device, log_every=100, requires_time=False, valid=False, model_path=None):
    if valid:
        primal_net = PrimalNet(in_dim=params['dim'], use_elu=params['use elu'])
        primal_net.load_state_dict(torch.load(model_path)['primal net'])
        plot_func(g, primal_net, 2, params['left boundary'], params['right boundary'])
        return 
    # pretrained
    if model_path:
        print('loaded model from: ', model_path)
        primal_net = PrimalNet(in_dim=params['dim'], use_elu=params['use elu'])
        primal_net.load_state_dict(torch.load(model_path)['primal net'])
        adv_net = PrimalNet(in_dim=params['dim'], use_elu=params['use elu'])
        adv_net.load_state_dict(torch.load(model_path)['adversial net'])
    else:
        primal_net = PrimalNet(in_dim=params['dim'], use_elu=params['use elu']).to(device)
        adv_net = AdverNet(in_dim=params['dim']).to(device)
    
    primal_net.train()
    adv_net.train()

    opt_primal = optim.Adam(params=primal_net.parameters(), lr=params['lr_primal'])    # min
    opt_adv = optim.Adam(params=adv_net.parameters(), lr=params['lr_adv'])     # max
    opt_adv.param_groups[0]['lr'] *= -1.     # maxmize task: negative lr
    
    moving_loss = []
    moving_err  = []
    freq_diff = []
    peak_index = [1,5,6][::-1]      # because heatmap arrange vertical-axis by top-down, alter it into bottom-up
    test_points = sample_lin(params['left boundary'], params['right boundary'], params['dim']).to(device)   # 100 * 100 (x[0] local shift)

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
        approx_y = primal_net(test_points).detach().squeeze()
        true_y = g(test_points)
        # calculate relative error
        err_l2 = torch.sqrt(torch.mean(torch.square(
                approx_y - true_y
                )))
        gt_l2 = torch.sqrt(torch.mean(torch.square(
                true_y
                )))  
        rel_err = err_l2 / gt_l2
        moving_err.append(rel_err.item())
        
        # save current fft fitting condition
        approx_y = approx_y.to('cpu').numpy()[5000:5100]    # center of y-axis, cross-section
        true_y = true_y.to('cpu').numpy()[5000:5100]
        diff_tmp = fft_utils.calc_fft_diff(approx_y, true_y, peak_index)
        freq_diff.append(diff_tmp)

        # logger
        if step % log_every == 0:
            print('\nTrain | GAN loss: {} | Relative Error: {} \n'.format(loss.item(), rel_err.item()))
            if len(freq_diff) > 100:
                fft_utils.plot_spectrum(freq_diff, peak_index)

    print('Training terminates normally.')
    
    plot_moving(moving_err, 'error')
    plot_moving(moving_loss, 'loss')
    plot_func(g, primal_net, params['dim'], params['left boundary'], params['right boundary'])
    fft_utils.plot_spectrum(freq_diff, peak_index)

    torch.save({
            'primal net': primal_net.state_dict(),
            'adversial net': adv_net.state_dict(),
            'params': params,
            },'WAN_{}_{}.pt'.format(params['name'], params['dim']))
    return