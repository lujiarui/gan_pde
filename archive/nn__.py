"""Implementation follows the paper
u_theta network has 6 hidden layers, 40 neurons per hidden layer;
...
phi_eta network has 8 hidden layers, 50 neurons per hidden layer;
...

To use torch.sinc(x), torch version >= 1.8.1
"""

import torch
import torch.nn as nn

# >>> Primal Network >>>(checked)
class PrimalNet(nn.Module):
    def __init__(self, in_dim, hidden_dim=40, out_dim=1, use_elu=False):
        super().__init__()
        self.chain = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),  # 0
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),  # 1
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),  # 2
            nn.ELU() if use_elu else nn.Softplus(),
            nn.Linear(hidden_dim, hidden_dim),  # 3
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),  # 4
            nn.ELU() if use_elu else nn.Softplus(),
            nn.Linear(hidden_dim, hidden_dim),  # 5
            nn.Tanh(),
            nn.Linear(hidden_dim, out_dim),      # 6 (output)
        )
    def forward(self, x):
        x = self.chain(x)
        return x


# >>> Adversarial Network >>>
class AdverNet(nn.Module):
    def __init__(self, in_dim, hidden_dim=50, out_dim=1, n_hidden=8):
        super().__init__()
        self.in_layer = nn.Linear(in_dim, hidden_dim)
        self.linears = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim
            ) for i in range(n_hidden)])
        self.out_layer = nn.Linear(hidden_dim, out_dim)
        self.softplus = nn.Softplus()
        self.tanh = nn.Tanh()
    def forward(self, x):
        x = torch.tanh(self.in_layer(x))
        for i, lin in enumerate(self.linears):
            if i in [0]:
                x = lin(x)
                x = self.tanh(x)
            elif i in [1,3,5]:
                x = lin(x)
                x = self.softplus(x)
            elif i in [2,4,6]:
                x = torch.sinc(lin(x))  # torch >= 1.8.1
        x = self.out_layer(x)
        return x

if __name__ == '__main__':
    pn = PrimalNet(10)
    an = AdverNet(10)
    x  = torch.rand(10)
    print(pn(x))
    print(an(x))
    