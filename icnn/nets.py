import torch
import torch.nn as nn
import numpy as np
import json
import matplotlib.pyplot as plt
from tqdm import tqdm, trange

class ICNN(nn.Module):
    def __init__(self, activ, layers):
        super(ICNN, self).__init__()
        weight_dims = list(zip(layers[1:], layers))
        self.As = nn.ParameterList()
        self.Ws = nn.ParameterList()
        self.bs = nn.ParameterList()
        first_idim = weight_dims[0][1]
        self.layers = layers
        self.activ_id = activ
        for odim, idim in weight_dims:
            self.As.append(nn.Parameter(torch.tensor(np.random.normal(size=(odim, first_idim)))))
            self.Ws.append(nn.Parameter(torch.tensor(np.random.uniform(size=(odim, idim), low=0, high=0.1))))
            self.bs.append(nn.Parameter(torch.tensor(np.random.normal(size=(odim,)))))
        if(activ == "relu"):
            self.activ = nn.ReLU()
        elif(activ == "linear"):
            self.activ = lambda x: x
        else:
            raise Exception("Unsupported activation function")
            
    def forward(self, z):
        z0 = z.clone()
        for (A, W, b) in zip(self.As, self.Ws, self.bs):
            z = self.activ(z0 @ torch.t(A) + z @ torch.t(W) + b)
        return z
    
    def loss(self, x, y):
        return torch.sum((self.forward(x) - y)**2)
    
    def project_ws(self):
        for w in self.Ws:
            w.data = torch.clamp(w.data, 0, np.inf)
            
    def train(self, fit_fn, a, b, dx, tparams):
        '''
        Train the net to minimize a least squares objective between itself and the given fit_fn on the interval [a, b] 
        by fitting regularly sampled points [a, a+dx, a+2dx, ..., b].
        
        fit_fn - (np.ndarray) -> (np.ndarray): map an input tensor to an output tensor
        a - (torch.double): beginning of interval
        b - (torch.double): end of interval
        dx - (torch.double): step size
        tparams - TParams: parameters to use for training
        '''
        self.tparams = tparams
        
        x = torch.tensor(np.arange(a, b, dx), dtype=torch.double).view([-1, 1])
        y = fit_fn(x)
        
        opt = torch.optim.Adam(params=self.parameters(), lr=tparams.lr, betas=(tparams.b1, tparams.b2))
        
        for itr in trange(tparams.iters):
            opt.zero_grad()
            loss_this_itr = self.loss(x, y)
            loss_this_itr.backward()
            opt.step()
            self.project_ws()
            
    def plot(self, fit_fn, a, b, dx):
        x = np.arange(a, b, dx)
        x_pt = torch.tensor(x, dtype=torch.double).view([-1, 1])
        y_true = fit_fn(x)
        y_est = self.forward(x_pt).cpu().detach().numpy()
        
        plt.plot(x, y_true, label="True")
        plt.plot(x, y_est, label="Estimate")
        
class MLP(ICNN):
    def __init__(self, activ, layers):
        super(activ, layers)
        weight_dims = list(zip(layers[1:], layers))
        first_idim = weight_dims[0][1]
        self.Ws = []
        for odim, idim in weight_dims:
            self.Ws.append(nn.Parameter(torch.tensor(np.random.normal(size=(odim, idim)))))

    def train(self, fit_fn, a, b, dx, tparams):
        '''
        Train the net to minimize a least squares objective between itself and the given fit_fn on the interval [a, b] 
        by fitting regularly sampled points [a, a+dx, a+2dx, ..., b].
        
        fit_fn - (np.ndarray) -> (np.ndarray): map an input tensor to an output tensor
        a - (torch.double): beginning of interval
        b - (torch.double): end of interval
        dx - (torch.double): step size
        tparams - TParams: parameters to use for training
        '''
        self.tparams = tparams
        
        x = torch.tensor(np.arange(a, b, dx), dtype=torch.double).view([-1, 1])
        y = fit_fn(x)
        
        opt = torch.optim.Adam(params=self.parameters(), lr=tparams.lr, betas=(tparams.b1, tparams.b2))
        
        for itr in trange(tparams.iters):
            opt.zero_grad()
            loss_this_itr = self.loss(x, y)
            loss_this_itr.backward()
            opt.step()

class TParams():
    def __init__(self, lr, b1, b2, iters):
        self.lr = lr
        self.b1 = b1
        self.b2 = b2
        self.iters = iters
    def __str__(self):
        return json.dumps({
            "lr": self.lr,
            "b1": self.b1,
            "b2": self.b2,
            "iters": self.iters
        })
