import torch
import torch.nn as nn
import numpy as np
import json
import matplotlib.pyplot as plt
from tqdm import tqdm, trange

class ICNN(nn.Module):
    def __init__(self, activ, layers, scale=None, device="cpu"):
        super(ICNN, self).__init__()
        weight_dims = list(zip(layers[1:], layers))
        self.As = nn.ParameterList()
        self.Ws = nn.ParameterList()
        self.bs = nn.ParameterList()
        self.scale = scale
        first_idim = weight_dims[0][1]
        self.layers = layers
        self.activ_id = activ
        self.device = device
        for odim, idim in weight_dims:
            self.As.append(nn.Parameter(torch.tensor(np.random.normal(size=(odim, first_idim)), device=self.device)))
            self.Ws.append(nn.Parameter(torch.tensor(np.random.uniform(size=(odim, idim), low=0, high=10), device=self.device)))
            self.bs.append(nn.Parameter(torch.tensor(np.random.normal(size=(odim,)), device=self.device)))
        if(activ == "relu"):
            self.activ = nn.ReLU()
        elif(activ == "linear"):
            self.activ = lambda x: x
        else:
            raise Exception("Unsupported activation function")
            
    def forward(self, z):
        z0 = z.clone()
        layers = list(zip(self.As, self.Ws, self.bs))
        
        for (A, W, b) in layers[:-1]:
            z = self.activ(z0 @ torch.t(A) + b) # + z @ torch.t(W) + b)
        
        out_A, out_W, out_b = layers[-1]
        z = z0 @ torch.t(out_A) + z @ torch.t(out_W) + out_b
        if(self.scale is not None):
            return z * self.scale
        else:
            return z
    
    def loss(self, x, y):
        return torch.sum((self.forward(x) - y)**2)
    
    def project_ws(self):
        for w in self.Ws:
            w.data = torch.clamp(w.data, 0, np.inf)
            
    def train(self, fit_fn, a, b, dx, tparams, verbose=True):
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
        
        x = torch.tensor(np.arange(a, b, dx), dtype=torch.double, device=self.device).view([-1, 1])
        y = fit_fn(x)
        
        opt = torch.optim.Adam(params=self.parameters(), lr=tparams.lr, betas=(tparams.b1, tparams.b2))

        if verbose:
            rng = trange(tparams.iters)
        else:
            rng = range(tparams.iters)

        for itr in rng:
            opt.zero_grad()
            loss_this_itr = self.loss(x, y)
            loss_this_itr.backward()
            opt.step()
            self.project_ws()
            
    def plot(self, fit_fn, a, b, dx):
        x = np.arange(a, b, dx)
        x_pt = torch.tensor(x, dtype=torch.double, device=self.device).view([-1, 1])
        y_true = fit_fn(x)
        y_est = self.forward(x_pt).cpu().detach().numpy()
        
        plt.plot(x, y_true, label="True")
        plt.plot(x, y_est, label="Estimate")
        
class MLP(ICNN):
    def __init__(self, activ, layers, scale=None, device="cpu"):
        super(MLP, self).__init__(activ, layers, scale=scale, device=device)
        weight_dims = list(zip(layers[1:], layers))
        first_idim = weight_dims[0][1]
        self.Ws = nn.ParameterList()
        for odim, idim in weight_dims:
            self.Ws.append(nn.Parameter(torch.tensor(np.random.normal(size=(odim, idim)), device=self.device)))

    def train(self, fit_fn, a, b, dx, tparams, verbose=True):
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
        
        x = torch.tensor(np.arange(a, b, dx), dtype=torch.double, device=self.device).view([-1, 1])
        y = fit_fn(x)
        
        opt = torch.optim.Adam(params=self.parameters(), lr=tparams.lr, betas=(tparams.b1, tparams.b2))
        
        if verbose:
            rng = trange(tparams.iters)
        else:
            rng = range(tparams.iters)

        for itr in rng:
            opt.zero_grad()
            loss_this_itr = self.loss(x, y)
            loss_this_itr.backward()
            opt.step()

class du_MLP(nn.Module):
    def __init__(self, activ, layers, scale=None, device="cpu"):
        super(du_MLP, self).__init__()
        weight_dims = list(zip(layers[1:], layers))
        self.Ws = nn.ParameterList()
        self.bs = nn.ParameterList()
        self.scale = scale
        first_idim = weight_dims[0][1]
        self.layers = layers
        self.activ_id = activ
        self.device = device
        for odim, idim in weight_dims:
            self.Ws.append(nn.Parameter(torch.tensor(np.random.normal(size=(odim, idim)), device=self.device)))
            self.bs.append(nn.Parameter(torch.tensor(np.random.normal(size=(odim,)), device=self.device)))
        if(activ == "relu"):
            self.activ = nn.ReLU()
        elif(activ == "linear"):
            self.activ = lambda x: x
        else:
            raise Exception("Unsupported activation function")
            
    def forward(self, z):
        z0 = z.clone()
        
        layers = list(zip(self.Ws, self.bs))
        final_W, final_b = layers[-1]
        for (W, b) in layers[:-1]:
            z = self.activ(z @ torch.t(W) + b)
        
        z = z @ torch.t(final_W)
        if(self.scale is not None):
            return z * self.scale
        else:
            return z
    
    def loss(self, x, y):
        return torch.sum((self.forward(x) - y)**2)
    
    def train(self, fit_fn, a, b, dx, tparams, verbose=True):
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
        
        x = torch.tensor(np.arange(a, b, dx), dtype=torch.double, device=self.device).view([-1, 1])
        y = fit_fn(x)
        
        opt = torch.optim.Adam(params=self.parameters(), lr=tparams.lr, betas=(tparams.b1, tparams.b2))

        if verbose:
            rng = trange(tparams.iters)
        else:
            rng = range(tparams.iters)

        for itr in rng:
            opt.zero_grad()
            loss_this_itr = self.loss(x, y)
            loss_this_itr.backward()
            opt.step()
            
    def plot(self, fit_fn, a, b, dx):
        x = np.arange(a, b, dx)
        x_pt = torch.tensor(x, dtype=torch.double, device=self.device).view([-1, 1])
        y_true = fit_fn(x)
        y_est = self.forward(x_pt).cpu().detach().numpy()
        
        plt.plot(x, y_true, label="True")
        plt.plot(x, y_est, label="Estimate")


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
