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
            self.Ws.append(nn.Parameter(torch.tensor(np.random.uniform(size=(odim, idim), low=0, high=1), device=self.device)))
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
            z = self.activ(z0 @ torch.t(A) + z @ torch.t(W) + b)
        
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
        
        self.train(x, y, tparams, verbose)
            
    def train(self, x, y, tparams, verbose=True):
        x_pt = torch.tensor(x, dtype=torch.double, device=self.device)
        y_pt = torch.tensor(y, dtype=torch.double, device=self.device)
        
        if(tparams.b1 is not None):
            opt = torch.optim.Adam(params=self.parameters(), lr=tparams.lr, betas=(tparams.b1, tparams.b2))
        else: 
            opt = torch.optim.SGD(params=self.parameters(), lr=tparams.lr)
            
        if verbose:
            rng = trange(tparams.iters)
        else:
            rng = range(tparams.iters)
        
        losses = []
        for itr in rng:
            opt.zero_grad()
            loss_this_itr = self.loss(x_pt, y_pt)
            loss_this_itr.backward()
            losses.append(loss_this_itr.item())
            opt.step()
        return losses
            
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
        if(activ == "relu"):
            self.activ = nn.ReLU()
        elif(activ == "linear"):
            self.activ = lambda x: x
        else:
            raise Exception("Unsupported activation function")
            
    def forward(self, z):
        z0 = z.clone()
        
        for W in self.Ws[:-1]:
            z = self.activ(z @ torch.t(W))
        
        z = z @ torch.t(self.Ws[-1])
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
        self.train(x, y, tparams, verbose)
        
    def train(self, x, y, tparams, verbose=True):
        x_pt = torch.tensor(x, dtype=torch.double, device=self.device)
        y_pt = torch.tensor(y, dtype=torch.double, device=self.device)
        
        if verbose:
            rng = trange(tparams.iters)
        else:
            rng = range(tparams.iters)
            
        if(tparams.b1 is not None):
            opt = torch.optim.Adam(params=self.parameters(), lr=tparams.lr, betas=(tparams.b1, tparams.b2))
        else: 
            opt = torch.optim.SGD(params=self.parameters(), lr=tparams.lr)
            
        losses = []
        for itr in rng:
            opt.zero_grad()
            loss_this_itr = self.loss(x_pt, y_pt)
            loss_this_itr.backward()
            losses.append(loss_this_itr.item())
            opt.step()
        return losses
    
    def plot(self, fit_fn, a, b, dx):
        x = np.arange(a, b, dx)
        x_pt = torch.tensor(x, dtype=torch.double, device=self.device).view([-1, 1])
        y_true = fit_fn(x)
        y_est = self.forward(x_pt).cpu().detach().numpy()
        
        plt.plot(x, y_true, label="True")
        plt.plot(x, y_est, label="Estimate")

class IdealICNN(nn.Module):
    def __init__(self, inp_dim, width, device="cpu"):
        super(IdealICNN, self).__init__()
        layers = (inp_dim, width, 1)
        weight_dims = list(zip(layers[1:], layers))
        self.Ws = nn.ParameterList()
        self.width = width
        first_idim = weight_dims[0][1]
        self.layers = layers
        self.activ_id = "relu"
        self.activ = nn.ReLU()
        self.device = device
        for odim, idim in weight_dims:
            self.Ws.append(nn.Parameter(torch.tensor(np.random.normal(size=(odim, idim)), device=self.device)))
        self.A = nn.Parameter(torch.tensor(np.random.normal(size=(1, first_idim)), device=self.device))
            
    def forward(self, z):
        z0 = z.clone()
        
        for W in self.Ws[:-1]:
            z = self.activ(z @ torch.t(W))
        
        out_W = self.Ws[-1]
        z = (z0 @ torch.t(self.A)) + 1/np.sqrt(self.width) * torch.sum(z, dim=-1, keepdim=True) - np.sqrt(self.width / (2 * np.pi))
        return z

    def loss(self, x, y):
        return torch.sum((self.forward(x) - y)**2)
    
    def train(self, x, y, tparams, verbose=True):
        x_pt = torch.tensor(x, dtype=torch.double, device=self.device)
        y_pt = torch.tensor(y, dtype=torch.double, device=self.device)
        assert y.shape[-1] == 1, "Target output y must be a 1 dimensional scalar variable per input x, ie its shape should be (..., 1)"
        
        if verbose:
            rng = trange(tparams.iters)
        else:
            rng = range(tparams.iters)
        
        if(tparams.b1 is not None):
            opt = torch.optim.Adam(params=self.parameters(), lr=tparams.lr, betas=(tparams.b1, tparams.b2))
        else: 
            opt = torch.optim.SGD(params=self.parameters(), lr=tparams.lr)
            
        losses = []
        for itr in rng:
            opt.zero_grad()
            loss_this_itr = self.loss(x_pt, y_pt)
            losses.append(loss_this_itr.item())
            loss_this_itr.backward()
            opt.step()
        return losses
       

class TParams():
    def __init__(self, lr=0.001, b1=None, b2=None, iters=1000):
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
