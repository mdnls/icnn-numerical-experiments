import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .nets import IdealICNN, TParams, du_MLP
import torch

class DataFunc():
    def checkerboard(self, x1, x2, n=None):
        if(n is None):
            n = len(x1)
        thetas = np.arctan2(x2, x1)
        return np.cos(n/2*thetas)
    def cliff(self, x1, x2):
        thetas = np.arctan2(x2, x1)
        y = np.ones_like(thetas)
        y[thetas > 0] = -1
        return y
    def trench(self, x1, x2):
        thetas = np.arctan2(x2, x1)
        y = np.ones_like(thetas)
        y[thetas > 0] = -1
        y = y-2
        return y
    def gaussian_x(self, x1, x2):
        return np.exp(- x1**2)
    def quadratic(self, x1, x2, a=1, b=1, c=0):
        return a*x1**2 + b*x2**2 + 2*c*x1*x2
    
def make_data(data_dist, sampling, n, f=None):
    '''
    data_dist is a string in ["checkerboard", "cliff", "gaussian_x", "quadratic", func"]
    - Checkerboard: generate data x_i for which y_i is given by sin(|x|), oscillating up and down around the circle
    - Cliff: generate data x_i for which y_i is +-1 if x is below or above the origin resp.
    - Unit_Normal: generate data x_i for which y_i is e^{-(x_0)^2}
    - Func: generate data by applying fit_fn : R^2 -> R 
    
    sampling is a string in ["regular", "unif"]
    - Regular: sample n points in regular intervals around the perimeter of the circle
    - Uniform: sample n points uniform randomly around the permieter of the circle
    '''
    if(sampling == "regular"):
        thetas = np.linspace(0, 2*np.pi, n, endpoint=False)
    elif(sampling == "unif"):
        thetas = np.random.uniform(size=(n,), low=0, high=2*np.pi)
    else:
        raise ValueError("The requested sampling method was not recognized.")
        
    x = np.stack((np.cos(thetas), np.sin(thetas)), axis=1)
    if data_dist == "checkerboard":
        f = DataFunc().checkerboard
    elif data_dist == "cliff":
        f = DataFunc().cliff
    elif data_dist == "gaussian_x":
        f = DataFunc().gaussian_x
    elif data_dist == "quadratic":
        f = DataFunc().quadratic
    elif data_dist == "func":
        pass
    else:
        raise ValueError("The requested sampling distribution was not recognized.")
        
    if(f is not None):
        y = f(x[:, 0], x[:, 1]).reshape((-1, 1))
    else:
        raise ValueError("Tried to use data function but none was provided.")
    return x, y

def heatmap(data_fn, xrange=[-1.1, 1.1], dx=0.05):
    '''
    data_fn is a function with 2 arguments x1, x2 mapping a data vector (x1, x2) to a response y'''
    x1 = np.arange(*xrange, dx)
    x2 = np.arange(*xrange, dx)
    x_grid, y_grid = np.meshgrid(x1, x2)
    z = np.squeeze(data_fn(x_grid, y_grid))
    plt.contourf(x1, x2, z)
    plt.colorbar()
    
def apply_net(x1, x2, i):
    inp = torch.tensor(np.stack((x1, x2), axis=-1), dtype=torch.double, device="cpu")
    y = i.forward(inp)
    return y.detach().cpu().numpy()


def net_heatmap_comparison(net_arch, width, data_func, n_points, sampling_method, tparams=None):
    
    if(net_arch == "icnn"):
        i = IdealICNN(inp_dim=2, width=width, device="cpu")
        net_name = "ICNN"
    elif(net_arch == "mlp"):
        i = du_MLP(activ="relu", layers=(2, width, 1), scale=1/np.sqrt(width))
        net_name = "MLP"
    else: 
        raise ValueError("Network architecture not recognized.")
    
    if(tparams is None):
        tparams = TParams(lr=0.001, iters=2000)
    x, y = make_data("func", sampling_method, f=data_func, n=n_points)
    
    
    plt.subplot(1, 4, 1)
    heatmap(data_func)
    plt.scatter(*x.T, color="red", label="Data")
    plt.title("Ground Truth")
    plt.xlabel("$x_2$")
    plt.ylabel("$x_1$")
    plt.legend()

    plt.subplot(1, 4, 2)
    heatmap(lambda x1, x2: apply_net(x1, x2, i))
    plt.scatter(*x.T, color="red", label="Data")
    plt.title(f"{net_name} at Initialization")
    plt.xlabel("$x_2$")
    plt.ylabel("$x_1$")
    plt.legend()

    plt.subplot(1, 4, 3)
    losses = i.train(x, y, tparams)
    heatmap(lambda x1, x2: apply_net(x1, x2, i))
    plt.scatter(*x.T, color="red", label="Data")
    plt.title(f"{net_name} after Training")
    plt.xlabel("$x_2$")
    plt.ylabel("$x_1$")
    plt.legend()

    plt.subplot(1, 4, 4)
    plt.plot(losses)
    plt.ylim(0, np.max(losses))
    plt.ylabel("Training Loss")
    plt.xlabel("Iterations")
    plt.title("Training Loss")

    plt.gcf().set_size_inches(30, 5)
    plt.show()
    
    return (x, y, apply_net(*x.T, i), i)