import torch
import torch.distributions as D
import math

def unif(x, y, a = -1, b = 1, device = 'cpu'):
    return (b-a)*torch.rand(x, y, device = device) + a

def gaus(x, y, device = 'cpu'):
    return torch.normal(0, math.sqrt(2), size = (x,y)).to(device)

def h_sphere(x, y, device = 'cpu'):
    xyz = torch.normal(0, 1, size = (x,y))
    return (xyz/xyz.norm(dim = 1).unsqueeze(1)).to(device)

def multinomial(x, y, device = 'cpu'):
    return torch.eye(y)[torch.randint(y,(x,))].to(device)

def generate_yale_condition(x, y, device = 'cpu'): 
    # For azimuth and elevation in eYaleFace data
    return torch.cat((multinomial(x, 28, device), multinomial(x, 10, device), unif(x, 1, -130/180, 130/180, device), unif(x, 1, -40/90, 1, device)), axis = 1)

def embedded_yale_condition(x, y, device = 'cpu'): 
    # For azimuth and elevation in eYaleFace data
    return torch.cat((gaus(x, y-2, device), unif(x, 1, -130/180, 130/180, device), unif(x, 1, -40/90, 1, device)), axis = 1)