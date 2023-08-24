import torch
from sklearn.datasets import make_s_curve

def s_curve_dataloader():
    s_curve, _ = make_s_curve(n_samples=10000, noise=0.1)
    s_curve = s_curve[:, [0,2]]/10.0  # shape of (10000, 2)
    
    dataset = torch.Tensor(s_curve).float()
    
    return torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)