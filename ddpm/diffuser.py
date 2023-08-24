import torch
import torch.nn as nn

class MLPDiffuser(nn.Module):
    def __init__(self, n_steps, num_units=128):
        super(MLPDiffuser, self).__init__()
        
        self.linears = nn.ModuleList(
            [
                nn.Linear(2, num_units),  # input X, Y 2 channels (10000, 2)
                nn.ReLU(),
                nn.Linear(num_units, num_units),
                nn.ReLU(),
                nn.Linear(num_units, num_units),
                nn.ReLU(),
                nn.Linear(num_units, 2)
            ]
        )
        self.step_embeddings = nn.ModuleList(
            [
                nn.Embedding(n_steps, num_units),
                nn.Embedding(n_steps, num_units),
                nn.Embedding(n_steps, num_units),
            ]
        )
                
    def forward(self, x, t):
        
        # 这里相当于有三个 embedding 层去学 t 的位置信息，虽然似乎可以只用一个
        for i, layer in enumerate(self.step_embeddings):
            t_embeddings = layer(t)  # one-hot t embedding to num_units(embed_dim)
            x = self.linears[2*i](x)  # Linear Projection
            x += t_embeddings  # Simply Add
            x = self.linears[2*i+1](x)  # ReLU
            
        x = self.linears[-1](x)
        
        return x