import torch

import matplotlib.pyplot as plt

from .diffuser import *
from .my_dataloader import *

class MLPDiffuserTrainer:
    def __init__(self, 
                 num_steps: int = 100, 
                 num_epoch: int = 4000, 
                 gpu: bool = True):
        # Basic Config
        self.num_steps = num_steps  # T
        self.num_epoch = num_epoch
        self.shape = (10000, 2) 

        # Key Parameters
        betas = torch.linspace(-6, 6, self.num_steps)  # evenly spaced
        self.betas = torch.sigmoid(betas) * (0.5e-2 - 1e-5) + 1e-5  # 1e-5 to 0.5e-2 where sigmoid opt 0-1

        self.alphas = 1 - self.betas
        self.alphas_prod = torch.cumprod(self.alphas, dim=0)  # cumulative product
        self.alphas_bar_sqrt = torch.sqrt(self.alphas_prod)  # sqrt cumulative product
        self.one_minus_alphas_bar_sqrt = torch.sqrt(1 - self.alphas_prod)

        # DataLoader
        self.dataloader = s_curve_dataloader()

        # Model
        self.model = MLPDiffuser(self.num_steps)

        # To GPU Device
        self.gpu = gpu
        if self.gpu:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model.to(self.device)

    # Given x_0, t and get x_t using reparameterization trick
    def q_x(self, x_0, t):
        noise = torch.randn_like(x_0)  # get a noise from N(0,1)
        alpha_bar_t = self.alphas_bar_sqrt[t]
        alpha_1_m_t = self.one_minus_alphas_bar_sqrt[t]
        
        return (alpha_bar_t * x_0) + (alpha_1_m_t * noise)  # mu + sigma * epsilon

    # Calculate the loss
    def diffusion_loss_fn(self, x_0):
        batch_size = x_0.shape[0]
        
        t = torch.randint(0, self.num_steps, size=(batch_size//2,))
        t = torch.cat([t, self.num_steps - t - 1], 0)
        t = t.unsqueeze(-1)
        
        a = self.alphas_bar_sqrt[t].to(self.device)
        
        am1 = self.one_minus_alphas_bar_sqrt[t].to(self.device)
        
        e = torch.randn_like(x_0).to(self.device)
        
        x = (x_0 * a + am1 * e)
        t = t.squeeze(-1).to(self.device)
        
        opt = self.model(x, t)
        
        return (e-opt).square().mean()
    
    def train(self):

        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        for epoch in range(self.num_epoch):
            for _, batch_x in enumerate(self.dataloader):
                loss = self.diffusion_loss_fn(batch_x.to(self.device))

                optimizer.zero_grad()
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

            if (epoch % 100 == 0):
                print(loss)
    
    def save(self, path: str = './MLPDiffuser.pt'):
        torch.save(self.model.state_dict(), path)
    
    def load(self, path: str = './MLPDiffuser.pt'):
        self.model.load_state_dict(torch.load(path))
        if self.gpu:
            self.model.to(self.device)

    def p_sample(self, x, t):
        t = torch.tensor([t])  # type conversion
        coeff = self.betas[t] / self.one_minus_alphas_bar_sqrt[t]
        pre = (1/(1-self.betas[t]).sqrt()).to(self.device)
        sigma_ = self.betas[t].sqrt().to(self.device)  # 这里继续采用加噪时的固定 beta
        t = t.to(self.device)
        coeff = coeff.to(self.device)
        
        eps_theta = self.model(x, t)
        
        mean = pre * (x - coeff * eps_theta)
        
        z = torch.randn_like(x)
        
        
        sample = mean + sigma_ * z
        return (sample)
        

    def p_sample_loop(self):
        cur_x = torch.randn(self.shape).to(self.device)
        
        x_seq = [cur_x]
        
        # 这里相当于展示 去噪 num_steps 步，一步一步展示给你
        for i in reversed(range(self.num_steps)):
            cur_x = self.p_sample(cur_x, i)
            
            x_seq.append(cur_x)
        
        return x_seq
    
    def demo(self):
        self.model.eval
        x_seq = self.p_sample_loop()

        _, axs = plt.subplots(1, 10, figsize=(28, 3))
        for i in range(1, 11):
            cur_x = x_seq[i * 10].detach().cpu()
            axs[i-1].scatter(cur_x[:,0], cur_x[:,1], color='black')
            axs[i-1].axis('off')