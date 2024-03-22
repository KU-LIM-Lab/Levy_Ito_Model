import torch
from torch.distributions.exponential import Exponential

import numpy as np
import math

class LevyStable:

    def sample(self, alpha, beta=0, size=1, loc=0, scale=1, 
               type=torch.float32, device=None, reject_threshold=20,clamp_threshold=None,clamp=10):
        
        if alpha==2.0:
            return np.sqrt(2)*torch.randn(size, device=device)

        else :
            if isinstance(size, int):
                size_scalar = size
                size = (size,)
            else:
                size_scalar = 1
                for i in size:
                    size_scalar *= i

            num_sample = size[0]
            dim = int(size_scalar / num_sample)

            x = self._sample(alpha / 2, beta=1, size=2*num_sample , device=device)
            
            
            # if clamp_threshold is not None:
            #         indices = x.norm(dim=1)[:, None] > clamp_threshold*x.shape[-1]
            #         x[indices] = x[indices] / x[indices].norm(dim=1)[:, None] * clamp_threshold*x.shape[-1]
            x = x * 2 * scale**2*np.cos(np.pi * alpha / 4) ** (2 / alpha)
            
            
            z = torch.randn(size=(2*num_sample, dim),device=device)
            e = x[:,None] ** (1 / 2) * z
    
            if reject_threshold is not None:
                e =e[x<reject_threshold]
            # if clamp is not None:
            #         e = torch.clamp(e, -clamp, clamp)
            a = e[:num_sample].reshape(size)
        

            return a
        


    # def _sample(self, alpha, beta=0, size=1,device='cuda'):

    #     def otherwise(alpha, beta, TH, aTH, bTH, cosTH, tanTH, W):
    #         # val0 = beta * np.tan(np.pi * alpha / 2)
    #         # th0 = np.arctan(val0) / alpha
    #         # val3 = W / ((1e-5+cosTH) / torch.tan(alpha * (th0 + TH)) + torch.sin(TH))
    #         # res3 = val3 * ((torch.cos(aTH) + torch.sin(aTH) * tanTH -
    #         #                 val0 * (torch.sin(aTH) - torch.cos(aTH) * tanTH)) / W) ** (1 / alpha)
            
    #         s_ab=(1+beta**2*np.tan(np.pi*alpha/2))**(1/2/alpha)
    #         b_ab = np.arctan(beta*np.tan(np.pi/2*alpha))/alpha
    #         res3 = s_ab*torch.sin(alpha*(TH+b_ab))/(torch.cos(TH)**(1/alpha))*(torch.cos(TH-alpha*(TH+b_ab))/W)**((1-alpha)/alpha)
            
    #         return res3

    #     TH = torch.rand(size, device=device) * (torch.pi-1e-5) - ((torch.pi-1e-5) / 2.0)
    #     W = Exponential(torch.tensor([1.0])).sample([size]).reshape(-1).to(device)
    #     aTH = alpha * TH
    #     bTH = beta * TH
    #     cosTH = torch.cos(TH)
    #     tanTH = torch.tan(TH)
    

 
    #     return otherwise(alpha, beta, TH, aTH, bTH, cosTH, tanTH, W)
    def _sample(self, alpha, beta=0, size=1, type=torch.float32,device='cuda'):
    
        def alpha1func(alpha, beta, TH, aTH, bTH, cosTH, tanTH, W):
            return 2 / torch.pi * ((torch.pi / 2 + bTH) * tanTH
                                   - beta * torch.log((torch.pi / 2 * W * cosTH) / (torch.pi / 2 + bTH)))

        def beta0func(alpha, beta, TH, aTH, bTH, cosTH, tanTH, W):
            return (W / (cosTH / torch.tan(aTH) + torch.sin(TH)) *
                    ((torch.cos(aTH) + torch.sin(aTH) * tanTH) / W) ** (1 / alpha))

        def otherwise(alpha, beta, TH, aTH, bTH, cosTH, tanTH, W):
            # alpha != 1 and beta != 0
            val0 = beta * torch.tan(torch.tensor([torch.pi * alpha / 2], dtype=torch.float64)).to(device)
            th0 = torch.arctan(val0) / alpha
            val3 = W / (cosTH / torch.tan(alpha * (th0 + TH)) + torch.sin(TH))
            res3 = val3 * ((torch.cos(aTH) + torch.sin(aTH) * tanTH -
                            val0 * (torch.sin(aTH) - torch.cos(aTH) * tanTH)) / W) ** (1 / alpha)
            
            s_ab=(1+beta**2*np.tan(np.pi*alpha/2))**(1/2/alpha)
            b_ab = np.arctan(beta*np.tan(np.pi/2*alpha))/alpha
            res3 = s_ab*torch.sin(alpha*(TH+b_ab))/(torch.cos(TH)**(1/alpha))*(torch.cos(TH-alpha*(TH+b_ab))/W)**((1-alpha)/alpha)
            return res3

        # TH = torch.rand(size, dtype=torch.float64) * torch.pi - (torch.pi / 2.0)
        TH = torch.rand(size, dtype=torch.float64,device=device) * (torch.pi-1e-5) - ((torch.pi-1e-5 ) / 2.0)
        W = Exponential(torch.tensor([1.0])).sample([size]).reshape(-1).to(device)
        aTH = alpha * TH
        bTH = beta * TH
        cosTH = torch.cos(TH)
        tanTH = torch.tan(TH)

        if alpha == 1:
            return alpha1func(alpha, beta, TH, aTH, bTH, cosTH, tanTH, W).to(type)
        elif beta == 0:
            return beta0func(alpha, beta, TH, aTH, bTH, cosTH, tanTH, W).to(type)
        else:
            return otherwise(alpha, beta, TH, aTH, bTH, cosTH, tanTH, W).to(type)
