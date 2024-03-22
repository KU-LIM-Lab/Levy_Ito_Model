from torchvision.utils import make_grid
import tqdm
import matplotlib.pyplot as plt
import numpy as np
import torch
from functions.Diffusion import *
import os 
import torch.backends.cudnn as cudnn
import torch
import numpy as np
from dataload import data_store
from evaluate.fid_score import fid_score
from torchlevy import LevyStable
import torchvision.utils as tvu 
levy = LevyStable()

def impainted_noise(sde, data, noise, mask,t):

    sigma = sde.marginal_std(t)
    alpha =sde.alpha
    x_coeff = sde.diffusion_coeff(t)

    if alpha == 2:
        e_L = torch.randn(size=(data.shape)) * np.sqrt(2)
        e_L = e_L.to(device)
    else:
        e_L = levy.sample(alpha, 0, size=(data.shape),  clamp=20).to(device)

    data = x_coeff[:, None, None, None] * data +  e_L* sigma[:, None, None, None]
    masked_data =data*mask+noise*(1-mask)

    return masked_data

def euler_maruyama_sampler(score_model, sde,  x_s, s, t, 
                           y=None, device='cuda', 
                           imputation=False,data=None, mask=None):
    alpha = sde.alpha
    if  y is not None:
        y = torch.ones((x_s.shape[0],))*y
        y = y.to(device)
    score_s = score_model(x_s, s, y)*torch.pow(sde.marginal_std(s),-(alpha-1))[:,None,None,None]
    time_step = s-t
    beta_step = sde.beta(s)*time_step
    x_coeff = 1 + beta_step/alpha
    e_L = levy.sample(alpha, 0, size=(x_s.shape)).to(device)
    noise_coeff = torch.pow(beta_step, 1 / alpha)
    score_coeff = alpha*beta_step
    x_t = x_coeff[:, None, None, None] * x_s + score_coeff[:, None, None, None] * score_s+noise_coeff[:, None, None,None] * e_L
    if imputation:
        x_t =impainted_noise(sde, data, x_t, mask,t)
    return x_t

def euler_variant_sampler(score_model, sde,x_s, s, t, 
                          y=None, device='cuda',
                          imputation=False,mask=None,data=None):
    alpha = sde.alpha
    if  y is not None:
        y = torch.ones((x_s.shape[0],))*y
        y = y.to(device)
    score_s = score_model(x_s, s, y)*torch.pow(sde.marginal_std(s),-(alpha-1))[:,None,None,None]
    a = torch.exp(sde.marginal_log_mean_coeff(t) - sde.marginal_log_mean_coeff(s))
    x_coeff = a        
    noise_coeff = torch.pow(-1 + torch.pow(a, sde.alpha), 1/sde.alpha)
    score_coeff = sde.alpha ** 2 * (-1 + a)
    e_L = levy.sample(alpha, 0, size=(x_s.shape)).to(device)
    x_t = x_coeff[:, None, None, None] * x_s + score_coeff[:, None, None, None] * score_s+noise_coeff[:, None, None,None] * e_L
    if imputation:
        x_t =impainted_noise(sde, data, x_t, mask,t)
    return x_t

def ode_variant_sampler(score_model, sde,  x_s, s, t, 
                        y=None, device='cuda',
                        imputation=False,mask=None,data=None):
    alpha = sde.alpha
    if  y is not None:
        y = torch.ones((x_s.shape[0],))*y
        y = y.to(device)
    score_s = score_model(x_s, s, y)*torch.pow(sde.marginal_std(s),-(alpha-1))[:,None,None,None]
    a = torch.exp(sde.marginal_log_mean_coeff(t) - sde.marginal_log_mean_coeff(s))
    x_coeff = a        
    score_coeff = sde.alpha * (-1 + a)
    x_t = x_coeff[:, None, None, None] * x_s + score_coeff[:, None, None, None] * score_s
    if imputation:
        x_t =impainted_noise(sde, data, x_t, mask,t)
    return x_t


def sampler(config,
            args,
            score_model,
            sde,
            batch_size,
            device='cuda',
            y=None,
            trajectory=False,
            mask=None,
            data=None):
    score_model.eval()
    x = sde.marginal_std(torch.ones((batch_size,),device=device)*sde.T)[:,None,None,None]*levy.sample(alpha=sde.alpha, size=(batch_size,config.data.channels,config.data.image_size,config.data.image_size),device=device)
    
    if trajectory:
        samples = []
        samples.append((x+1)/2)
        
    if args.sampler_type =="euler_maruyama_sampler":
        sampler = euler_maruyama_sampler
    elif args.sampler_type =="euler_variant_sampler":
        sampler = euler_variant_sampler
    elif args.sampler_type =="ode_variant_sampler":
        sampler = ode_variant_sampler
    
    
    timesteps = torch.linspace(sde.T,1e-5,args.nfe+1)
    
    with torch.no_grad():
        for i in tqdm.tqdm(range(args.nfe)):
            s = torch.ones((x.size(0),),device=device)*timesteps[i]
            t = torch.ones((x.size(0),),device=device)*timesteps[i+1]      
            x = sampler(score_model, sde, x, s, t, y, device=device,
                        imputation=args.imputation,mask=mask,data=data)  
            print('x',x.min(),x.max())    
            tvu.save_image((x+1)/2,'step.png')
            if trajectory:
                samples.append((1+x)/2)
    if trajectory:
        return samples
    else:
        return (x+1)/2


def fid_measure(config, args,train_loader,sde,model,rank):
    # img_id = len(glob.glob(f"{args.image_folder}/*"))
    img_id =0
    print(f"starting from image {img_id}")
    total_n_samples = 2000
    n_rounds = (total_n_samples) // config.sampling.fid_batch_size
    n_rounds = int(n_rounds // args.world_size)
    print('n_round', n_rounds)
    dataname = config.data.dataset
    dataset_path = "data/" + dataname.lower() + "_train_fid"
    if rank==0:
     if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)
        print('train dataset is storing...')
        data_store(train_loader ,dataset_path)
    for _ in tqdm.tqdm(range(n_rounds), desc="Generating image samples" ):
        x = sampler(config,
            args,
            model,
            sde,
            config.sampling.fid_batch_size,
            device='cuda',
            y=None,
            trajectory=False,
            mask=None)
        for i in range(len(x)):
            additional = str(int(rank)) + '_'+str(img_id)         
            tvu.save_image(x[i], os.path.join(args.image_folder, f"{additional}.png"),dpi =500)
            img_id +=1
    if rank ==0:
    #  fid_value = fid_score(dataset_path, args.image_folder, 50, device, num_workers=0)
     fid_value = fid_score(dataset_path, args.image_folder, 50, device, num_workers=0)
     print(f"FID with train dataset : {fid_value}")
     return fid_value
    else:
     return None