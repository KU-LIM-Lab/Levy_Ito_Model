import os
from functions.util import *
from functions.losses import *
from functions.Diffusion import *
import numpy as np
import torch
from transformers import get_scheduler
from functions.sampler import sampler, fid_measure
from torchlevy import LevyStable
import torch.nn as nn
levy = LevyStable()



def train(args,config):
    lr = config.optim.lr
    alpha = config.diffusion.alpha
    n_epochs = config.training.n_epochs
    if args.ddp:
     device = torch.device('cuda:{}'.format(args.rank))
    else:
     device = 'cuda'    
    condition = config.model.condition

    sde = VPSDE(config)
    model = get_model(config)
    model.to(device)
    if args.ddp:
     model = nn.parallel.DistributedDataParallel(model, device_ids=[args.rank])
 
    data_loader, validation_loader, train_sampler = get_datasets(args,config)



    num_training_steps = n_epochs * len(data_loader)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )
    model.train()

    counter = 0
    if args.resume:
        checkpoint_file = os.path.join(args.exp, 'logs/ckpt.pth')
        checkpoint = torch.load(checkpoint_file, map_location=device)
        init_epoch = checkpoint['epoch']
        epoch = init_epoch
        model.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['scheduler'])
        global_step = checkpoint['global_step']
        print("=> loaded checkpoint (epoch {})"
                  .format(checkpoint['epoch']))
    else:
        global_step, epoch, init_epoch = 0, 0, 0

    if args.sample:
        if args.fid:
         fid_value =fid_measure(config, args,data_loader,sde,model,args.rank,n_samples=50000)
         if args.rank==0:
          x= sampler(config,args,model,sde,config.sampling.batch_size,device=device,y=None,trajectory=False,mask=None)
          torchvision.utils.save_image(x, os.path.join(args.exp, 'samples/sample_step_{}_FID_{:.3f}.png'.format(global_step,fid_value)))
        else: 
         if args.rank==0:
          x= sampler(config,args,model,sde,config.sampling.batch_size,device=device,y=None,trajectory=False,mask=None)
          torchvision.utils.save_image(x, os.path.join(args.exp, 'samples/sample_step_{}.png'.format(global_step)))
        
        return None 

    for epoch in range(init_epoch,n_epochs):
        if args.ddp:
         train_sampler.set_epoch(epoch)
        counter += 1
        avg_loss = 0.
        num_items = 0
        for x, y in data_loader:
            x = 2 * x - 1
            x = x.to(device)
            y= y.to(device)
            t = torch.rand(x.shape[0]).to(device)*(sde.T-0.00001)+0.00001
            e_L = levy.sample(alpha, 0, size=(x.shape)).to(device)

            if condition == False:
                y = None
            if np.random.random() < 0.2:
                y = None

            loss = loss_fn(model, sde, x, t,y, e_L=e_L)

            optimizer.zero_grad()
            loss.backward()
            args.tb_logger.add_scalar("train_loss", loss, global_step=global_step)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.optim.grad_clip)
            optimizer.step()
            lr_scheduler.step()
            print('step {}, Loss: {}, lr: {}'.format(global_step, loss.item(),optimizer.param_groups[0]['lr']))
            global_step += 1
            optimizer.step()
            avg_loss += loss.item() * x.shape[0]
            num_items += x.shape[0]
            if global_step%config.training.ckpt_store == 0 :
             if args.rank ==0:  
                print('ckpt store...')     
                content = {'epoch': epoch + 1, 
                            'global_step': global_step, 
                            'net': model.state_dict(), 
                            'optimizer': optimizer.state_dict(),
                            'scheduler': lr_scheduler.state_dict(),
                            }
                    
                torch.save(content, os.path.join(args.exp, 'logs/ckpt.pth'))
                torch.save(content, os.path.join(args.exp, 'logs/net_{}.pth'.format(global_step)))
                x= sampler(config,args,model,sde,config.sampling.batch_size,device=device,y=None,trajectory=False,mask=None)
                torchvision.utils.save_image(x, os.path.join(args.exp, 'samples/sample_step_{}.png'.format(global_step)))
             if args.rank==0:
                 torchvision.utils.save_image(x, os.path.join(args.exp, 'samples/sample_step_{}.png'.format(global_step)))
        else:
            with torch.no_grad():
                counter += 1
                val_avg_loss = 0.
                val_num_items = 0
                for x, y in validation_loader:
                    x= 2*x-1
                    n = x.size(0)
                    x = x.to(device)
                    y= y.to(device)
                    t = torch.rand(x.shape[0]).to(device)*(sde.T-0.00001)+0.00001
                    if condition == False:
                        y = None
                    if np.random.random() < 0.2:
                        y = None
                    e_L = levy.sample(alpha, 0, size=(x.shape)).to(device)
                    val_loss = loss_fn(model, sde, x, t,y, e_L)
                    val_avg_loss += val_loss.item() * x.shape[0]
                    val_num_items += x.shape[0]
        val_avg_loss = val_loss /val_num_items
        args.tb_logger.add_scalar("val_loss", val_avg_loss, global_step=global_step)

        