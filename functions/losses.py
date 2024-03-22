import torch




def loss_fn(model, sde,
            x0: torch.Tensor,
            t: torch.LongTensor,
            y,
            e_L: torch):
    sigma = sde.marginal_std(t)
    x_coeff = sde.diffusion_coeff(t)
    score = -e_L/sde.alpha
    x_t = x_coeff[:, None, None, None] * x0 + e_L * sigma[:, None, None, None]
    output = model(x_t, t,y)
    weight = output-score
    loss = (weight).square().sum(dim=(1, 2, 3)).mean(dim=0)
    return  loss
