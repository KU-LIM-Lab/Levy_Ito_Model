
import argparse
import os
from training import *
import torch.distributed as dist
import os
import time
import yaml
from torch.backends import cudnn
import torch.utils.tensorboard as tb


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace

#arges
parser = argparse.ArgumentParser(description='LIM')
parser.add_argument("--sample", action="store_true")
parser.add_argument("--resume", action="store_true")
parser.add_argument('--sampler_type', type=str, default='euler_maruyama_sampler')
parser.add_argument('--imputation',type=bool, default=False)
parser.add_argument( "--config", type=str, required=True, help="Path to the config file")
parser.add_argument("--seed", type=int, default=1234, help="Random seed")
parser.add_argument("--exp", type=str, default="exp", help="Path for saving running related data.")
parser.add_argument( "--ddp", action="store_true",help="Whether to perform imputation sampling")
parser.add_argument( "--fid", action="store_true",help="Whether to measure fid")
parser.add_argument("--nfe", type=int, default=500, help="number of function evaluations")
args = parser.parse_args()

#foldes 
args.log_path = os.path.join(args.exp, "logs")
args.samples= os.path.join(args.exp, "samples")
args.image_folder = os.path.join(args.exp, "fid")
if os.path.exists(args.log_path):
    pass
else:
    os.makedirs(args.log_path)
if os.path.exists(args.samples):
    pass
else:
    os.makedirs(args.samples)
if os.path.exists(args.image_folder):
    pass
else:
    os.makedirs(args.image_folder)
    
#tensorboard 
tb_path = os.path.join(args.exp, "tensorboard")
args.tb_logger = tb.SummaryWriter(log_dir=tb_path)

# ddp setting 
if args.ddp:
    args.rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(args.rank)
    torch.cuda.empty_cache()
    args.world_size =  int(os.environ['WORLD_SIZE'])
    dist.init_process_group(backend='nccl',
                            init_method='env://',
                            world_size=args.world_size,
                            rank=args.rank)
    args.world_size = float(dist.get_world_size())
else: 
    args.rank = 0
    args.world_size = 1
    

# seed 
if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

# config 
with open(os.path.join("config", args.config), "r") as f:
    config = yaml.safe_load(f)
config = dict2namespace(config)


def main():
    train(args,config)

if __name__=='__main__':
    cudnn.benchmark = False
    main()

