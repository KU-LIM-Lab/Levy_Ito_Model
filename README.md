# Levy_Ito_Model

## Environment Setting

First, navigate into the directory.

```
cd Levy_Ito_Model
```

Next, install the necessary packages.

```
pip install -r requirements.txt
```

## Training Command

To train the model, use the following command:

```
CUDA_VISIBLE_DEVICES=0,1 LOCAK_RANK=0,1 torchrun --master_port 12355 --nproc_per_node 2 main.py --exp t_cifar10_2.0  --seed 0 --config cifar10_ddpm.yaml --ddp 
```

Parameters:
- `--exp [folder_name]`: Name of the folder to save the experiment results.
- `--seed [integer]`: Seed for reproducibility.
- `--config [configuration_file]`: Configuration file to reference. Example files (`cifar10_ddpm.yaml` and `cifar10_ncnsnpp.yaml`) are located in the `config` directory.
- `--ddp`: Run the code in a GPU environment.
- `--resume`: Resume training from a saved checkpoint in the specified folder.

## Sampling Command

To sample from the trained model, use the following command:

```
CUDA_VISIBLE_DEVICES=0,1 LOCAK_RANK=0,1 torchrun --master_port 12355 --nproc_per_node 2 main.py --exp t_cifar10_2.0  --seed 0 --config cifar10_ddpm.yaml --ddp --resume --sampling 
```

Parameters:
- `--exp [folder_name]`: Name of the folder containing the trained model.
- `--seed [integer]`: Seed for reproducibility.
- `--config [configuration_file]`: Configuration file to reference. Example files (`cifar10_ddpm.yaml` and `cifar10_ncnsnpp.yaml`) are located in the `config` directory.
- `--ddp`: Run the code in a GPU environment.
- `--resume`: Load the saved checkpoint from the specified folder for sampling.
- `--sample`: Enable sampling mode.
- `--fid`: Calculate the Frechet Inception Distance (FID).
