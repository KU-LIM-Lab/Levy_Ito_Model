# ivus_frame_generation

Environment setting 
```
cd Levy_Ito_Model
```
```
pip install -r requirements.txt
```


command
```
CUDA_VISIBLE_DEVICIES=0,1 LOCAK_RANK=0,1 torchrun --master_port 12355 --nproc_per_node 2 main.py --exp t_cifar10_2.0  --seed 0 --config cifar10_ncsnpp_2.0.yaml --ddp --nfe 1000  
```
