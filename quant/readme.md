# OuroMamaba-Quant
```
.
├── vim/              # Image Classification
└── pixart/           # Image Generation
```

## ViM
### Setting
1. conda environment setting
```
cd ICCV-OuroMamba/quant/vim
conda env create -f vim.yml
conda activate vim_quant

pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu121 \
    "torch==2.1.2+cu121" "torchvision==0.16.2+cu121"

wget https://github.com/state-spaces/mamba/releases/download/v1.2.0.post1/mamba_ssm-1.2.0.post1+cu122torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install mamba_ssm-1.2.0.post1+cu122torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

wget https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.1.1/causal_conv1d-1.1.1+cu122torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install causal_conv1d-1.1.1+cu122torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

rm ./causal_conv1d-1.1.1+cu122torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
rm ./mamba_ssm-1.2.0.post1+cu122torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
```

2. Download VMM Checkpoints
```
huggingface-cli download hustvl/Vim-tiny-midclstok --local-dir ./pretrained/
huggingface-cli download hustvl/Vim-small-midclstok --local-dir ./pretrained/
huggingface-cli download hustvl/Vim-base-midclstok --local-dir ./pretrained/
```

3. Prepare Dataset in ./dataset
Make sure to follow the structure outlined below.
```
dataset/
├── imagenet/ or ouromamba_dataset/
    ├── train/
    ├── test/   (fix to imagenet for comparison)
    └── val/    (fix to imagenet for comparison)
```
To use the pre-generated Data-Free Quantization (DFQ) image set, run:
```
pip install gdown
gdown --folder "https://drive.google.com/drive/folders/16AuCvOOGo7pYJ2nqqu7R949_HjTfcJ5K?usp=sharing" -O ./dataset/ --remaining-ok
```

4. Modify mamba files
```
cp ./modified/mamba_simple.py                  ~/miniconda3/envs/vim_quant/lib/python3.10/site-packages/mamba_ssm/modules/
cp ./modified/selective_scan_interface.py      ~/miniconda3/envs/vim_quant/lib/python3.10/site-packages/mamba_ssm/ops/
```

### Quantization - ViM
#### Quick Evaluation
```
cd quantize/
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.run --nproc_per_node 1 --master_port 12340 quant.py --eval \
   --mode-quantize --epochs 10 --lr-a 5e-4 --lr-w 5e-4\
    --model <model_name>  \
    --resume <model_checkpoint_path> \
    --data-path <calibration_dataset_path> \
    --batch-size <inference_batch_size> --train-batch <calibration_batch_size> \
    --n-lva <target_act_bit_width> --n-lvw <target_weight_bit_width> --n-lva-o <target_outlier_bit_width>
```

#### ImageNet Evaluation script
```
cd ICCV-OuroMamba/quant/vim/
bash eval_vim.sh
```

<!-- 
### [Example] Vim small
```
cd quantize

CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.run --nproc_per_node 1 --master_port 12340 quant.py --eval \
   --mode-quantize --epochs 10 --lr-a 5e-4 --lr-w 5e-4\
    --model vim_small_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2  \
    --resume ../pretrained/vim_s_midclstok_80p5acc.pth \
    --data-path ../dataset/imagenet \
    --batch-size 128 --train-batch 128 --n-lva 16 --n-lvw 16 --n-lva-o 256 
```

### [Example] Vim Tiny
```
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.run --nproc_per_node 1 --master_port 12340 quant.py --eval \
   --mode-quantize --epochs 10 --lr-a 5e-4 --lr-w 5e-4\
    --model vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2  \
    --resume /nethome/mlee864/ICCV-OuroMamba/quant/vim/pretrained/vim_t_midclstok_76p1acc.pth \
    --data-path /nethome/mlee864/ICCV-OuroMamba/quant/vim/dataset/imagenet \
    --static_outlier_list ./static_list_t.pt
    --batch-size 128 --train-batch 128 --n-lva 16 --n-lvw 16 --n-lva-o 256
```

### [Example] Vim-Base
```
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.run --nproc_per_node 1 --master_port 12340 quant.py --eval \
   --mode-quantize --epochs 10 --lr-a 5e-4 --lr-w 5e-4 \
    --resume ../pretrained/vim_b_midclstok_81p9acc.pth \
    --model vim_base_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_middle_cls_token_div2  \
    --data-path ../dataset/imagenet \
    --batch-size 128 --train-batch 128 --n-lva 16 --n-lvw 16 --n-lva-o 256
``` -->


## PixArt-sigma
### Setting
1. conda environment setting
```
cd ICCV-OuroMamba/quant/pixart
conda env create -f pixart.yml
conda activate pixart_quant
```

2. Image Generation with Ouromamba-Quant (W4A8, W4A4)
```
python ./PixArt-sigma/generate.py --mode fp16 --filename fp16.png
python ./PixArt-sigma/generate.py --mode w4a8 --filename w4a8.png
python ./PixArt-sigma/generate.py --mode w4a4 --filename w4a4.png
```