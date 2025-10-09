epochs=10
lr_a=5e-4
lr_w=5e-4
batch_size=64
train_batch=64
# n_lva=16 
# n_lvw=16 
# n_lva_o=256 
data_path="./dataset/ouromamba_dataset" # "./dataset/ouromamba_dataset" or "./dataset/imagenet"
model_t="vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2"
model_s="vim_small_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2"
model_b="vim_base_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_middle_cls_token_div2"
resume_t="./pretrained/vim_t_midclstok_76p1acc.pth"
resume_s="./pretrained/vim_s_midclstok_80p5acc.pth"
resume_b="./pretrained/vim_b_midclstok_81p9acc.pth"

# ==========================
# Vim Small Quantization
# ==========================
## W4A4O8 Quantization
echo "**********************************************************************"
echo "Running Vim Small W4A4O8 Quantization"
CUDA_VISIBLE_DEVICES=0 python ./quantize/quant.py --eval --mode-quantize --num_workers 1\
    --epochs ${epochs} --lr-a ${lr_a} --lr-w ${lr_w} \
    --model ${model_s} --resume ${resume_s} --data-path ${data_path} \
    --batch-size ${batch_size} --train-batch ${train_batch} --n-lva 16 --n-lvw 16 --n-lva-o 256 \
    > ./results/vim_s_w4a4o8.log 2>&1

## W4A8O16 Quantization
echo "**********************************************************************"
echo "Running Vim Small W4A8O16 Quantization"
CUDA_VISIBLE_DEVICES=0 python ./quantize/quant.py --eval --mode-quantize  --num_workers 1 \
    --epochs ${epochs} --lr-a ${lr_a} --lr-w ${lr_w} \
    --model ${model_s} --resume ${resume_s} --data-path ${data_path} \
    --batch-size ${batch_size} --train-batch ${train_batch} --n-lva 256 --n-lvw 16 --n-lva-o 65536 \
    > ./results/vim_s_w4a8o16.log 2>&1

# # [Optional] W4A4O8 Quantization w/ Static Outlier List
# echo "**********************************************************************"
# echo "Generating Static Outlier List for Vim Small"
# CUDA_VISIBLE_DEVICES=0 python ./quantize/generate_static_outlier_list.py \
#     --resume ${resume_s} --model ${model_s} --data-path ${data_path} --batch-size 12 --list-output-path ./quantize/

# echo "Running Vim Small W4A4O8 Quantization with Static Outlier List"
# CUDA_VISIBLE_DEVICES=0 python ./quantize/quant.py --eval  --mode-quantize  --num_workers 1 \
#     --epochs ${epochs} --lr-a ${lr_a} --lr-w ${lr_w} \
#     --model ${model_s} --resume ${resume_s} --data-path ${data_path} \
#     --batch-size ${batch_size} --train-batch ${train_batch} --n-lva 16 --n-lvw 16 --n-lva-o 256 \
#     --static_outlier_list ./quantize/static_list_s.pt \
#     > ./results/vim_s_w4a4o8_static.log 2>&1





# ==========================
# Vim Base Quantization
# ==========================
## W4A4O8 Quantization
echo "**********************************************************************"
echo "Running Vim Base W4A4O8 Quantization"
CUDA_VISIBLE_DEVICES=0 python ./quantize/quant.py --eval --mode-quantize  --num_workers 1 \
    --epochs ${epochs} --lr-a ${lr_a} --lr-w ${lr_w} \
    --model ${model_b} --resume ${resume_b} --data-path ${data_path} \
    --batch-size ${batch_size} --train-batch ${train_batch} --n-lva 16 --n-lvw 16 --n-lva-o 256 \
    > ./results/vim_b_w4a4o8.log 2>&1

## W4A8O16 Quantization
echo "**********************************************************************"
echo "Running Vim Base W4A8O16 Quantization"
CUDA_VISIBLE_DEVICES=0 python ./quantize/quant.py --eval --mode-quantize  --num_workers 1 \
    --epochs ${epochs} --lr-a ${lr_a} --lr-w ${lr_w} \
    --model ${model_b} --resume ${resume_b} --data-path ${data_path} \
    --batch-size ${batch_size} --train-batch ${train_batch} --n-lva 256 --n-lvw 16 --n-lva-o 65536 \
    > ./results/vim_b_w4a8o16.log 2>&1

# # [Optional] W4A4O8 Quantization w/ Static Outlier List
# echo "**********************************************************************"
# echo "Generating Static Outlier List for Vim Base"
# CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.run --nproc_per_node 1 ./quantize/generate_static_outlier_list.py \
#     --resume ${resume_b} --model ${model_b} --data-path ${data_path} --batch-size 12 --list-output-path ./quantize/
# echo "Running Vim Base W4A4O8 Quantization with Static Outlier List"
# CUDA_VISIBLE_DEVICES=0 python ./quantize/quant.py --eval --mode-quantize  --num_workers 1 \
#     --epochs ${epochs} --lr-a ${lr_a} --lr-w ${lr_w} \
#     --model ${model_b} --resume ${resume_b} --data-path ${data_path} \
#     --batch-size ${batch_size} --train-batch ${train_batch} --n-lva 16 --n-lvw 16 --n-lva-o 256 \
#     --static_outlier_list ./quantize/static_list_b.pt \
#     > ./results/vim_b_w4a4o8_static.log 2>&1

