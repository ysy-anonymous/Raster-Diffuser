export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Training with 4 gpus
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=29501 run/train_comp_ddp.py \
--model_id 1 \
--config_path /exhdd/seungyu/diffusion_motion/core/comp_diffusion/configs/comp_v2_ddp_cfg.py \
--dataset_id 4 \
--use_ddp > comp_v2_ddp_40k_train.log 2>&1 &


CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 --master_port=29501 run/train_comp_ddp.py \
--model_id 1 \
--config_path /exhdd/seungyu/diffusion_motion/core/comp_diffusion/configs/comp_v2_ddp_cfg.py \
--dataset_id 5 \
--use_ddp > comp_v2_ddp_100k_train.log 2>&1 &



CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=29500 run/train_comp_ddp.py \
--model_id 0 \
--config_path /exhdd/seungyu/diffusion_motion/core/comp_diffusion/configs/comp_diffusion_ddp_cfg.py \
--dataset_id 4 \
--use_ddp > comp_diff_ddp_40k_train.log 2>&1 &


CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 --master_port=29501 run/train_comp_ddp.py \
--model_id 0 \
--config_path /exhdd/seungyu/diffusion_motion/core/comp_diffusion/configs/comp_diffusion_ddp_cfg.py \
--dataset_id 5 \
--use_ddp > comp_diff_ddp_100k_train.log 2>&1 &


# 16x16, 64 horizon, 100k training dataset
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 --master_port=29501 run/train_comp_ddp.py \
--model_id 1 \
--config_path /exhdd/seungyu/diffusion_motion/core/comp_diffusion/configs/comp_v2_ddp_cfg.py \
--dataset_id 7 \
--use_ddp > comp_v2_ddp_16x16_64h_100k_train.log 2>&1 &