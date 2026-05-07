export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Training with 4 gpus
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=29500 run/train_pb_ddp.py \
--model_id 2 \
--config_path /exhdd/seungyu/diffusion_motion/core/pb_diffusion/configs/pb_ddp_v2_cfg.py \
--dataset_id 4 \
--use_ddp > pb_v2_ddp_40k_train.log 2>&1 &


# Training with 4 gpus
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=29500 run/train_pb_ddp.py \
--model_id 2 \
--config_path /exhdd/seungyu/diffusion_motion/core/pb_diffusion/configs/pb_ddp_v2_cfg.py \
--dataset_id 7 \
--use_ddp > pb_v2_ddp_16x16_64h_50m_100k_train.log 2>&1 &