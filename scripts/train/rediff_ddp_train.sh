# ReDiff DDP Training
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 --master_port=29500 run/train_rediffuser_ddp.py \
--model_id 0 \
--ddp_trained \
--config_path /exhdd/seungyu/diffusion_motion/core/rediffuser/configs/rediff_ddp_cfg.py \
--dataset_id 5 > rediff_ddp_40k_train.log 2>&1 &



CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 --master_port=29501 run/train_rediffuser_ddp.py \
--model_id 0 \
--ddp_trained \
--config_path /exhdd/seungyu/diffusion_motion/core/rediffuser/configs/rediff_ddp_cfg.py \
--dataset_id 7 > rediff_ddp_16x16_64h_100k_train_2nd.log 2>&1 &