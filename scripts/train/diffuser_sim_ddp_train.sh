export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1


# Training with 4 gpus
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=29500 run/train_diffusion_ddp.py \
--model_id 9 \
--epochs 1000 \
--save_step 500 \
--save_path /exhdd/seungyu/diffusion_motion/trained_weights/diffuser_sim_fusion_ddp_feat128_40k/run1 \
--config_path /exhdd/seungyu/diffusion_motion/core/diffuser/config/DDP/diffuser_sim_ddp.py \
--dataset_id 4 \
--batch_size 256 \
--num_workers 2 \
--pin_memory > diffuser_ddp_sim_fusion_feat128_40k_train.log 2>&1 &


CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 --master_port=29501 run/train_diffusion_ddp.py \
--model_id 9 \
--epochs 1000 \
--save_step 500 \
--save_path /exhdd/seungyu/diffusion_motion/trained_weights/diffuser_sim_fusion_ddp_feat128_100k/run1 \
--config_path /exhdd/seungyu/diffusion_motion/core/diffuser/config/DDP/diffuser_sim_ddp.py \
--dataset_id 5 \
--batch_size 256 \
--num_workers 2 \
--pin_memory > diffuser_ddp_sim_fusion_feat128_100k_train.log 2>&1 &