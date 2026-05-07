export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1


CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=29500 run/train_diffusion_ddp.py \
--model_id 3 \
--epochs 1000 \
--save_step 500 \
--save_path /exhdd/seungyu/diffusion_motion/trained_weights/diffuser_ddp_45m_2k/run1 \
--config_path /exhdd/seungyu/diffusion_motion/core/diffuser/config/DDP/diffuser_ddp.py \
--dataset_id 1 \
--batch_size 256 \
--num_workers 2 \
--pin_memory > diffuser_ddp_45m_2k_train.log 2>&1 &


CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=29500 run/train_diffusion_ddp.py \
--model_id 3 \
--epochs 1000 \
--save_step 500 \
--save_path /exhdd/seungyu/diffusion_motion/trained_weights/diffuser_50m_100k_ddp_16x16_64h_mask_only/run1 \
--config_path /exhdd/seungyu/diffusion_motion/core/diffuser/config/DDP/diffuser_ddp.py \
--dataset_id 7 \
--batch_size 256 \
--num_workers 2 \
--pin_memory > diffuser_50m_100k_ddp_16x16_64h_mask_only_train.log 2>&1 &


CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 --master_port=29501 run/train_diffusion_ddp.py \
--model_id 3 \
--epochs 1000 \
--save_step 500 \
--save_path /exhdd/seungyu/diffusion_motion/trained_weights/diffuser_53m_100k_ddp_32x32_128h_mask_only/run1 \
--config_path /exhdd/seungyu/diffusion_motion/core/diffuser/config/DDP/diffuser_ddp.py \
--dataset_id 8 \
--batch_size 256 \
--num_workers 2 \
--pin_memory > diffuser_53m_100k_ddp_32x32_128h_mask_only_train.log 2>&1 &
