export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Training with 4 gpus
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=29500 run/train_diffusion_ddp.py \
--model_id 4 \
--epochs 1000 \
--save_step 500 \
--save_path /exhdd/seungyu/diffusion_motion/trained_weights/diffuser_bil_45m_40k_ddp_iter_2/run1 \
--config_path /exhdd/seungyu/diffusion_motion/core/diffuser/config/DDP/diffuser_bil_ddp.py \
--dataset_id 4 \
--batch_size 256 \
--num_workers 2 \
--pin_memory > diffuser_ddp_bil_45M_40K_iter_2_train.log 2>&1 &


CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 --master_port=29501 run/train_diffusion_ddp.py \
--model_id 4 \
--epochs 1000 \
--save_step 500 \
--save_path /exhdd/seungyu/diffusion_motion/trained_weights/diffuser_bil_45m_40k_ddp_iter_1/run1 \
--config_path /exhdd/seungyu/diffusion_motion/core/diffuser/config/DDP/diffuser_bil_ddp.py \
--dataset_id 4 \
--batch_size 256 \
--num_workers 2 \
--pin_memory > diffuser_ddp_bil_45M_40K_iter_1_train.log 2>&1 &


# Training with 4 gpus
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=29500 run/train_diffusion_ddp.py \
--model_id 4 \
--epochs 1000 \
--save_step 500 \
--save_path /exhdd/seungyu/diffusion_motion/trained_weights/diffuser_bil_45m_40k_ddp_raster_only/run1 \
--config_path /exhdd/seungyu/diffusion_motion/core/diffuser/config/DDP/diffuser_bil_ddp.py \
--dataset_id 4 \
--batch_size 256 \
--num_workers 2 \
--pin_memory > diffuser_ddp_bil_45M_40K_raster_only_train.log 2>&1 &


CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 --master_port=29501 run/train_diffusion_ddp.py \
--model_id 4 \
--epochs 1000 \
--save_step 500 \
--save_path /exhdd/seungyu/diffusion_motion/trained_weights/diffuser_bil_45m_40k_ddp_raster_only_no_distmap/run1 \
--config_path /exhdd/seungyu/diffusion_motion/core/diffuser/config/DDP/diffuser_bil_ddp.py \
--dataset_id 4 \
--batch_size 256 \
--num_workers 2 \
--pin_memory > diffuser_ddp_bil_45M_40K_raster_only_no_distmap_train.log 2>&1 &



# Training with 4 gpus
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=29500 run/train_diffusion_ddp.py \
--model_id 4 \
--epochs 1000 \
--save_step 500 \
--save_path /exhdd/seungyu/diffusion_motion/trained_weights/diffuser_bil_45m_40k_ddp_upcoeff_05/run1 \
--config_path /exhdd/seungyu/diffusion_motion/core/diffuser/config/DDP/diffuser_bil_ddp.py \
--dataset_id 4 \
--batch_size 256 \
--num_workers 2 \
--pin_memory > diffuser_ddp_bil_45M_40K_upcoeff_05_train.log 2>&1 &


CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 --master_port=29501 run/train_diffusion_ddp.py \
--model_id 4 \
--epochs 1000 \
--save_step 500 \
--save_path /exhdd/seungyu/diffusion_motion/trained_weights/diffuser_bil_45m_40k_ddp_upcoeff_10/run1 \
--config_path /exhdd/seungyu/diffusion_motion/core/diffuser/config/DDP/diffuser_bil_ddp.py \
--dataset_id 4 \
--batch_size 256 \
--num_workers 2 \
--pin_memory > diffuser_ddp_bil_45M_40K_upcoeff_10_train.log 2>&1 &



# Training with 4 gpus
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=29500 run/train_diffusion_ddp.py \
--model_id 4 \
--epochs 1000 \
--save_step 500 \
--save_path /exhdd/seungyu/diffusion_motion/trained_weights/diffuser_bil_45m_40k_ddp_upcoeff_learnable/run1 \
--config_path /exhdd/seungyu/diffusion_motion/core/diffuser/config/DDP/diffuser_bil_ddp.py \
--dataset_id 4 \
--batch_size 256 \
--num_workers 2 \
--pin_memory > diffuser_ddp_bil_45M_40K_upcoeff_learnable_train.log 2>&1 &


CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=29500 run/train_diffusion_ddp.py \
--model_id 4 \
--epochs 1000 \
--save_step 500 \
--save_path /exhdd/seungyu/diffusion_motion/trained_weights/diffuser_bil_45m_40k_ddp_sdf_only/run1 \
--config_path /exhdd/seungyu/diffusion_motion/core/diffuser/config/DDP/diffuser_bil_ddp.py \
--dataset_id 4 \
--batch_size 256 \
--num_workers 2 \
--pin_memory > diffuser_ddp_bil_45M_40K_sdf_only_train.log 2>&1 &


CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 --master_port=29501 run/train_diffusion_ddp.py \
--model_id 4 \
--epochs 1000 \
--save_step 500 \
--save_path /exhdd/seungyu/diffusion_motion/trained_weights/diffuser_bil_45m_40k_ddp_bmask_only/run1 \
--config_path /exhdd/seungyu/diffusion_motion/core/diffuser/config/DDP/diffuser_bil_ddp.py \
--dataset_id 4 \
--batch_size 256 \
--num_workers 2 \
--pin_memory > diffuser_ddp_bil_45M_40K_bmask_only_train.log 2>&1 &


CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=29500 run/train_diffusion_ddp.py \
--model_id 4 \
--epochs 1000 \
--save_step 500 \
--save_path /exhdd/seungyu/diffusion_motion/trained_weights/diffuser_bil_45m_40k_ddp_dmap_only/run1 \
--config_path /exhdd/seungyu/diffusion_motion/core/diffuser/config/DDP/diffuser_bil_ddp.py \
--dataset_id 4 \
--batch_size 256 \
--num_workers 2 \
--pin_memory > diffuser_ddp_bil_45M_40K_dmap_only_train.log 2>&1 &



# Training with 4 gpus
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=29500 run/train_diffusion_ddp.py \
--model_id 4 \
--epochs 1000 \
--save_step 500 \
--save_path /exhdd/seungyu/diffusion_motion/trained_weights/diffuser_bil_45m_100k_ddp_raster_only/run1 \
--config_path /exhdd/seungyu/diffusion_motion/core/diffuser/config/DDP/diffuser_bil_ddp.py \
--dataset_id 5 \
--batch_size 256 \
--num_workers 2 \
--pin_memory > diffuser_ddp_bil_45M_100K_raster_only_train.log 2>&1 &



CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=29500 run/train_diffusion_ddp.py \
--model_id 4 \
--epochs 1000 \
--save_step 500 \
--save_path /exhdd/seungyu/diffusion_motion/trained_weights/diffuser_bil_45m_40k_ddp_no_weight_sharing/run1 \
--config_path /exhdd/seungyu/diffusion_motion/core/diffuser/config/DDP/diffuser_bil_ddp.py \
--dataset_id 4 \
--batch_size 256 \
--num_workers 2 \
--pin_memory > diffuser_ddp_bil_45M_40K_no_weight_sharing.log 2>&1 &



CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 --master_port=29501 run/train_diffusion_ddp.py \
--model_id 4 \
--epochs 1000 \
--save_step 500 \
--save_path /exhdd/seungyu/diffusion_motion/trained_weights/diffuser_bil_45m_40k_ddp_no_weight_sharing_iter5/run1 \
--config_path /exhdd/seungyu/diffusion_motion/core/diffuser/config/DDP/diffuser_bil_ddp.py \
--dataset_id 4 \
--batch_size 256 \
--num_workers 2 \
--pin_memory > diffuser_ddp_bil_45M_40K_no_weight_sharing_iter5.log 2>&1 &


CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=29500 run/train_diffusion_ddp.py \
--model_id 4 \
--epochs 1000 \
--save_step 500 \
--save_path /exhdd/seungyu/diffusion_motion/trained_weights/diffuser_bil_45m_40k_ddp_no_weight_sharing_iter2/run1 \
--config_path /exhdd/seungyu/diffusion_motion/core/diffuser/config/DDP/diffuser_bil_ddp.py \
--dataset_id 4 \
--batch_size 256 \
--num_workers 2 \
--pin_memory > diffuser_ddp_bil_45M_40K_no_weight_sharing_iter2.log 2>&1 &



CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 --master_port=29501 run/train_diffusion_ddp.py \
--model_id 4 \
--epochs 1000 \
--save_step 500 \
--save_path /exhdd/seungyu/diffusion_motion/trained_weights/diffuser_bil_50m_100k_ddp_16x16_64h_raster_mask_only/run1 \
--config_path /exhdd/seungyu/diffusion_motion/core/diffuser/config/DDP/diffuser_bil_ddp.py \
--dataset_id 7 \
--batch_size 256 \
--num_workers 2 \
--pin_memory > diffuser_bil_50m_100k_ddp_16x16_64h_raster_mask_only_train.log 2>&1 &



CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=29500 run/train_diffusion_ddp.py \
--model_id 4 \
--epochs 1000 \
--save_step 500 \
--save_path /exhdd/seungyu/diffusion_motion/trained_weights/diffuser_bil_45m_40k_ddp_sigma_01/run1 \
--config_path /exhdd/seungyu/diffusion_motion/core/diffuser/config/DDP/diffuser_bil_ddp.py \
--dataset_id 4 \
--batch_size 256 \
--num_workers 2 \
--pin_memory > diffuser_ddp_bil_45M_40K_ddp_sigma_01_train.log 2>&1 &



CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 --master_port=29501 run/train_diffusion_ddp.py \
--model_id 4 \
--epochs 1000 \
--save_step 500 \
--save_path /exhdd/seungyu/diffusion_motion/trained_weights/diffuser_bil_56m_100k_ddp_32x32_128h_raster_mask_only/run1 \
--config_path /exhdd/seungyu/diffusion_motion/core/diffuser/config/DDP/diffuser_bil_ddp.py \
--dataset_id 8 \
--batch_size 256 \
--num_workers 2 \
--pin_memory > diffuser_bil_56m_100k_ddp_32x32_128h_raster_mask_only_train.log 2>&1 &


# Run CUDA_VISIBLE_DEVICES without close signal
# disown -h
CUDA_VISIBLE_DEVICES=4,5,6,7 nohup torchrun --nproc_per_node=4 --master_port=29501 run/train_diffusion_ddp.py \
--model_id 4 \
--epochs 1000 \
--save_step 500 \
--save_path /exhdd/seungyu/diffusion_motion/trained_weights/diffuser_bil_56m_100k_ddp_32x32_128h_raster_mask_only_heavy/run1 \
--config_path /exhdd/seungyu/diffusion_motion/core/diffuser/config/DDP/diffuser_bil_ddp.py \
--dataset_id 8 \
--batch_size 256 \
--num_workers 2 \
--pin_memory > diffuser_bil_56m_100k_ddp_32x32_128h_raster_mask_only_heavy_train.log 2>&1 < /dev/null &


# Run CUDA_VISIBLE_DEVICES without close signal
# disown -h
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup torchrun --nproc_per_node=4 --master_port=29500 run/train_diffusion_ddp.py \
--model_id 4 \
--epochs 1000 \
--save_step 500 \
--save_path /exhdd/seungyu/diffusion_motion/trained_weights/diffuser_bil_45m_8k_ddp_8x8_32h/run1 \
--config_path /exhdd/seungyu/diffusion_motion/core/diffuser/config/DDP/diffuser_bil_ddp.py \
--dataset_id 9 \
--batch_size 256 \
--num_workers 2 \
--pin_memory > diffuser_bil_45m_8k_ddp_8x8_32h.log 2>&1 < /dev/null &

