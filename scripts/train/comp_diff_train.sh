# Comp V2 2k training 
python3 run/train_comp_diffusion.py \
--model_id 1 \
--config_path /exhdd/seungyu/diffusion_motion/core/comp_diffusion/configs/comp_diffusion_cfg_v2.py \
--dataset_id 1 > comp_v2_2k_train.log 2>&1 &


# Comp 2k training
python3 run/train_comp_diffusion.py \
--model_id 0 \
--config_path /exhdd/seungyu/diffusion_motion/core/comp_diffusion/configs/comp_diffusion_cfg.py \
--dataset_id 1 > comp_diff_2k_train.log 2>&1 &