nohup python3 run/train_diffusion.py --model_id=9 \
    --epochs=1000 --save_step=500 --save_path=/exhdd/seungyu/diffusion_motion/trained_weights/diffuser_sim_fusion_feat128_2k \
    --config_path=/exhdd/seungyu/diffusion_motion/core/diffuser/config/diffuser_sim_fusion.py --device=cuda:0 --dataset_id=1 > diffuser_sim_fusion_feat128_2k_train.log 2>&1 &
