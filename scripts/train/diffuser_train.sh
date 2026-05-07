nohup python3 run/train_diffusion.py --model_id=3 \
    --epochs=1000 --save_step=500 --save_path=/exhdd/seungyu/diffusion_motion/trained_weights/diffuser_v2_45m_2000 \
    --config_path=/exhdd/seungyu/diffusion_motion/core/diffuser/config/diffuser_v2.py --device=cuda:2 --dataset_id=1 > diffuser_v2_45m_2000_train.log 2>&1 &
