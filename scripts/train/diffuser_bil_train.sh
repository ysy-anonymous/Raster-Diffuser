nohup python3 run/train_diffusion.py --model_id=4 \
    --epochs=1000 --save_step=500 --save_path=/exhdd/seungyu/diffusion_motion/trained_weights/diffuser_bil_45m_2000 \
    --config_path=/exhdd/seungyu/diffusion_motion/core/diffuser/config/diffuser_bil.py --device=cuda:1 --dataset_id=1 > diffuser_bil_45m_2000_train.log 2>&1 &


nohup python3 run/train_diffusion.py --model_id=4 \
    --epochs=1000 --save_step=500 --save_path=/exhdd/seungyu/diffusion_motion/trained_weights/diffuser_bil_45m_2k_raster_only \
    --config_path=/exhdd/seungyu/diffusion_motion/core/diffuser/config/diffuser_bil.py --device=cuda:0 --dataset_id=1 > diffuser_bil_45m_2k_raster_only_train.log 2>&1 &