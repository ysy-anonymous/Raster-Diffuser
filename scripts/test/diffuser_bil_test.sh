# nohup python3 run/test_diffusion.py --model_id=4 --ddp_trained --cp_path=/exhdd/seungyu/diffusion_motion/trained_weights/diffuser_bil_45m_2k_ddp/run1/ckpt_final.ckpt \
#     --device=cuda:1 --test_num=1000 --map_size=8 --num_vis=500 --vis_fname=diffuser_bil_ddp_45m_2k > test_results_log/diffuser_bil_ddp_45m_2k.log 2>&1 &

nohup python3 run/test_diffusion.py --model_id=4 --cp_path=/exhdd/seungyu/diffusion_motion/trained_weights/diffuser_bil_45m_2000/run1/ckpt_final.ckpt \
    --device=cuda:2 --test_num=1000 --map_size=8 --num_vis=500 --vis_fname=diffuser_bil_45m_2k > test_results_log/diffuser_bil_45m_2k.log 2>&1 &


nohup python3 run/test_diffusion.py --model_id=4 --cp_path=/exhdd/seungyu/diffusion_motion/trained_weights/diffuser_bil_50m_100k_ddp_16x16_64h_raster_mask_only/run1/ckpt_final.ckpt \
    --device=cuda:4 --test_num=1000 --map_size=8 --num_vis=1000 --data_seed=0 --test_seed=0 