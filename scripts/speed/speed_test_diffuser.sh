# For the test, Average FPS of 1000 samples from the test set is measured.

# Set the configuration (diffuser_bil_ddp.py) to have same model settings. 
nohup python3 run/speed_test_diffuser.py --model_id=4 --ddp_trained --cp_path=/exhdd/seungyu/diffusion_motion/trained_weights/diffuser_bil_45m_40k_ddp_no_weight_sharing_iter5/run1/ckpt_final.ckpt \
    --device=cuda:6 --test_path=/exhdd/seungyu/diffusion_motion/dataset/test_scenarios_20000.npy --num_warmup=10 --num_eval=10 > test_speeds_log/diffuser_bil_ddp_45m_40k_no_weight_sharing_iter5.log 2>&1 &