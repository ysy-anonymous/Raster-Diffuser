# 1. First Test (V)
# Set the configuration (diffuser_bil_ddp.py) to have same model settings. 
nohup python3 run/test_diffuser_offline.py --model_id=4 --ddp_trained --cp_path=/exhdd/seungyu/diffusion_motion/trained_weights/diffuser_bil_45m_40k_ddp/run1/ckpt_final.ckpt \
    --device=cuda:0 --num_vis=500 --vis_fname=diffuser_bil_ddp_45m_40k_offline \
    --test_path=/exhdd/seungyu/diffusion_motion/dataset/test_scenarios_20000.npy --test_bs=64 > test_results_log/diffuser_bil_ddp_45m_40k_offline.log 2>&1 &

# 2. Second Test (V)
# Set the configuration (diffuser_bil_ddp.py) to have same model settings. 
nohup python3 run/test_diffuser_offline.py --model_id=4 --ddp_trained --cp_path=/exhdd/seungyu/diffusion_motion/trained_weights/diffuser_bil_45m_40k_ddp_bdim_128/run1/ckpt_final.ckpt \
    --device=cuda:0 --num_vis=500 --vis_fname=diffuser_bil_45m_40k_bdim_128_offline \
    --test_path=/exhdd/seungyu/diffusion_motion/dataset/test_scenarios_20000.npy --test_bs=64 > test_results_log/diffuser_bil_45m_40k_bdim_128_offline.log 2>&1 &

# 3. Third Test (V)
# Set the configuration (diffuser_bil_ddp.py) to have same model settings. Reduce the batch size to 32 to prevent OOM.
nohup python3 run/test_diffuser_offline.py --model_id=4 --ddp_trained --cp_path=/exhdd/seungyu/diffusion_motion/trained_weights/diffuser_bil_45m_40k_ddp_bdim_256/run1/ckpt_final.ckpt \
    --device=cuda:1 --num_vis=500 --vis_fname=diffuser_bil_45m_40k_ddp_bdim_256_offline \
    --test_path=/exhdd/seungyu/diffusion_motion/dataset/test_scenarios_20000.npy --test_bs=32 > test_results_log/diffuser_bil_45m_40k_ddp_bdim_256_offline.log 2>&1 &

# 4. Fourth Test (V)
# Set the configuration (diffuser_bil_ddp.py) to have same model settings. Reduce the batch size to 32 to prevent OOM.
nohup python3 run/test_diffuser_offline.py --model_id=4 --ddp_trained --cp_path=/exhdd/seungyu/diffusion_motion/trained_weights/diffuser_bil_45m_40k_ddp_bdim_384/run1/ckpt_final.ckpt \
    --device=cuda:2 --num_vis=500 --vis_fname=diffuser_bil_45m_40k_ddp_bdim_384_offline \
    --test_path=/exhdd/seungyu/diffusion_motion/dataset/test_scenarios_20000.npy --test_bs=32 > test_results_log/diffuser_bil_45m_40k_ddp_bdim_384_offline.log 2>&1 &

# 5. Fifth Test (V)
# Set the configuration (diffuser_bil_ddp.py) to have same model settings. 
nohup python3 run/test_diffuser_offline.py --model_id=4 --ddp_trained --cp_path=/exhdd/seungyu/diffusion_motion/trained_weights/diffuser_bil_45m_40k_ddp_half_sigma/run1/ckpt_final.ckpt \
    --device=cuda:3 --num_vis=500 --vis_fname=diffuser_bil_45m_40k_ddp_half_sigma_offline \
    --test_path=/exhdd/seungyu/diffusion_motion/dataset/test_scenarios_20000.npy --test_bs=64 > test_results_log/diffuser_bil_45m_40k_ddp_half_sigma_offline.log 2>&1 &

# 6. Sixth Test (V)
# Set the configuration (diffuser_bil_ddp.py) to have same model settings. 
nohup python3 run/test_diffuser_offline.py --model_id=4 --ddp_trained --cp_path=/exhdd/seungyu/diffusion_motion/trained_weights/diffuser_bil_45m_40k_ddp_sigma_plus/run1/ckpt_final.ckpt \
    --device=cuda:4 --num_vis=500 --vis_fname=diffuser_bil_45m_40k_ddp_sigma_plus_offline \
    --test_path=/exhdd/seungyu/diffusion_motion/dataset/test_scenarios_20000.npy --test_bs=64 > test_results_log/diffuser_bil_45m_40k_ddp_sigma_plus_offline.log 2>&1 &

# 7. Seventh Test (V)
# Set the configuration (diffuser_bil_ddp.py) to have same model settings. 
nohup python3 run/test_diffuser_offline.py --model_id=4 --ddp_trained --cp_path=/exhdd/seungyu/diffusion_motion/trained_weights/diffuser_bil_45m_40k_ddp_iter_1/run1/ckpt_final.ckpt \
    --device=cuda:5 --num_vis=500 --vis_fname=diffuser_bil_45m_40k_ddp_iter_1_offline \
    --test_path=/exhdd/seungyu/diffusion_motion/dataset/test_scenarios_20000.npy --test_bs=64 > test_results_log/diffuser_bil_45m_40k_ddp_iter_1_offline.log 2>&1 &

# 8. Eighth Test (V)
# Set the configuration (diffuser_bil_ddp.py) to have same model settings. 
nohup python3 run/test_diffuser_offline.py --model_id=4 --ddp_trained --cp_path=/exhdd/seungyu/diffusion_motion/trained_weights/diffuser_bil_45m_40k_ddp_iter_5/run1/ckpt_final.ckpt \
    --device=cuda:6 --num_vis=500 --vis_fname=diffuser_bil_45m_40k_ddp_iter_5_offline \
    --test_path=/exhdd/seungyu/diffusion_motion/dataset/test_scenarios_20000.npy --test_bs=64 > test_results_log/diffuser_bil_45m_40k_ddp_iter_5_offline.log 2>&1 &

# 9. Ninth Test (V)
# Set the configuration (diffuser_bil_ddp.py) to have same model settings. Reduce the batch size to 32 to prevent OOM.
nohup python3 run/test_diffuser_offline.py --model_id=4 --ddp_trained --cp_path=/exhdd/seungyu/diffusion_motion/trained_weights/diffuser_bil_45m_40k_ddp_upsample_2x/run1/ckpt_final.ckpt \
    --device=cuda:7 --num_vis=500 --vis_fname=diffuser_bil_45m_40k_ddp_upsample_2x_offline \
    --test_path=/exhdd/seungyu/diffusion_motion/dataset/test_scenarios_20000.npy --test_bs=32 > test_results_log/diffuser_bil_45m_40k_ddp_upsample_2x_offline.log 2>&1 &

# 10. Tenth Test (V)
# Set the configuration (diffuser_bil_ddp.py) to have same model settings. 
nohup python3 run/test_diffuser_offline.py --model_id=4 --ddp_trained --cp_path=/exhdd/seungyu/diffusion_motion/trained_weights/diffuser_bil_45m_40k_ddp_upsample_4x/run1/ckpt_final.ckpt \
    --device=cuda:0 --num_vis=500 --vis_fname=diffuser_bil_45m_40k_ddp_upsample_4x_offline \
    --test_path=/exhdd/seungyu/diffusion_motion/dataset/test_scenarios_20000.npy --test_bs=16 > test_results_log/diffuser_bil_45m_40k_ddp_upsample_4x_offline.log 2>&1 &

# 11. Eleventh Test (V)
# Set the configuration (diffuser_bil_ddp.py) to have same model settings. 
nohup python3 run/test_diffuser_offline.py --model_id=4 --ddp_trained --cp_path=/exhdd/seungyu/diffusion_motion/trained_weights/diffuser_bil_45m_100k_ddp/run1/ckpt_final.ckpt \
    --device=cuda:1 --num_vis=500 --vis_fname=diffuser_bil_45m_100k_ddp_offline \
    --test_path=/exhdd/seungyu/diffusion_motion/dataset/test_scenarios_20000.npy --test_bs=64 > test_results_log/diffuser_bil_45m_100k_ddp_offline.log 2>&1 &

# 12. Twelveth Test (V)
# Set the configuration (diffuser_bil_ddp.py) to have same model settings. 
nohup python3 run/test_diffuser_offline.py --model_id=4 --cp_path=/exhdd/seungyu/diffusion_motion/trained_weights/diffuser_bil_45m_2000/run1/ckpt_final.ckpt \
    --device=cuda:2 --num_vis=500 --vis_fname=diffuser_bil_45m_2k_offline \
    --test_path=/exhdd/seungyu/diffusion_motion/dataset/test_scenarios_20000.npy --test_bs=64 > test_results_log/diffuser_bil_45m_2k_offline.log 2>&1 &

# 13. Thirteenth Test (V)
# Set the configuration (diffuser_ddp.py) to have same model settings. 
nohup python3 run/test_diffuser_offline.py --model_id=3 --ddp_trained --cp_path=/exhdd/seungyu/diffusion_motion/trained_weights/diffuser_ddp_45m_40k/run1/ckpt_final.ckpt \
    --device=cuda:6 --num_vis=500 --vis_fname=diffuser_ddp_45m_40k_offline \
    --test_path=/exhdd/seungyu/diffusion_motion/dataset/test_scenarios_20000.npy --test_bs=64 > test_results_log/diffuser_ddp_45m_40k_offline.log 2>&1 &

# 14. Fourteenth Test (V)
# Set the configuration (diffuser_ddp.py) to have same model settings. 
nohup python3 run/test_diffuser_offline.py --model_id=3 --ddp_trained --cp_path=/exhdd/seungyu/diffusion_motion/trained_weights/diffuser_ddp_45m_100k/run1/ckpt_final.ckpt \
    --device=cuda:7 --num_vis=500 --vis_fname=diffuser_ddp_45m_100k_offline \
    --test_path=/exhdd/seungyu/diffusion_motion/dataset/test_scenarios_20000.npy --test_bs=64 > test_results_log/diffuser_ddp_45m_100k_offline.log 2>&1 &

# 15. Fifteenth Test (V)
# Set the configuration (diffuser_ddp.py) to have same model settings. 
nohup python3 run/test_diffuser_offline.py --model_id=3 --cp_path=/exhdd/seungyu/diffusion_motion/trained_weights/diffuser_ddp_45m_2k/run1/ckpt_final.ckpt \
    --device=cuda:0 --num_vis=500 --vis_fname=diffuser_ddp_45m_2k_offline \
    --test_path=/exhdd/seungyu/diffusion_motion/dataset/test_scenarios_20000.npy --test_bs=64 > test_results_log/diffuser_ddp_45m_2k_offline.log 2>&1 &


# 16. Sixteenth Test (V)
# Set the configuration (diffuser_bil_ddp.py) to have intented model settings.
nohup python3 run/test_diffuser_offline.py --model_id=4 --ddp_trained --cp_path=/exhdd/seungyu/diffusion_motion/trained_weights/diffuser_bil_45m_40k_ddp_iter_2/run1/ckpt_final.ckpt \
    --device=cuda:4 --num_vis=500 --vis_fname=diffuser_bil_45m_40k_ddp_iter_2_offline \
    --test_path=/exhdd/seungyu/diffusion_motion/dataset/test_scenarios_20000.npy --test_bs=64 > test_results_log/diffuser_bil_45m_40k_ddp_iter_2_offline.log 2>&1 &


# 18. Seventeenth Test (V)
# Set the configuration (DDP/diffuser_sim_ddp.py) to have intended model settings.
nohup python3 run/test_diffuser_offline.py --model_id=9 --ddp_trained --cp_path=/exhdd/seungyu/diffusion_motion/trained_weights/diffuser_sim_fusion_ddp_feat128_40k/run1/ckpt_final.ckpt \
    --device=cuda:1 --num_vis=500 --vis_fname=diffuser_sim_fusion_ddp_feat128_40k_offline \
    --test_path=/exhdd/seungyu/diffusion_motion/dataset/test_scenarios_20000.npy --test_bs=64 > test_results_log/diffuser_sim_fusion_ddp_feat128_40k_offline.log 2>&1 &

# 19. Nineteenth Test(V)
# Set the configuration (DDP/diffuser_sim_ddp.py) to have intended model settings.
nohup python3 run/test_diffuser_offline.py --model_id=9 --ddp_trained --cp_path=/exhdd/seungyu/diffusion_motion/trained_weights/diffuser_sim_fusion_ddp_feat128_100k/run1/ckpt_final.ckpt \
    --device=cuda:1 --num_vis=500 --vis_fname=diffuser_sim_fusion_ddp_feat128_100k_offline \
    --test_path=/exhdd/seungyu/diffusion_motion/dataset/test_scenarios_20000.npy --test_bs=64 > test_results_log/diffuser_sim_fusion_ddp_feat128_100k_offline.log 2>&1 &

# 20. twenthy Test(V)
# Set the configuration (diffuser_sim_fusion.py) to have intended model settings.
nohup python3 run/test_diffuser_offline.py --model_id=9 --cp_path=/exhdd/seungyu/diffusion_motion/trained_weights/diffuser_sim_fusion_feat128_2k/run1/ckpt_final.ckpt \
    --device=cuda:1 --num_vis=500 --vis_fname=diffuser_sim_fusion_feat128_2k_offline \
    --test_path=/exhdd/seungyu/diffusion_motion/dataset/test_scenarios_20000.npy --test_bs=64 > test_results_log/diffuser_sim_fusion_feat128_2k_offline.log 2>&1 &


# 21. twenthy First Test (V)
# Set the configuration (diffuser_bil_ddp.py) to have intented model settings.
nohup python3 run/test_diffuser_offline.py --model_id=4 --ddp_trained --cp_path=/exhdd/seungyu/diffusion_motion/trained_weights/diffuser_bil_45m_40k_ddp_raster_only_no_distmap/run1/ckpt_final.ckpt \
    --device=cuda:4 --num_vis=500 --vis_fname=diffuser_bil_45m_40k_ddp_raster_only_no_distmap_offline \
    --test_path=/exhdd/seungyu/diffusion_motion/dataset/test_scenarios_20000.npy --test_bs=64 > test_results_log/diffuser_bil_45m_40k_ddp_raster_only_no_distmap_offline.log 2>&1 &

# 22. twenthy Second test (V)
nohup python3 run/test_diffuser_offline.py --model_id=4 --ddp_trained --cp_path=/exhdd/seungyu/diffusion_motion/trained_weights/diffuser_bil_45m_40k_ddp_raster_only/run1/ckpt_final.ckpt \
    --device=cuda:5 --num_vis=500 --vis_fname=diffuser_bil_45m_40k_ddp_raster_only_offline \
    --test_path=/exhdd/seungyu/diffusion_motion/dataset/test_scenarios_20000.npy --test_bs=64 > test_results_log/diffuser_bil_45m_40k_ddp_raster_only_offline.log 2>&1 &


# 23. twenthy Third Test (V)
# Set the configuration (diffuser_bil_ddp.py) to have intented model settings.
nohup python3 run/test_diffuser_offline.py --model_id=4 --ddp_trained --cp_path=/exhdd/seungyu/diffusion_motion/trained_weights/diffuser_bil_45m_40k_ddp_upcoeff_05/run1/ckpt_final.ckpt \
    --device=cuda:4 --num_vis=500 --vis_fname=diffuser_bil_45m_40k_ddp_upcoeff_05_offline \
    --test_path=/exhdd/seungyu/diffusion_motion/dataset/test_scenarios_20000.npy --test_bs=64 > test_results_log/diffuser_bil_45m_40k_ddp_upcoeff_05_offline.log 2>&1 &

# 24. twenthy Fourth test (V)
nohup python3 run/test_diffuser_offline.py --model_id=4 --ddp_trained --cp_path=/exhdd/seungyu/diffusion_motion/trained_weights/diffuser_bil_45m_40k_ddp_upcoeff_10/run1/ckpt_final.ckpt \
    --device=cuda:5 --num_vis=500 --vis_fname=diffuser_bil_45m_40k_dpp_upcoeff_10_offline \
    --test_path=/exhdd/seungyu/diffusion_motion/dataset/test_scenarios_20000.npy --test_bs=64 > test_results_log/diffuser_bil_45m_40k_dpp_upcoeff_10_offline.log 2>&1 &

# 25. twenthy Fifth test (V)
nohup python3 run/test_diffuser_offline.py --model_id=4 --ddp_trained --cp_path=/exhdd/seungyu/diffusion_motion/trained_weights/diffuser_bil_45m_40k_ddp_upcoeff_learnable/run1/ckpt_final.ckpt \
    --device=cuda:5 --num_vis=500 --vis_fname=diffuser_bil_45m_40k_dpp_upcoeff_learnable_offline \
    --test_path=/exhdd/seungyu/diffusion_motion/dataset/test_scenarios_20000.npy --test_bs=64 > test_results_log/diffuser_bil_45m_40k_dpp_upcoeff_learnable_offline.log 2>&1 &

# 26. twenthy Sixth test (V)
nohup python3 run/test_diffuser_offline.py --model_id=4 --ddp_trained --cp_path=/exhdd/seungyu/diffusion_motion/trained_weights/diffuser_bil_45m_40k_ddp_sdf_only/run1/ckpt_final.ckpt \
    --device=cuda:6 --num_vis=500 --vis_fname=diffuser_bil_45m_40k_dpp_sdf_only_offline \
    --test_path=/exhdd/seungyu/diffusion_motion/dataset/test_scenarios_20000.npy --test_bs=64 > test_results_log/diffuser_bil_45m_40k_dpp_sdf_only_offline.log 2>&1 &

# 27. twenthy Seventh test (V)
nohup python3 run/test_diffuser_offline.py --model_id=4 --ddp_trained --cp_path=/exhdd/seungyu/diffusion_motion/trained_weights/diffuser_bil_45m_40k_ddp_bmask_only/run1/ckpt_final.ckpt \
    --device=cuda:4 --num_vis=500 --vis_fname=diffuser_bil_45m_40k_dpp_bmask_only_offline \
    --test_path=/exhdd/seungyu/diffusion_motion/dataset/test_scenarios_20000.npy --test_bs=64 > test_results_log/diffuser_bil_45m_40k_dpp_bmask_only_offline.log 2>&1 &
    
# 28. twenthy Eighth test (V)
nohup python3 run/test_diffuser_offline.py --model_id=4 --ddp_trained --cp_path=/exhdd/seungyu/diffusion_motion/trained_weights/diffuser_bil_45m_40k_ddp_dmap_only/run1/ckpt_final.ckpt \
    --device=cuda:7 --num_vis=500 --vis_fname=diffuser_bil_45m_40k_dpp_dmap_only_offline \
    --test_path=/exhdd/seungyu/diffusion_motion/dataset/test_scenarios_20000.npy --test_bs=64 > test_results_log/diffuser_bil_45m_40k_dpp_dmap_only_offline.log 2>&1 &

# 29. twenthy Nineth test (V)
nohup python3 run/test_diffuser_offline.py --model_id=4 --ddp_trained --cp_path=/exhdd/seungyu/diffusion_motion/trained_weights/diffuser_bil_45m_100k_ddp_raster_only/run1/ckpt_final.ckpt \
    --device=cuda:1 --num_vis=500 --vis_fname=diffuser_bil_45m_100k_dpp_raster_only_offline \
    --test_path=/exhdd/seungyu/diffusion_motion/dataset/test_scenarios_20000.npy --test_bs=64 > test_results_log/diffuser_bil_45m_100k_dpp_raster_only_offline.log 2>&1 &


# 30. Thirteen test (V)
nohup python3 run/test_diffuser_offline.py --model_id=4 --cp_path=/exhdd/seungyu/diffusion_motion/trained_weights/diffuser_bil_45m_2k_raster_only/run1/ckpt_final.ckpt \
    --device=cuda:2 --num_vis=500 --vis_fname=diffuser_bil_45m_2k_raster_only_offline \
    --test_path=/exhdd/seungyu/diffusion_motion/dataset/test_scenarios_20000.npy --test_bs=64 > test_results_log/diffuser_bil_45m_2k_raster_only_offline.log 2>&1 &

# 31. Thirty first test (V)
nohup python3 run/test_diffuser_offline.py --model_id=4 --ddp_trained --cp_path=/exhdd/seungyu/diffusion_motion/trained_weights/diffuser_bil_45m_40k_ddp_no_weight_sharing/run1/ckpt_final.ckpt \
    --device=cuda:4 --num_vis=500 --vis_fname=diffuser_bil_45m_40k_dpp_no_weight_sharing_offline \
    --test_path=/exhdd/seungyu/diffusion_motion/dataset/test_scenarios_20000.npy --test_bs=64 > test_results_log/diffuser_bil_45m_40k_dpp_no_weight_sharing_offline.log 2>&1 &

# 32. Thirty two test (V)
nohup python3 run/test_diffuser_offline.py --model_id=4 --ddp_trained --cp_path=/exhdd/seungyu/diffusion_motion/trained_weights/diffuser_bil_45m_40k_ddp_no_weight_sharing_iter5/run1/ckpt_final.ckpt \
    --device=cuda:5 --num_vis=500 --vis_fname=diffuser_bil_45m_40k_dpp_no_weight_sharing_iter_5_offline \
    --test_path=/exhdd/seungyu/diffusion_motion/dataset/test_scenarios_20000.npy --test_bs=64 > test_results_log/diffuser_bil_45m_40k_dpp_no_weight_sharing_iter_5_offline.log 2>&1 &

# 33. Thirty three test (V)
nohup python3 run/test_diffuser_offline.py --model_id=4 --ddp_trained --cp_path=/exhdd/seungyu/diffusion_motion/trained_weights/diffuser_bil_45m_40k_ddp_no_weight_sharing_iter2/run1/ckpt_final.ckpt \
    --device=cuda:0 --num_vis=500 --vis_fname=diffuser_bil_45m_40k_dpp_no_weight_sharing_iter_2_offline \
    --test_path=/exhdd/seungyu/diffusion_motion/dataset/test_scenarios_20000.npy --test_bs=64 > test_results_log/diffuser_bil_45m_40k_dpp_no_weight_sharing_iter_2_offline.log 2>&1 &

# 34. Thirty Fourth test (V)
# Change the --cp_path name according to the name below.
nohup python3 run/test_diffuser_offline.py --model_id=4 --ddp_trained --cp_path=/exhdd/seungyu/diffusion_motion/trained_weights/diffuser_bil_50m_100k_ddp_16x16_64h_raster_mask_only/run1/ckpt_final.ckpt \
    --device=cuda:0 --num_vis=500 --vis_fname=diffuser_bil_50m_100k_ddp_16x16_64h_raster_mask_only_offline \
    --test_path=/exhdd/seungyu/diffusion_motion/dataset/test_scenarios_20000_16x16_64h.npy --test_bs=64 > test_results_log/diffuser_bil_50m_100k_ddp_16x16_64h_raster_mask_only_offline.log 2>&1 &

# 35. Thirty Five test (V)
nohup python3 run/test_diffuser_offline.py --model_id=3 --ddp_trained --cp_path=/exhdd/seungyu/diffusion_motion/trained_weights/diffuser_50m_100k_ddp_16x16_64h_mask_only/run1/ckpt_final.ckpt \
    --device=cuda:1 --num_vis=500 --vis_fname=diffuser_50m_100k_ddp_16x16_64h_mask_only_offline \
    --test_path=/exhdd/seungyu/diffusion_motion/dataset/test_scenarios_20000_16x16_64h.npy --test_bs=64 > test_results_log/diffuser_50m_100k_ddp_16x16_64h_mask_only_offline.log 2>&1 &


# 36. Thirty Six test (V)
# Raster-Diffuser with soft rasterization close to hard rasterization.
nohup python3 run/test_diffuser_offline.py --model_id=4 --ddp_trained --cp_path=/exhdd/seungyu/diffusion_motion/trained_weights/diffuser_bil_45m_40k_ddp_sigma_01/run1/ckpt_final.ckpt \
    --device=cuda:0 --num_vis=500 --vis_fname=diffuser_bil_45m_40k_ddp_sigma_01_offline \
    --test_path=/exhdd/seungyu/diffusion_motion/dataset/test_scenarios_20000.npy --test_bs=64 > test_results_log/diffuser_bil_45m_40k_ddp_sigma_01_offline.log 2>&1 &


# Raster-Diffuser Visualization Purpose
nohup python3 run/test_diffuser_offline.py --model_id=4 --ddp_trained --cp_path=/exhdd/seungyu/diffusion_motion/trained_weights/diffuser_bil_50m_100k_ddp_16x16_64h_raster_mask_only/run1/ckpt_final.ckpt \
    --device=cuda:6 --num_vis=1000 --vis_fname=diffuser_bil_50m_100k_ddp_16x16_64h_raster_mask_only_vis \
    --test_path=/exhdd/seungyu/diffusion_motion/dataset/test_scenarios_20000_16x16_64h.npy --test_bs=100 > test_results_log/diffuser_bil_50m_100k_ddp_16x16_64h_raster_mask_only_vis.log 2>&1 &


# Diffuser Visualization Purpose
nohup python3 run/test_diffuser_offline.py --model_id=3 --ddp_trained --cp_path=/exhdd/seungyu/diffusion_motion/trained_weights/diffuser_50m_100k_ddp_16x16_64h_mask_only/run1/ckpt_final.ckpt \
    --device=cuda:7 --num_vis=1000 --vis_fname=diffuser_50m_100k_ddp_16x16_64h_mask_only_vis \
    --test_path=/exhdd/seungyu/diffusion_motion/dataset/test_scenarios_20000_16x16_64h.npy --test_bs=100 > test_results_log/diffuser_50m_100k_ddp_16x16_64h_mask_only_vis.log 2>&1 &


# 37. Thirty Seventh Test (V)
nohup python3 run/test_diffuser_offline.py --model_id=4 --ddp_trained --cp_path=/exhdd/seungyu/diffusion_motion/trained_weights/diffuser_bil_56m_100k_ddp_32x32_128h_raster_mask_only/run1/ckpt_final.ckpt \
    --device=cuda:1 --num_vis=500 --vis_fname=diffuser_bil_56m_100k_ddp_32x32_128h_raster_mask_only_offline \
    --test_path=/exhdd/seungyu/diffusion_motion/dataset/test_scenarios_30000_32x32_128h.npy --test_bs=256 > test_results_log/diffuser_bil_56m_100k_ddp_32x32_128h_raster_mask_only_offline.log 2>&1 &


# 38. Thirty Eighth Test (V)
nohup python3 run/test_diffuser_offline.py --model_id=3 --ddp_trained --cp_path=/exhdd/seungyu/diffusion_motion/trained_weights/diffuser_53m_100k_ddp_32x32_128h_mask_only/run1/ckpt_final.ckpt \
    --device=cuda:0 --num_vis=500 --vis_fname=diffuser_53m_100k_ddp_32x32_128h_mask_only_offline \
    --test_path=/exhdd/seungyu/diffusion_motion/dataset/test_scenarios_30000_32x32_128h.npy --test_bs=256 > test_results_log/diffuser_53m_100k_ddp_32x32_128h_mask_only_offline.log 2>&1 &


# Raster-Diffuser 32x32 map Visualization Purpose
# Change to the sub-dataloader (fast_dataloader)
nohup python3 run/test_diffuser_offline.py --model_id=4 --ddp_trained --cp_path=/exhdd/seungyu/diffusion_motion/trained_weights/diffuser_bil_56m_100k_ddp_32x32_128h_raster_mask_only/run1/ckpt_final.ckpt \
    --device=cuda:0 --num_vis=1000 --vis_fname=diffuser_bil_56m_100k_ddp_32x32_128h_raster_mask_only_vis \
    --test_path=/exhdd/seungyu/diffusion_motion/dataset/test_scenarios_30000_32x32_128h.npy --test_bs=100 > test_results_log/diffuser_bil_56m_100k_ddp_32x32_128h_raster_mask_only_vis.log 2>&1 &


# Diffuser 32x32 map Visualization Purpose
# Change to the sub-dataloader (fast_dataloader)
nohup python3 run/test_diffuser_offline.py --model_id=3 --ddp_trained --cp_path=/exhdd/seungyu/diffusion_motion/trained_weights/diffuser_53m_100k_ddp_32x32_128h_mask_only/run1/ckpt_final.ckpt \
    --device=cuda:2 --num_vis=1000 --vis_fname=diffuser_53m_100k_ddp_32x32_128h_mask_only_vis \
    --test_path=/exhdd/seungyu/diffusion_motion/dataset/test_scenarios_30000_32x32_128h.npy --test_bs=100 > test_results_log/diffuser_53m_100k_ddp_32x32_128h_raster_mask_only_vis.log 2>&1 &


# 39. Thirty Nine Test (V)
nohup python3 run/test_diffuser_offline.py --model_id=4 --ddp_trained --cp_path=/exhdd/seungyu/diffusion_motion/trained_weights/diffuser_bil_56m_100k_ddp_32x32_128h_raster_mask_only_heavy/run1/ckpt_final.ckpt \
    --device=cuda:2 --num_vis=500 --vis_fname=diffuser_bil_56m_100k_ddp_32x32_128h_raster_mask_only_heavy_offline \
    --test_path=/exhdd/seungyu/diffusion_motion/dataset/test_scenarios_30000_32x32_128h.npy --test_bs=1024 > test_results_log/diffuser_bil_56m_100k_ddp_32x32_128h_raster_mask_only_heavy_offline.log 2>&1 &


# Diffuser 32x32 map Visualization Purpose
# Change to the sub-dataloader (fast_dataloader)
nohup python3 run/test_diffuser_offline.py --model_id=4 --ddp_trained --cp_path=/exhdd/seungyu/diffusion_motion/trained_weights/diffuser_bil_56m_100k_ddp_32x32_128h_raster_mask_only_heavy/run1/ckpt_final.ckpt \
    --device=cuda:1 --num_vis=1000 --vis_fname=diffuser_bil_56m_100k_ddp_32x32_128h_raster_mask_only_heavy_vis \
    --test_path=/exhdd/seungyu/diffusion_motion/dataset/test_scenarios_30000_32x32_128h.npy --test_bs=100 > test_results_log/diffuser_bil_56m_100k_ddp_32x32_128h_raster_mask_only_heavy_vis.log 2>&1 &


# Diffuser 8x8 map 8K
# 40. Forthy Test (-)
nohup python3 run/test_diffuser_offline.py --model_id=4 --ddp_trained --cp_path=/exhdd/seungyu/diffusion_motion/trained_weights/diffuser_bil_45m_8k_ddp_8x8_32h/run1/ckpt_final.ckpt \
    --device=cuda:0 --num_vis=500 --vis_fname=diffuser_bil_45m_8k_ddp_8x8_32h_offline \
    --test_path=/exhdd/seungyu/diffusion_motion/dataset/test_scenarios_20000.npy --test_bs=1024 > test_results_log/diffuser_bil_45m_8k_ddp_8x8_32h_offline.log 2>&1 &


################################################################################################
#           Random Seed Test for Failure Cases on 16x16 Raster-Diffuser Settings               #
################################################################################################
# For correcting, failure cases on 16x16 Raster-Diffuser settings
# Default start seed as 0
# seed list: [0, 1, 4, 8, 16, 24, 32, 67, 112, 176, 227, 234, 267, 341]
CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 run/test_diffuser_offline.py --model_id=4 --ddp_trained --cp_path=/exhdd/seungyu/diffusion_motion/trained_weights/diffuser_bil_50m_100k_ddp_16x16_64h_raster_mask_only/run1/ckpt_final.ckpt \
    --device=cuda:0 --num_vis=1000 --vis_fname=random_seed_log/seed0_16x16_vis \
    --test_path=/exhdd/seungyu/diffusion_motion/dataset/test_scenarios_20000_16x16_64h.npy --test_bs=256 --strict_seed --seed=0 > random_seed_log/seed0_16x16_log 2>&1 &

# Now we need to evaluate on the failure cases again. Visualize the all the failure cases or some of it.
# Check "num_vis"
CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 run/test_diffuser_offline.py --model_id=4 --ddp_trained --cp_path=/exhdd/seungyu/diffusion_motion/trained_weights/diffuser_bil_50m_100k_ddp_16x16_64h_raster_mask_only/run1/ckpt_final.ckpt \
    --device=cuda:0 --num_vis=1416 --vis_fname=random_seed_log/failure_case_seed0_vis \
    --test_path=/exhdd/seungyu/diffusion_motion/dataset/test_scenarios_20000_16x16_64h.npy --test_bs=256 --strict_seed --seed=0 --test_failure_case > random_seed_log/failure_case_seed0.log 2>&1 &


# Now We try different seeds: 1, 4, 8, 16, 24, 32, 67, 112, 176, 227
CUBLAS_WORKSPACE_CONFIG=:4096:8 nohup python3 run/test_diffuser_offline.py --model_id=4 --ddp_trained --cp_path=/exhdd/seungyu/diffusion_motion/trained_weights/diffuser_bil_50m_100k_ddp_16x16_64h_raster_mask_only/run1/ckpt_final.ckpt \
    --device=cuda:1 --num_vis=1416 --vis_fname=random_seed_log/failure_case_seed1_vis \
    --test_path=/exhdd/seungyu/diffusion_motion/dataset/test_scenarios_20000_16x16_64h.npy --test_bs=256 --strict_seed --seed=1 --test_failure_case > random_seed_log/failure_case_seed1.log 2>&1 &

CUBLAS_WORKSPACE_CONFIG=:4096:8 nohup python3 run/test_diffuser_offline.py --model_id=4 --ddp_trained --cp_path=/exhdd/seungyu/diffusion_motion/trained_weights/diffuser_bil_50m_100k_ddp_16x16_64h_raster_mask_only/run1/ckpt_final.ckpt \
    --device=cuda:2 --num_vis=1416 --vis_fname=random_seed_log/failure_case_seed4_vis \
    --test_path=/exhdd/seungyu/diffusion_motion/dataset/test_scenarios_20000_16x16_64h.npy --test_bs=256 --strict_seed --seed=4 --test_failure_case > random_seed_log/failure_case_seed4.log 2>&1 &

CUBLAS_WORKSPACE_CONFIG=:4096:8 nohup python3 run/test_diffuser_offline.py --model_id=4 --ddp_trained --cp_path=/exhdd/seungyu/diffusion_motion/trained_weights/diffuser_bil_50m_100k_ddp_16x16_64h_raster_mask_only/run1/ckpt_final.ckpt \
    --device=cuda:3 --num_vis=1416 --vis_fname=random_seed_log/failure_case_seed8_vis \
    --test_path=/exhdd/seungyu/diffusion_motion/dataset/test_scenarios_20000_16x16_64h.npy --test_bs=256 --strict_seed --seed=8 --test_failure_case > random_seed_log/failure_case_seed8.log 2>&1 &

CUBLAS_WORKSPACE_CONFIG=:4096:8 nohup python3 run/test_diffuser_offline.py --model_id=4 --ddp_trained --cp_path=/exhdd/seungyu/diffusion_motion/trained_weights/diffuser_bil_50m_100k_ddp_16x16_64h_raster_mask_only/run1/ckpt_final.ckpt \
    --device=cuda:4 --num_vis=1416 --vis_fname=random_seed_log/failure_case_seed16_vis \
    --test_path=/exhdd/seungyu/diffusion_motion/dataset/test_scenarios_20000_16x16_64h.npy --test_bs=256 --strict_seed --seed=16 --test_failure_case > random_seed_log/failure_case_seed16.log 2>&1 &

CUBLAS_WORKSPACE_CONFIG=:4096:8 nohup python3 run/test_diffuser_offline.py --model_id=4 --ddp_trained --cp_path=/exhdd/seungyu/diffusion_motion/trained_weights/diffuser_bil_50m_100k_ddp_16x16_64h_raster_mask_only/run1/ckpt_final.ckpt \
    --device=cuda:5 --num_vis=1416 --vis_fname=random_seed_log/failure_case_seed24_vis \
    --test_path=/exhdd/seungyu/diffusion_motion/dataset/test_scenarios_20000_16x16_64h.npy --test_bs=256 --strict_seed --seed=24 --test_failure_case > random_seed_log/failure_case_seed24.log 2>&1 &

CUBLAS_WORKSPACE_CONFIG=:4096:8 nohup python3 run/test_diffuser_offline.py --model_id=4 --ddp_trained --cp_path=/exhdd/seungyu/diffusion_motion/trained_weights/diffuser_bil_50m_100k_ddp_16x16_64h_raster_mask_only/run1/ckpt_final.ckpt \
    --device=cuda:6 --num_vis=1416 --vis_fname=random_seed_log/failure_case_seed32_vis \
    --test_path=/exhdd/seungyu/diffusion_motion/dataset/test_scenarios_20000_16x16_64h.npy --test_bs=256 --strict_seed --seed=32 --test_failure_case > random_seed_log/failure_case_seed32.log 2>&1 &

CUBLAS_WORKSPACE_CONFIG=:4096:8 nohup python3 run/test_diffuser_offline.py --model_id=4 --ddp_trained --cp_path=/exhdd/seungyu/diffusion_motion/trained_weights/diffuser_bil_50m_100k_ddp_16x16_64h_raster_mask_only/run1/ckpt_final.ckpt \
    --device=cuda:7 --num_vis=1416 --vis_fname=random_seed_log/failure_case_seed67_vis \
    --test_path=/exhdd/seungyu/diffusion_motion/dataset/test_scenarios_20000_16x16_64h.npy --test_bs=256 --strict_seed --seed=67 --test_failure_case > random_seed_log/failure_case_seed67.log 2>&1 &

CUBLAS_WORKSPACE_CONFIG=:4096:8 nohup python3 run/test_diffuser_offline.py --model_id=4 --ddp_trained --cp_path=/exhdd/seungyu/diffusion_motion/trained_weights/diffuser_bil_50m_100k_ddp_16x16_64h_raster_mask_only/run1/ckpt_final.ckpt \
    --device=cuda:0 --num_vis=1416 --vis_fname=random_seed_log/failure_case_seed112_vis \
    --test_path=/exhdd/seungyu/diffusion_motion/dataset/test_scenarios_20000_16x16_64h.npy --test_bs=256 --strict_seed --seed=112 --test_failure_case > random_seed_log/failure_case_seed112.log 2>&1 &

CUBLAS_WORKSPACE_CONFIG=:4096:8 nohup python3 run/test_diffuser_offline.py --model_id=4 --ddp_trained --cp_path=/exhdd/seungyu/diffusion_motion/trained_weights/diffuser_bil_50m_100k_ddp_16x16_64h_raster_mask_only/run1/ckpt_final.ckpt \
    --device=cuda:1 --num_vis=1416 --vis_fname=random_seed_log/failure_case_seed176_vis \
    --test_path=/exhdd/seungyu/diffusion_motion/dataset/test_scenarios_20000_16x16_64h.npy --test_bs=256 --strict_seed --seed=176 --test_failure_case > random_seed_log/failure_case_seed176.log 2>&1 &

CUBLAS_WORKSPACE_CONFIG=:4096:8 nohup python3 run/test_diffuser_offline.py --model_id=4 --ddp_trained --cp_path=/exhdd/seungyu/diffusion_motion/trained_weights/diffuser_bil_50m_100k_ddp_16x16_64h_raster_mask_only/run1/ckpt_final.ckpt \
    --device=cuda:2 --num_vis=1416 --vis_fname=random_seed_log/failure_case_seed227_vis \
    --test_path=/exhdd/seungyu/diffusion_motion/dataset/test_scenarios_20000_16x16_64h.npy --test_bs=256 --strict_seed --seed=227 --test_failure_case > random_seed_log/failure_case_seed227.log 2>&1 &
