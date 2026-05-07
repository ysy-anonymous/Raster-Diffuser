# Comp Diffusion V2 DDP Test 1
nohup python3 run/test_comp_offline.py --model_id=1 --ddp_trained --cp_path=/exhdd/seungyu/diffusion_motion/trained_weights/comp_v2_ddp_38345/run1/state_100000.pt \
    --device=cuda:3 --num_vis=500 --vis_fname=comp_v2_40k_ddp_offline --test_path=dataset/test_scenarios_20000.npy --test_bs=64 > test_results_log/comp_v2_40k_ddp_offline.log 2>&1 &

# Comp Diffusion V2 DDP Test 2
nohup python3 run/test_comp_offline.py --model_id=1 --ddp_trained --cp_path=/exhdd/seungyu/diffusion_motion/trained_weights/comp_v2_ddp_95792/run1/state_200000.pt \
    --device=cuda:4 --num_vis=500 --vis_fname=comp_v2_100k_ddp_offline --test_path=dataset/test_scenarios_20000.npy --test_bs=64 > test_results_log/comp_v2_100k_ddp_offline.log 2>&1 &

# Comp Diffusion V2 DDP Test 3
nohup python3 run/test_comp_offline.py --model_id=1 --cp_path=/exhdd/seungyu/diffusion_motion/trained_weights/comp_diffusion_v2/run1/state_20000.pt \
    --device=cuda:5 --num_vis=500 --vis_fname=comp_v2_2k_offline --test_path=dataset/test_scenarios_20000.npy --test_bs=64 > test_results_log/comp_v2_2k_offline.log 2>&1 &

# Comp Diffusion DDP Test 4
nohup python3 run/test_comp_offline.py --model_id=0 --ddp_trained --cp_path=/exhdd/seungyu/diffusion_motion/trained_weights/comp_diffusion_ddp_38345/run1/state_100000.pt \
    --device=cuda:6 --num_vis=500 --vis_fname=comp_v1_40k_ddp_offline --test_path=dataset/test_scenarios_20000.npy --test_bs=64 > test_results_log/comp_v1_40k_ddp_offline.log 2>&1 &

# Comp Diffusion DDP Test 5
nohup python3 run/test_comp_offline.py --model_id=0 --ddp_trained --cp_path=/exhdd/seungyu/diffusion_motion/trained_weights/comp_diffusion_ddp_95792/run1/state_200000.pt \
    --device=cuda:7 --num_vis=500 --vis_fname=comp_v1_100k_ddp_offline --test_path=dataset/test_scenarios_20000.npy --test_bs=64 > test_results_log/comp_v1_100k_ddp_offline.log 2>&1 &

# Comp Diffusion DDP Test 6
nohup python3 run/test_comp_offline.py --model_id=0 --cp_path=/exhdd/seungyu/diffusion_motion/trained_weights/comp_diffusion/run1/state_20000.pt \
    --device=cuda:5 --num_vis=500 --vis_fname=comp_v1_2k_offline --test_path=dataset/test_scenarios_20000.npy --test_bs=64 > test_results_log/comp_v1_2k_offline.log 2>&1 &


# For Visualization
nohup python3 run/test_comp_offline.py --model_id=1 --ddp_trained --cp_path=/exhdd/seungyu/diffusion_motion/trained_weights/comp_v2_ddp_38345/run1/state_100000.pt \
    --device=cuda:0 --num_vis=500 --vis_fname=comp_v2_for_vis_500 --test_path=dataset/test_scenarios_20000.npy --test_bs=1 > comp_v2_for_vis_500.log 2>&1 &


# Comp Diffusion V2 DDP Test 7 (16x16 map, 64 horizon)
nohup python3 run/test_comp_offline.py --model_id=1 --ddp_trained --cp_path=/exhdd/seungyu/diffusion_motion/trained_weights/comp_v2_ddp_16x16_64h_100k/run1/state_280000.pt \
    --device=cuda:1 --num_vis=500 --vis_fname=comp_v2_ddp_16x16_64h_100k_offline --test_path=dataset/test_scenarios_20000_16x16_64h.npy --test_bs=64 > test_results_log/comp_v2_ddp_16x16_64h_100k_offline.log 2>&1 &

# For Visualization
nohup python3 run/test_comp_offline.py --model_id=1 --ddp_trained --cp_path=/exhdd/seungyu/diffusion_motion/trained_weights/comp_v2_ddp_16x16_64h_100k/run1/state_280000.pt \
    --device=cuda:6 --num_vis=1000 --vis_fname=comp_v2_ddp_16x16_64h_100k_vis --test_path=dataset/test_scenarios_20000_16x16_64h.npy --test_bs=100 > test_results_log/comp_v2_ddp_16x16_64h_100k_vis.log 2>&1 &