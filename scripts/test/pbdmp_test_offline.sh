# 1. PBDMP V2 Test 1 (V)
nohup python3 run/test_pbdmp_offline.py --model_id=2 --cp_path=/exhdd/seungyu/diffusion_motion/trained_weights/pb_diffusion_v2_38345/run1/state_300000.pt \
    --device=cuda:0 --num_vis=500 --vis_fname=pbdmp_v2_38345_offline --test_path=dataset/test_scenarios_20000.npy --test_bs=64 > test_results_log/pbdmp_v2_38345_offline.log 2>&1 &

# 2. PBDMP V2 Test 2 (V)
nohup python3 run/test_pbdmp_offline.py --model_id=2 --cp_path=/exhdd/seungyu/diffusion_motion/trained_weights/pb_diffusion_v2_95792/run1/state_400000.pt \
    --device=cuda:1 --num_vis=500 --vis_fname=pbdmp_v2_95792_offline --test_path=dataset/test_scenarios_20000.npy --test_bs=64 > test_results_log/pbdmp_v2_95792_offline.log 2>&1 &


# 3. PBDMP V2 Test 3 (V)
nohup python3 run/test_pbdmp_offline.py --model_id=2 --cp_path=/exhdd/seungyu/diffusion_motion/trained_weights/pb_diffusion_v2/run1/state_20000.pt \
    --device=cuda:2 --num_vis=500 --vis_fname=pbdmp_v2_2000_offline --test_path=dataset/test_scenarios_20000.npy --test_bs=64 > test_results_log/pbdmp_v2_2000_offline.log 2>&1 &


# 4. PBDMP Test 1 (V)
nohup python3 run/test_pbdmp_offline.py --model_id=0 --cp_path=/exhdd/seungyu/diffusion_motion/trained_weights/pb_diffusion_38345/run1/state_200000.pt \
    --device=cuda:3 --num_vis=500 --vis_fname=pbdmp_38345_offline --test_path=dataset/test_scenarios_20000.npy --test_bs=64 > test_results_log/pbdmp_38345_offline.log 2>&1 &


# 5. PBDMP Test 2 (V)
nohup python3 run/test_pbdmp_offline.py --model_id=0 --cp_path=/exhdd/seungyu/diffusion_motion/trained_weights/pb_diffusion_95792/run1/state_380000.pt \
    --device=cuda:4 --num_vis=500 --vis_fname=pbdmp_95792_offline --test_path=dataset/test_scenarios_20000.npy --test_bs=64 > test_results_log/pbdmp_95792_offline.log 2>&1 &


# 6. PBDMP Test 3 (V)
nohup python3 run/test_pbdmp_offline.py --model_id=0 --cp_path=/exhdd/seungyu/diffusion_motion/trained_weights/pb_diffusion/run1/state_90000.pt \
    --device=cuda:5 --num_vis=500 --vis_fname=pbdmp_2000_offline --test_path=dataset/test_scenarios_20000.npy --test_bs=64 > test_results_log/pbdmp_2000_offline.log 2>&1 &


# Visualization purpose
nohup python3 run/test_pbdmp_offline.py --model_id=2 --cp_path=/exhdd/seungyu/diffusion_motion/trained_weights/pb_diffusion_v2_38345/run1/state_300000.pt \
    --device=cuda:1 --num_vis=500 --vis_fname=pbdmp_v2_for_vis_500 --test_path=dataset/test_scenarios_20000.npy --test_bs=1 > pbdmp_v2_for_vis_500.log 2>&1 &


# 7. PBDMP on 16x16 map, 64 horizon (-)
nohup python3 run/test_pbdmp_offline.py --model_id=2 --ddp_trained --cp_path=/exhdd/seungyu/diffusion_motion/trained_weights/pb_v2_ddp_16x16_64h_100k/run1/state_100000.pt \
    --device=cuda:0 --num_vis=500 --vis_fname=pbdmp_v2_ddp_16x16_64h_50m_100k_offline --test_path=dataset/test_scenarios_20000_16x16_64h.npy --test_bs=64 > test_results_log/pbdmp_v2_ddp_16x16_64h_50m_100k_offline.log 2>&1 &

# Visualization purpose
nohup python3 run/test_pbdmp_offline.py --model_id=2 --ddp_trained --cp_path=/exhdd/seungyu/diffusion_motion/trained_weights/pb_v2_ddp_16x16_64h_100k/run1/state_100000.pt \
    --device=cuda:7 --num_vis=1000 --vis_fname=pbdmp_v2_ddp_16x16_64h_50m_100k_vis --test_path=dataset/test_scenarios_20000_16x16_64h.npy --test_bs=100 > test_results_log/pbdmp_v2_ddp_16x16_64h_50m_100k_vis.log 2>&1 &
