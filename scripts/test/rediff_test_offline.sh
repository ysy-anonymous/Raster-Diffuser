# 1. ReDiffuser without RND network (-)
nohup python3 run/test_rediffuser_offline.py --model_id=0 --cp_path=/exhdd/seungyu/diffusion_motion/trained_weights/rediffuser/run1/state_20000.pt \
    --device=cuda:0 --num_vis=500 --vis_fname=rediffuser_2k --test_path=dataset/test_scenarios_20000.npy --test_bs=64 > test_results_log/rediffuser_2k.log 2>&1 &

# 2. ReDiffuser without RND network (-)
nohup python3 run/test_rediffuser_offline.py --model_id=0 --ddp_trained --cp_path=/exhdd/seungyu/diffusion_motion/trained_weights/rediff_ddp_40k/run1/state_40000.pt \
    --device=cuda:1 --num_vis=500 --vis_fname=rediff_ddp_40k --test_path=dataset/test_scenarios_20000.npy --test_bs=64 > test_results_log/rediff_ddp_40k.log 2>&1 &

# 3. ReDiffuser without RND network (-)
nohup python3 run/test_rediffuser_offline.py --model_id=0 --ddp_trained --cp_path=/exhdd/seungyu/diffusion_motion/trained_weights/rediff_ddp_100k/run1/state_100000.pt \
    --device=cuda:2 --num_vis=500 --vis_fname=rediff_ddp_100k --test_path=dataset/test_scenarios_20000.npy --test_bs=64 > test_results_log/rediff_ddp_100k.log 2>&1 &