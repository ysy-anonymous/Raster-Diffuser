# Below is the possible test for ReDiffuser Model Training.
# Currently, Training amount of RND network is not obvious and performance not helpful when using RND network also.
# Directly, Adapting RND network for our tasks degrades the performance. It looks like just learning familiarity of the output trajectory shape does not works well on
# our 2D obstacle conditioned task.

# Test ReDiff RND Trained Network 1 (V)
nohup python3 core/rediffuser/rnd/rnd_plan_offline.py --model_id=0 --ddp_trained --cp_path=/exhdd/seungyu/diffusion_motion/trained_weights/rediff_ddp_40k/run1/state_40000.pt \
    --device=cuda:0 --num_vis=500 --vis_fname=rediff_40k_rnd_40k_ddp_offline --test_path=dataset/test_scenarios_20000.npy --test_bs=64 \
    --n_plans=3 --n_alter=5 --rnd_keyword=rrt_8x8_40k --rnd_n_epochs=100 --discount_power=1.0 > test_results_log/rediff_40k_rnd_40k_ddp_offline.log 2>&1 &

# Test ReDiff RND Trained Network 2 (V)
nohup python3 core/rediffuser/rnd/rnd_plan_offline.py --model_id=0 --ddp_trained --cp_path=/exhdd/seungyu/diffusion_motion/trained_weights/rediff_ddp_100k/run1/state_100000.pt \
    --device=cuda:0 --num_vis=500 --vis_fname=rediff_100k_rnd_100k_ddp_offline --test_path=dataset/test_scenarios_20000.npy --test_bs=64 \
    --n_plans=3 --n_alter=5 --rnd_keyword=rrt_8x8_100k --rnd_n_epochs=100 --discount_power=1.0 > test_results_log/rediff_100k_rnd_100k_ddp_offline.log 2>&1 &

# Test ReDiff RND Trained Network 3 (V)
nohup python3 core/rediffuser/rnd/rnd_plan_offline.py --model_id=0 --cp_path=/exhdd/seungyu/diffusion_motion/trained_weights/rediffuser/run1/state_20000.pt \
    --device=cuda:1 --num_vis=500 --vis_fname=rediff_2k_rnd_2k_offline --test_path=dataset/test_scenarios_20000.npy --test_bs=64 \
    --n_plans=3 --n_alter=5 --rnd_keyword=rrt_8x8_2k --rnd_n_epochs=100 --discount_power=1.0 > test_results_log/rediff_2k_rnd_2k_offline.log 2>&1 &


# Test ReDiff RNDV2 Trained Network 4 (V)
nohup python3 core/rediffuser/rnd/rnd_plan_v2_offline.py --model_id=0 --ddp_trained --cp_path=/exhdd/seungyu/diffusion_motion/trained_weights/rediff_ddp_40k/run1/state_40000.pt \
    --device=cuda:0 --num_vis=500 --vis_fname=rediff_40k_rndv2_40k_ddp_offline --test_path=dataset/test_scenarios_20000.npy --test_bs=64 \
    --n_plans=3 --n_alter=5 --rnd_keyword=rrt_8x8_40k --rnd_n_epochs=100 --discount_power=1.0 > test_results_log/rediff_40k_rndv2_40k_ddp_offline.log 2>&1 &

# Test ReDiff RNDV2 Trained Network 5 (V)
nohup python3 core/rediffuser/rnd/rnd_plan_v2_offline.py --model_id=0 --ddp_trained --cp_path=/exhdd/seungyu/diffusion_motion/trained_weights/rediff_ddp_100k/run1/state_100000.pt \
    --device=cuda:1 --num_vis=500 --vis_fname=rediff_100k_rndv2_100k_ddp_offline --test_path=dataset/test_scenarios_20000.npy --test_bs=64 \
    --n_plans=3 --n_alter=5 --rnd_keyword=rrt_8x8_100k --rnd_n_epochs=100 --discount_power=1.0 > test_results_log/rediff_100k_rndv2_100k_ddp_offline.log 2>&1 &

# Test ReDiff RNDV2 Trained Network 6 (V)
nohup python3 core/rediffuser/rnd/rnd_plan_v2_offline.py --model_id=0 --ddp_trained --cp_path=/exhdd/seungyu/diffusion_motion/trained_weights/rediffuser/run1/state_20000.pt \
    --device=cuda:2 --num_vis=500 --vis_fname=rediff_2k_rndv2_2k_offline --test_path=dataset/test_scenarios_20000.npy --test_bs=64 \
    --n_plans=3 --n_alter=5 --rnd_keyword=rrt_8x8_2k --rnd_n_epochs=100 --discount_power=1.0 > test_results_log/rediff_2k_rndv2_2k_offline.log 2>&1 &


# For Visualization
nohup python3 core/rediffuser/rnd/rnd_plan_v2_offline.py --model_id=0 --ddp_trained --cp_path=/exhdd/seungyu/diffusion_motion/trained_weights/rediff_ddp_40k/run1/state_40000.pt \
    --device=cuda:0 --num_vis=500 --vis_fname=rediff_40k_rndv2_40k_ddp_for_vis_500 --test_path=dataset/test_scenarios_20000.npy --test_bs=1 \
    --n_plans=3 --n_alter=5 --rnd_keyword=rrt_8x8_40k --rnd_n_epochs=100 --discount_power=1.0 > rediff_40k_rndv2_40k_ddp_for_vis_500.log 2>&1 &


# Test ReDiff RNDV2 Trained Network 6 (-)
nohup python3 core/rediffuser/rnd/rnd_plan_v2_offline.py --model_id=0 --ddp_trained --cp_path=/exhdd/seungyu/diffusion_motion/trained_weights/rediff_ddp_16x16_64h_100k/run1/state_100000.pt \
    --device=cuda:0 --num_vis=500 --vis_fname=rediff_ddp_16x16_64h_100k_offline --test_path=dataset/test_scenarios_20000_16x16_64h.npy --test_bs=64 \
    --n_plans=3 --n_alter=5 --rnd_keyword=rrt_16x16_100k --rnd_n_epochs=100 --discount_power=1.0 --rnd_mask_channels=1 > test_results_log/rediff_ddp_16x16_64h_100k_offline.log 2>&1 &


# For Visualization
nohup python3 core/rediffuser/rnd/rnd_plan_v2_offline.py --model_id=0 --ddp_trained --cp_path=/exhdd/seungyu/diffusion_motion/trained_weights/rediff_ddp_16x16_64h_100k/run1/state_100000.pt \
    --device=cuda:1 --num_vis=1000 --vis_fname=rediff_ddp_16x16_64h_100k_vis --test_path=dataset/test_scenarios_20000_16x16_64h.npy --test_bs=100 \
    --n_plans=3 --n_alter=5 --rnd_keyword=rrt_16x16_100k --rnd_n_epochs=100 --discount_power=1.0 --rnd_mask_channels=1 > test_results_log/rediff_ddp_16x16_64h_100k_vis.log 2>&1 &
