nohup python3 core/rediffuser/rnd/rnd_plan.py --model_id=0 --ddp_trained --cp_path=/exhdd/seungyu/diffusion_motion/trained_weights/rediff_ddp_100k/run1/state_100000.pt \
    --device=cuda:0 --test_num=1000 --map_size=8 --num_vis=500 --vis_fname=rediff_rnd_ddp_100k \
    --n_plans=8 --rnd_keyword=rrt_8x8_100k --discount_power=1.0 > test_results_log/rediff_rnd_ddp_100k.log 2>&1 &
    