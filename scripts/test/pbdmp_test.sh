nohup python3 run/test_pb_diffusion.py --model_id=2 --cp_path=/exhdd/seungyu/diffusion_motion/trained_weights/pb_diffusion_v2_38345/run1/state_100000.pt \
    --device=cuda:1 --test_num=1000 --map_size=8 --num_vis=500 --vis_fname='PBDMP_V2_38345_st100K' > test_results_log/pbdmp_v2_38345_st100K.log 2>&1 &

nohup python3 run/test_pb_diffusion.py --model_id=2 --cp_path=/exhdd/seungyu/diffusion_motion/trained_weights/pb_diffusion_v2_95792/run1/state_200000.pt \
    --device=cuda:2 --test_num=1000 --map_size=8 --num_vis=500 --vis_fname='PBDMP_V2_95792_st200K' > test_results_log/pbdmp_v2_95792_st200K.log 2>&1 &