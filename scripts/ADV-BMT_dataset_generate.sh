dir="/bigdata/yuxin/scenarionet_waymo_training_500" 
save_dir="/bigdata/yuxin/SCGen_all_TF_except_adv_scenarionet_waymo_training_500_5_modes"
num_scenario=500
TF_mode="all_TF_except_adv"
num_mode=5

mkdir -p $save_dir

python SCGen_dataset_generate.py --dir $dir --save_dir $save_dir --TF_mode $TF_mode --num_scenario $num_scenario --num_mode $num_mode