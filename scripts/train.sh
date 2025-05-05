# prepare the data
export DATASET="/mllm_hdd/mllm_hdd/yfzhang/lmm-r1/examples/data/reward_model/50k_each_class_10k.jsonl"
export RAY_memory_monitor_refresh_ms=0


MODEL_CPK_NAME="r1_reward"
PRETRAIN_MODEL="Qwen/Qwen2.5-VL-7B-Instruct"
SAVE_PATH="/mllm_hdd/mllm_hdd/yfzhang/lmm-r1/ckpts"
mkdir -p "${SAVE_PATH}/${MODEL_CPK_NAME}"
NUM_GPUS=8
# nohup bash examples/scripts/reward_model/shs/train_rloo_qwenvl2_5_math_mllm_2.sh >> 50k_each_class_10k_128_256_3e_7_consist05_wolength.out 2>&1 &

python -m openrlhf.models.remote_rm.math_verifier_mllm --dataset $DATASET --input_key message --prompt-template chatml > "${SAVE_PATH}/${MODEL_CPK_NAME}/remote_rm.log" 2>&1 &
childpid=$!

ray start --head --node-ip-address 0.0.0.0 --num-gpus $NUM_GPUS \
  --temp-dir /mllm_hdd/mllm_hdd/yfzhang/lmm-r1/data/rlhf \
  --object-store-memory=20000000000  # 10GB

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json='{"working_dir": "./data/OpenRLHF"}' \
   -- python3 -m openrlhf.cli.train_ppo_ray \
   --ref_num_nodes 1 \
   --ref_num_gpus_per_node 8 \
   --remote_rm_url http://127.0.0.1:20240/get_reward_mllm \
   --actor_num_nodes 1 \
   --actor_num_gpus_per_node 8 \
   --vllm_num_engines 8 \
   --vllm_tensor_parallel_size 1 \
   --colocate_all_models \
   --vllm_gpu_memory_utilization 0.7 \
   --vllm_sync_backend gloo \
   --vllm_enable_sleep \
   --enable_prefix_caching \
   --pretrain $PRETRAIN_MODEL \
   --save_path $SAVE_PATH/$MODEL_CPK_NAME \
   --micro_train_batch_size 2 \
   --train_batch_size 128 \
   --micro_rollout_batch_size 4 \
   --rollout_batch_size 256 \
   --temperature 1 \
   --n_samples_per_prompt 1 \
   --max_epochs 1 \
   --num_episodes 1 \
   --adam_offload \
   --grad_accum_dtype bf16 \
   --prompt_max_len 20480 \
   --max_samples 300000 \
   --generate_max_len 1024 \
   --advantage_estimator reinforce \
   --zero_stage 3 \
   --bf16 \
   --train_vlm \
   --load_checkpoint \
   --actor_learning_rate 3e-7 \
   --init_kl_coef 1e-3 \
   --prompt_data $DATASET \
   --input_key message \
   --normalize_reward \
   --flash_attn \
   --gradient_checkpointing \
   --save_steps 25 \
   --max_ckpt_num 1000000 \
   --ckpt_path $SAVE_PATH/$MODEL_CPK_NAME/ckpt \
   --save_hf_ckpt \
   --use_tensorboard $SAVE_PATH/$MODEL_CPK_NAME/logs

# also supports --advantage_estimator rloo
# --vllm_enable_sleep \
ray stop