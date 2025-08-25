#!/bin/bash
#SBATCH --job-name=qwen3_4b_rloo
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=test_log/qwen3_4b_rloo_%j.out
#SBATCH --error=test_log/qwen3_4b_rloo_%j.err

module load singularity/3.9.7
module load cuda

singularity exec --nv \
  --bind /ibex/project/c2328:/ibex/project/c2328 \
  --bind /ibex/project/c2328/verl_singularity:/workspace/ \
  /ibex/project/c2328/verl_singularity/verlai-verl-app-verl0.5-sglang0.4.9.post6-mcore0.12.2-te2.2.sif \
  bash -c "


export CUDA_VISIBLE_DEVICES=0,1,2,3
export SSL_CERT_FILE=/ibex/project/c2328/verl_singularity/cacert.pem
export HF_DATASETS_CACHE=/ibex/project/c2328/.cache
export HF_CACHE_DIR=/ibex/project/c2328/.cache
export HF_HOME=/ibex/project/c2328/.cache/huggingface
export HF_HUB_CACHE=/ibex/project/c2328/.cache/huggingface/hub
export TRITON_CACHE_DIR=/ibex/project/c2328/.cache/triton

PROJECT_DIR=/ibex/project/c2328/verl_singularity
DATA_DIR=\$PROJECT_DIR/data/compression_dataset
OUT_DIR=\$PROJECT_DIR/data/efficient_reasoning_qwen3_4b_rloo_length_aware
mkdir -p \"\$PROJECT_DIR/test_log\" \"\$OUT_DIR\"

PARQUET_TRAIN=\$OUT_DIR/train.parquet
PARQUET_VAL=\$OUT_DIR/val.parquet

python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=rloo \
  data.train_files=\$PARQUET_TRAIN \
  data.val_files=\$PARQUET_VAL \
  data.train_batch_size=16 \
  data.max_prompt_length=1024 \
  data.max_response_length=8196 \
  data.filter_overlong_prompts=True \
  data.truncation='error' \
  data.shuffle=False \
  actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.model.path=Qwen/Qwen3-4B \
  actor_rollout_ref.actor.optim.lr=5e-6 \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.actor.ppo_mini_batch_size=4 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.rollout.name=sglang \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
  actor_rollout_ref.rollout.n=4 \
  actor_rollout_ref.rollout.load_format=safetensors \
  actor_rollout_ref.rollout.layered_summon=True \
  actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
  actor_rollout_ref.rollout.val_kwargs.n=4 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
  algorithm.use_kl_in_reward=False \
  trainer.logger='[\"console\", \"wandb\"]' \
  trainer.project_name='verl_efficient_reasoning_compression_noeos' \
  trainer.experiment_name='qwen3_4b_rloo_length_aware' \
  trainer.n_gpus_per_node=4 \
  trainer.nnodes=1 \
  trainer.save_freq=50 \
  trainer.max_actor_ckpt_to_keep=10 \
  trainer.total_training_steps=600 \
  trainer.total_epochs=1 \
  trainer.val_before_train=False \
  reward_model.reward_manager=batch \
  +reward_model.reward_kwargs.tokenizer_name=Qwen/Qwen3-4B \
  +reward_model.reward_kwargs.alpha=0.1 \
  +reward_model.reward_kwargs.check_eos=False \
  custom_reward_function.path=\$PROJECT_DIR/verl/recipe/efficient_reasoning/reward_function.py \
  custom_reward_function.name=compute_score_batch
"


