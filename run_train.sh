#!/bin/bash
export WANDB_API_KEY=""

MODEL_SIZE=4
MODEL_TYPE=3
BATCH_SIZE=256
TEST_FREQ=$((256 * 78 / BATCH_SIZE))
SAVE_FREQ=$((256 * 78 / BATCH_SIZE))

TEST_DATASET=hotpotqa
ALPHA=0.5

if [ "$MODEL_TYPE" == "3" ]; then
    BASE_MODEL_PATH="/models/Qwen/Qwen${MODEL_TYPE}-${MODEL_SIZE}B-Base"
else
    BASE_MODEL_PATH="/models/Qwen/Qwen${MODEL_TYPE}-${MODEL_SIZE}B"
fi

#  +data.apply_chat_template_kwargs.enable_thinking=False \
NCCL_NVLS_ENABLE=0 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
 data.train_files=/oases/data/2wiki_hotpotqa_train.parquet \
 data.val_files=/oases/data/test_${TEST_DATASET}.parquet \
 data.train_batch_size=$BATCH_SIZE \
 data.val_batch_size=256 \
 data.max_prompt_length=3000 \
 data.max_response_length=2000 \
 actor_rollout_ref.model.path=$BASE_MODEL_PATH \
 actor_rollout_ref.actor.optim.lr=2e-7 \
 actor_rollout_ref.model.use_remove_padding=True \
 actor_rollout_ref.actor.ppo_mini_batch_size=32 \
 actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
 actor_rollout_ref.model.use_shm=True \
 actor_rollout_ref.model.enable_gradient_checkpointing=True \
 actor_rollout_ref.actor.use_kl_loss=False \
 actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
 actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
 actor_rollout_ref.rollout.name=vllm_agentic \
 actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
 actor_rollout_ref.rollout.agentic_agent.name=agentic_agent_soft_truncation \
 actor_rollout_ref.rollout.agentic_agent.tools.tool_instances.search.params.host=xx.xxx.xx.xxx \
 actor_rollout_ref.rollout.agentic_agent.tools.tool_instances.search.params.ports="[8000,8001,8002,8003,8004,8005,8006,8007]" \
 critic.optim.lr=1e-6 \
 critic.model.use_remove_padding=True \
 critic.model.path=$BASE_MODEL_PATH \
 critic.model.enable_gradient_checkpointing=True \
 critic.ppo_micro_batch_size_per_gpu=2 \
 algorithm.alpha=$ALPHA \
 algorithm.use_kl_in_reward=False \
 algorithm.final_reward_type=f1 \
 algorithm.process_reward_type=f1 \
 reward_model.reward_manager=f1 \
 trainer.joint=joint \
 trainer.val_before_train=False \
 trainer.val_only=False \
 trainer.critic_warmup=0 \
 trainer.logger="[console, wandb]" \
 trainer.project_name='oases' \
 trainer.experiment_name="oases_train_${MODEL_SIZE}B_joint_alpha${ALPHA}-base" \
 trainer.n_gpus_per_node=8 \
 trainer.nnodes=1 \
 trainer.save_freq=$SAVE_FREQ \
 trainer.test_freq=$TEST_FREQ \
 trainer.total_epochs=5 2>&1 | tee logs/oases_train_${MODEL_SIZE}B_joint_alpha${ALPHA}-base.log