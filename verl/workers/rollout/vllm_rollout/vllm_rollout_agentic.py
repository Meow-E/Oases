# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
The vllm_rollout that can be applied in different backend
When working with FSDP:
- Use DTensor weight loader (recommended) or HF weight loader
- Utilize state_dict from the FSDP to synchronize the weights among tp ranks in vLLM
When working with Megatron:
- Use Megatron weight loader
- During training, only the current pp stage holds the parameters
- Before inference, broadcast the parameters of the current pp rank
  to all other pp ranks (all pp ranks holds all the parameters)
- Bind the parameters to the inference engine
- Do inference in tp. pp is treated as additional dp
- After inference, all the parameters that doesn't belong to this pp rank is freed.
"""

import logging
import os
from copy import deepcopy

import numpy as np
import math
import torch
import torch.distributed
from omegaconf import DictConfig, OmegaConf
from tensordict import TensorDict
from verl.workers.rollout.agent import create_agent

from verl import DataProto
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.torch_functional import get_response_mask, pad_2d_list_to_length
from verl.workers.rollout.vllm_rollout.vllm_rollout_spmd import vLLMRollout
from verl.workers.rollout.agent.tools import *
from verl.workers.config import HFModelConfig, RolloutConfig
from torch.distributed.device_mesh import DeviceMesh


logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class vLLMAgenticRollout(vLLMRollout):
    def __init__(self, config: RolloutConfig, model_config: HFModelConfig, device_mesh: DeviceMesh):
        """A vLLM rollout. It requires the module is supported by the vllm.
        """
        super().__init__(config, model_config, device_mesh)

        self.tokenizer = model_config.tokenizer
        if not self.config.agentic_agent.activate_agent:
            self.agent = None
        else:
            self.agent = create_agent(
                config=config,
                vllm_engine=self.inference_engine,
                tokenizer=self.tokenizer
            )
        
        self.eos_token_id = self.tokenizer.eos_token_id


    @GPUMemoryLogger(role="vllm rollout spmd", logger=logger)
    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        """Generate sequences for a batch of prompts.

        Args:
            batch (DataProto): Input batch.

        Returns:
            DataProto: Output batch.
            - prompts: [bsz, prompt_length], prompt token ids from dataset.
            - responses: [bsz, response_length], output token ids include response tokens
              from LLM generation and observation tokens from tool_calls.
            - response_mask: [bsz, response_length], 1 for LLM generated tokens, 0 for observation/padding tokens.
            - input_ids: [bsz, prompt_length + response_length], whole sequence token ids, including prompt tokens
              and response tokens.
            - attention_mask: [bsz, prompt_length + response_length], 0 for padding tokens, 1 for other tokens.
            - position_ids: [bsz, prompt_length + response_length], incremental position ids.

            For multi-turn conversations:
            responses:     |<- LLM generation ->|<- tool_calls ->|<- LLM generation ->|<- padding ->|
            response_mask: | 1, 1, 1, ..., 1, 1 | 0, 0, .., 0, 0 | 1, 1, 1, ..., 1, 1 | 0, 0, ..., 0|
        """
        is_main = prompts.meta_info.get('is_main', True)

        if not self.agent or not is_main:
            return super().generate_sequences(prompts, **kwargs)

        non_tensor_batch = prompts.non_tensor_batch
        input_ids = prompts.batch["input_ids"]
        batch_size = input_ids.size(0)

        # --- 1. Prepare Sampling Parameters ---
        do_sample = prompts.meta_info.get('do_sample', True)
        is_validate = prompts.meta_info.get('validate', False)

        self.agent.set_generate_type('validate' if is_validate else "train")
        
        sampling_kwargs = {}
        if not do_sample:
            sampling_kwargs.update({'best_of': 1, 'top_p': 1.0, 'top_k': -1, 'min_p': 0.0, 'temperature': 0, 'n': 1})
        elif is_validate:
            sampling_kwargs.update({
                'top_k': self.config.val_kwargs.top_k,
                'top_p': self.config.val_kwargs.top_p,
                'temperature': self.config.val_kwargs.temperature,
                'n': 1
            })
        sampling_kwargs.update(kwargs)

        with self.update_sampling_params(**sampling_kwargs):
            # --- 2. Drive the agent's reset/step loop ---
            self.agent.reset(prompts, self.sampling_params)

            while True:
                dones = self.agent.step()
                if torch.all(dones):
                    break
            
            # --- 3. Get final results from the agent ---
            agent_output_batch = self.agent.get_final_responses()

        # --- 4. Finalize the DataProto ---
        responses = agent_output_batch.pop('responses')
        response_mask = agent_output_batch.pop('response_masks')
        loop_tags = agent_output_batch.pop('loop_tags')
        num_turns = agent_output_batch.pop('num_turns', None)
        probe_responses = agent_output_batch.pop('delta_response', None)

        input_ids = prompts.batch['input_ids']
        attention_mask = prompts.batch['attention_mask']
        position_ids = prompts.batch['position_ids']
        
        seq = torch.cat([input_ids, responses], dim=-1)

        response_length = responses.size(1)
        
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device).unsqueeze(0).expand(batch_size, -1)
        response_position_ids = position_ids[..., -1:] + delta_position_id
        final_position_ids = torch.cat([position_ids, response_position_ids], dim=-1)

        response_attention_mask = get_response_mask(response_id=responses, eos_token=self.eos_token_id, dtype=attention_mask.dtype)
        final_attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        batch_dict = {
            "prompts": input_ids,
            "responses": responses,
            "response_mask": response_mask,
            "response_tags": loop_tags,
            "input_ids": seq,
            "attention_mask": final_attention_mask,
            "position_ids": final_position_ids,
        }

        batch = TensorDict(batch_dict, batch_size=batch_size)
        
        if num_turns is not None:
            non_tensor_batch["__num_turns__"] = np.array(num_turns)
        
        if probe_responses is not None:
            probe_responses_array = np.empty(len(probe_responses), dtype=object)
            for i, row in enumerate(probe_responses):
                probe_responses_array[i] = row
            non_tensor_batch["probe_responses"] = probe_responses_array

        meta_info = deepcopy(prompts.meta_info)
                
        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch, meta_info=meta_info)