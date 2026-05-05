import concurrent.futures
import logging
import os
from collections import Counter
from copy import deepcopy
from typing import List, Dict, Any, Tuple
import random
import threading
import time

import re
import math
import torch
import torch.distributed as dist
from omegaconf import DictConfig
from tensordict import TensorDict
from vllm import LLM, SamplingParams
from vllm.distributed import parallel_state as vllm_ps

from verl import DataProto
from verl.utils.debug import GPUMemoryLogger
from verl.utils.torch_functional import pad_2d_list_to_length
from verl.workers.rollout.agent.base import BaseAgent
from verl.workers.rollout.agent.tools.tool_executor import ToolExecutor

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


class AgenticAgentSoft(BaseAgent):
    """
    A stateful agent that interacts with an environment over discrete steps,
    using tools to accomplish tasks. It aligns with the OpenAI Gym interface.
    """

    def __init__(self, config: DictConfig, vllm_engine: LLM, tokenizer: Any, **kwargs):
        super().__init__(config, vllm_engine, tokenizer, **kwargs)
        tools_config = self.agent_config.tools
        self.tool_executor = ToolExecutor(tools_config)
        if not self.tool_executor.has_tools():
            raise ValueError("ToolAgent requires tools to be configured.")

        # Static parameters
        self.tool_call_limit = tools_config.call_limit
        self.max_tool_workers = tools_config.max_workers
        self.tool_timeout = tools_config.timeout
        self.max_len = self.config.response_length
        self.eos_token_id = self.tokenizer.eos_token_id
        self.pad_token_id = self.tokenizer.pad_token_id
        self.max_tokens_per_step = 512

        # State variables, initialized in `reset`
        self.sampling_params = None
        self.batch_size = 0
        self.device = None
        self.prompts_len = None
        self.curr_inputs = None
        self.init_inputs = None
        self.result_masks = None
        self.num_turns = None
        self.call_counters = None
        self.loop_tags = None
        self.dones = None
        self.active_indices = None

        self.step_count = None

        self.TRUNCATION_INSTRUCTION = '[SYSTEM WARNING] The maximum limit of tool invocation rounds has been reached. Stop using tools immediately. Derive the final answer from the currently available information. You should first analyze the known facts in <think> tags, and then output the final conclusion within <answer> tags. \n'
        self.VOCAB_LIMIT = len(self.tokenizer)

        self.generate_type = None

    def set_generate_type(self, generate_type:str):
        self.generate_type = generate_type

    def reset(self, prompts: DataProto, sampling_params: SamplingParams) -> DataProto:
        self.sampling_params = sampling_params

        input_ids = prompts.batch['input_ids']
        self.device = input_ids.device

        raw_prompt_ids = prompts.non_tensor_batch.pop('raw_prompt_ids').tolist()

        self.curr_inputs = []
        self.init_inputs = []
        self.result_masks = []
        self.loop_tags = []
        self.prompts_len = []

        for ids in raw_prompt_ids:
            safe_ids = self._filter_tokens(ids)

            self.curr_inputs.append(safe_ids.copy())
            self.init_inputs.append(safe_ids.copy())
            self.prompts_len.append(len(safe_ids))
            self.result_masks.append([])
            self.loop_tags.append([])

        self.batch_size = len(self.curr_inputs)
        self.call_counters = torch.zeros(self.batch_size, dtype=torch.int)
        self.dones = torch.zeros(self.batch_size, dtype=torch.bool)
        self.active_indices = list(range(self.batch_size))
        self.num_turns = [0] * self.batch_size
        
        self.step_count = 0

    def _filter_tokens(self, tokens: List[int]) -> List[int]:
        if not tokens:
            return []
        
        if max(tokens) < self.VOCAB_LIMIT:
            return tokens
        
        safe_tokens = [t for t in tokens if t < self.VOCAB_LIMIT]

        return safe_tokens

    @torch.no_grad()
    def step(self) -> torch.Tensor:

        if not self.active_indices:
            # If all episodes are done, return the final dones state.
            return self.dones

        # --- 1. Generate with vLLM ---
        # max_tokens = max(1, max(self.max_len - (len(self.curr_inputs[i]) - self.prompts_len[i]) for i in self.active_indices))
        remaining_budget = max(
            1,
            max(self.max_len - (len(self.curr_inputs[i]) - self.prompts_len[i]) for i in self.active_indices),
        )
        max_tokens = min(remaining_budget, self.max_tokens_per_step)
        
        current_sampling_params = deepcopy(self.sampling_params)
        current_sampling_params.n = 1
        current_sampling_params.detokenize = True
        
        # Combine original and tool stop sequences
        original_stop = current_sampling_params.stop
        tool_stop = self.tool_executor.stop_sequences
        combined_stop = set(tool_stop)
        if original_stop:
            if isinstance(original_stop, str):
                combined_stop.add(original_stop)
            else:
                combined_stop.update(original_stop)
        current_sampling_params.stop = list(combined_stop)
        current_sampling_params.include_stop_str_in_output = True
        current_sampling_params.max_tokens = max_tokens

        active_prompts = [self.curr_inputs[i] for i in self.active_indices]
        outputs = self.vllm_engine.generate(
            prompt_token_ids=active_prompts,
            sampling_params=current_sampling_params,
            use_tqdm=False
        )

        # --- 2. Process outputs and prepare tool calls ---
        tool_requests = {tag: [] for tag in self.tool_executor.tools}
        next_active_indices = []

        for i, out_idx in enumerate(self.active_indices):
            output = outputs[i].outputs[0]
            generated_tokens = output.token_ids

            safe_generated_tokens = self._filter_tokens(generated_tokens)

            self.curr_inputs[out_idx].extend(safe_generated_tokens)
            self.result_masks[out_idx].extend([1] * len(safe_generated_tokens))
            self.loop_tags[out_idx].extend([self.step_count] * len(safe_generated_tokens))
            self.num_turns[out_idx] += 1
            
            is_tool_call = output.finish_reason == 'stop' and output.stop_reason in self.tool_executor.stop_sequences

            if is_tool_call:
                if self.call_counters[out_idx] < self.tool_call_limit:
                    tag = output.stop_reason.strip("</>")
                    full_text = self.tokenizer.decode(self.curr_inputs[out_idx])
                    content = self.tool_executor.extract_content(full_text, tag)

                    if content:
                        tool_requests[tag].append({"index": out_idx, "content": content})
                        next_active_indices.append(out_idx)
                    # If content is not extracted, it's considered finished and will not be in next_active_indices.
                    self.call_counters[out_idx] += 1
            elif output.finish_reason == 'length':
                # If stopped due to length, but still within max_len, continue.
                response_len = len(self.curr_inputs[out_idx]) - self.prompts_len[out_idx]
                if response_len < self.max_len:
                    next_active_indices.append(out_idx)

            # If 'stop' due to EOS, or any other case, it's finished and not added to next_active_indices.


        # --- 3. Execute tools ---
        if any(tool_requests.values()):

            max_concurrent_requests = 100
            api_semaphore = threading.Semaphore(max_concurrent_requests)

            def limited_api_call(tool, content):
                with api_semaphore:
                    return self.tool_executor.execute_with_retry(tool, content)

            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_tool_workers) as executor:
                futures = {
                    executor.submit(limited_api_call, self.tool_executor.get_tool(tag), req["content"]):
                        {"index": req["index"], "tag": tag}
                    for tag, requests in tool_requests.items() for req in requests
                }
                for future in concurrent.futures.as_completed(futures):
                    info = futures[future]
                    idx, tag = info["index"], info["tag"]
                    try:
                        result = future.result(timeout=self.tool_timeout)
                        raw_result = result["result"]

                        if isinstance(raw_result, list):
                            truncated_list = []
                            for item in raw_result:
                                item_ids = self.tokenizer.encode(str(item), add_special_tokens=False)
                                
                                if len(item_ids) > 200:
                                    item_ids = item_ids[:200]
                                    truncated_item = self.tokenizer.decode(item_ids)
                                    truncated_list.append(truncated_item)
                                else:
                                    truncated_list.append(str(item))

                            result_text = str(truncated_list)
                        else:
                            result_text = raw_result
                    except Exception as e:
                        logger.error(f"Tool execution future failed for sample {idx}: {e}")
                        result_text = f"Error: Tool execution failed."
                    
                    if self.call_counters[idx] >= self.tool_call_limit:
                        formatted_result = f" <result> {result_text} </result> \n{self.TRUNCATION_INSTRUCTION}\n"
                    else:
                        formatted_result = f" <result> {result_text} </result>"

                    result_tokens = self.tokenizer.encode(formatted_result, add_special_tokens=False)
                    safe_result_tokens = self._filter_tokens(result_tokens)

                    self.curr_inputs[idx].extend(safe_result_tokens)
                    self.result_masks[idx].extend([0] * len(safe_result_tokens))
                    self.loop_tags[idx].extend([self.step_count] * len(safe_result_tokens))

        # --- 4. Update state and prepare outputs ---
        # Final check on active indices based on length after tool results are added
        final_active_indices = []

        for idx in next_active_indices:
            response_len = len(self.curr_inputs[idx]) - self.prompts_len[idx]
            if response_len < self.max_len:
                final_active_indices.append(idx)

        # Update dones for newly finished trajectories before branching
        newly_finished_indices = set(self.active_indices) - set(final_active_indices)
        for idx in newly_finished_indices:
            self.dones[idx] = True

        self.active_indices = final_active_indices
        self.step_count += 1

        return self.dones

    def get_final_responses(self) -> Dict[str, Any]:
        output_sequences, output_result_masks, output_loop_tags = [], [], []

        # rearrange the responses to the original order
        for idx in range(self.batch_size):
            response_tokens = self.curr_inputs[idx][self.prompts_len[idx]:]
            response_mask = self.result_masks[idx]
            loop_tags = self.loop_tags[idx]

            # Truncate if over max length and replace the last token with EOS
            if len(response_tokens) > self.max_len:
                response_tokens = response_tokens[:self.max_len]
                response_mask = response_mask[:self.max_len]
                loop_tags = loop_tags[:self.max_len]

            if response_tokens[-1] != self.eos_token_id:
                if len(response_tokens) == self.max_len:
                    response_tokens[-1] = self.eos_token_id
                    response_mask[-1] = 1  # Ensure mask is 1 for the EOS token
                    loop_tags[-1] = self.step_count - 1
                elif len(response_tokens) < self.max_len:
                    response_tokens.append(self.eos_token_id)
                    response_mask.append(1)
                    loop_tags.append(loop_tags[-1])
            
            output_sequences.append(response_tokens)
            output_result_masks.append(response_mask)
            output_loop_tags.append(loop_tags)

        padded_responses = pad_2d_list_to_length(output_sequences, self.pad_token_id, self.max_len).to(self.device)
        padded_masks = pad_2d_list_to_length(output_result_masks, 0, self.max_len).to(self.device)
        padded_tags = pad_2d_list_to_length(output_loop_tags, -1, self.max_len).to(self.device)

        return {
            "responses": padded_responses,
            "response_masks": padded_masks,
            "loop_tags": padded_tags,
            "num_turns": self.num_turns
        }
