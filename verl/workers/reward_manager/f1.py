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

from collections import defaultdict
from typing import Any
import re
import torch
from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager
import numpy as np
import math

@register("f1")
class F1RewardManager(AbstractRewardManager):
    """The reward manager."""

    def __init__(self, config, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source") -> None:
        """
        Initialize the NaiveRewardManager instance.

        Args:
            config: PPO trainer configuration object containing reward manager fields.
            tokenizer: The tokenizer used to decode token IDs into text.
            num_examine: The number of batches of decoded responses to print to the console for debugging purpose.
            compute_score: A function to compute the reward score. If None, `default_compute_score` will be used.
            reward_fn_key: The key used to access the data source in the non-tensor batch data. Defaults to
                "data_source".
        """
        self.alpha = config.algorithm.alpha
        self.final_reward_type = config.algorithm.final_reward_type
        self.process_reward_type = config.algorithm.process_reward_type
        self.tokenizer = tokenizer  # Store the tokenizer for decoding token IDs
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key  # Store the key for accessing the data source

    def _check_format_nothink(self, response_str: str, is_main_task: bool) -> bool:
        response_str = response_str.strip()

        if not is_main_task:
            return (
                response_str.startswith("<eval>") and
                response_str.endswith("</eval>") and
                response_str.count("<eval>") == 1 and
                response_str.count("</eval>") == 1
            )

        TRUNCATION_INSTRUCTION = '[SYSTEM WARNING] The maximum limit of tool invocation rounds has been reached. Stop using tools immediately. Derive the final answer from the currently available information. You should first analyze the known facts in <think> tags, and then output the final conclusion within <answer> tags. \n'
        
        response_str = response_str.replace(TRUNCATION_INSTRUCTION, "")

        if not response_str.endswith("</answer>"):
            return False
        
        pattern = r"</?(?:search|python|result|answer)>"
        tags = re.findall(pattern, response_str)

        state = "expect_action"
        
        current_tool = None 

        for tag in tags:
            if state == "expect_action":
                if tag in ["<search>", "<python>"]:
                    state = "expect_end_tool"
                    current_tool = tag[1:-1]
                elif tag == "<answer>":
                    state = "expect_end_answer"
                else:
                    return False

            elif state == "expect_end_tool":
                expected_close = f"</{current_tool}>"
                if tag == expected_close:
                    state = "expect_result"
                else:
                    return False

            elif state == "expect_result":
                if tag == "<result>":
                    state = "expect_end_result"
                else:
                    return False

            elif state == "expect_end_result":
                if tag == "</result>":
                    state = "expect_action"
                else:
                    return False

            elif state == "expect_end_answer":
                if tag == "</answer>":
                    state = "end"
                else:
                    return False
            
            elif state == "end":
                return False

        if state == "end":
            return True
        
        return False

    def _check_format(self, response_str: str, is_main_task: bool) -> bool:
        response_str = response_str.strip()

        if not is_main_task:
            return (
                response_str.startswith("<eval>") and
                response_str.endswith("</eval>") and
                response_str.count("<eval>") == 1 and
                response_str.count("</eval>") == 1
            )

        TRUNCATION_INSTRUCTION = '[SYSTEM WARNING] The maximum limit of tool invocation rounds has been reached. Stop using tools immediately. Derive the final answer from the currently available information. You should first analyze the known facts in <think> tags, and then output the final conclusion within <answer> tags. \n'
        
        response_str = response_str.replace(TRUNCATION_INSTRUCTION, "")

        if not response_str.startswith(("<think>", "<thinking>")):
            return False

        if not response_str.endswith("</answer>"):
            return False
        
        pattern = r"</?(?:think|thinking|search|python|result|answer)>"
        tags = re.findall(pattern, response_str)

        state = "expect_think"
        
        current_tool = None 

        for tag in tags:
            if state == "expect_think":
                if tag in ["<think>", "<thinking>"]:
                    state = "expect_end_think"
                else:
                    return False

            elif state == "expect_end_think":
                if tag in ["</think>", "</thinking>"]:
                    state = "expect_action"
                else:
                    return False

            elif state == "expect_action":
                if tag in ["<search>", "<python>"]:
                    state = "expect_end_tool"
                    current_tool = tag[1:-1]
                elif tag == "<answer>":
                    state = "expect_end_answer"
                else:
                    return False

            elif state == "expect_end_tool":
                expected_close = f"</{current_tool}>"
                if tag == expected_close:
                    state = "expect_result"
                else:
                    return False

            elif state == "expect_result":
                if tag == "<result>":
                    state = "expect_end_result"
                else:
                    return False

            elif state == "expect_end_result":
                if tag == "</result>":
                    state = "expect_think"
                else:
                    return False

            elif state == "expect_end_answer":
                if tag == "</answer>":
                    state = "end"
                else:
                    return False
            
            elif state == "end":
                return False

        if state == "end":
            return True
        
        return False


    def __call__(self, data: DataProto, return_dict: bool = False) -> torch.Tensor | dict[str, Any]:
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                reward_extra_keys = data.meta_info.get("reward_extra_keys", [])
                reward_extra_info = {key: data.non_tensor_batch[key] for key in reward_extra_keys}
                return {"reward_tensor": data.batch["rm_scores"], "reward_extra_info": reward_extra_info}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        already_print_data_sources = {}

        batch_probe_responses = data.non_tensor_batch.get("delta_response", None)
        
        if isinstance(batch_probe_responses, np.ndarray):
            batch_probe_responses = batch_probe_responses.tolist()
        
        batch_loop_tags = data.batch.get("response_tags", None)
        batch_is_main = data.non_tensor_batch.get("is_main", None)

        global_steps = data.meta_info.get('global_steps', None)
        total_steps = data.meta_info.get('total_training_steps', None)
        
        max_coeff = 1
        min_coeff = 0.0

        step_capped = min(global_steps, total_steps)

        if total_steps > 0:
            cosine_factor = 0.5 * (1 + math.cos(math.pi * step_capped / total_steps))
            process_reward_coeff = min_coeff + (max_coeff - min_coeff) * cosine_factor
        else:
            process_reward_coeff = max_coeff
        
        # print("global_steps:", global_steps, "total_steps:", total_steps, "process_reward_coeff", process_reward_coeff)

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            is_main_task = True
            if batch_is_main is not None:
                is_main_task = bool(batch_is_main[i])

            prompt_ids = data_item.batch["prompts"]

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            data_source = data_item.non_tensor_batch[self.reward_fn_key]
            extra_info = data_item.non_tensor_batch.get("extra_info", {})
            num_turns = data_item.non_tensor_batch.get("__num_turns__", None)
            extra_info["num_turns"] = num_turns

            extra_info["is_format_valid"] = self._check_format(response_str, is_main_task)
            
            score = self.compute_score(
                reward_type=self.final_reward_type,
                data_source=data_source,
                solution_str=(
                    f'<answer>{response_str.replace("<eval>", "").replace("</eval>", "").strip()}</answer>'
                    if not is_main_task else response_str
                ),
                ground_truth=ground_truth,
                extra_info=extra_info,
            )

            if isinstance(score, dict):
                reward = score["score"]
                # Store the information including original reward
                for key, value in score.items():
                    reward_extra_info[key].append(value)
            else:
                reward = score

            reward_tensor[i, valid_response_length - 1] = reward

            probe_str = []
            probe_scores = []

            if is_main_task and batch_probe_responses is not None and batch_loop_tags is not None:
                current_probes = batch_probe_responses[i] # List[str]: [Ans0, Ans1, Ans2...]
                current_tags = batch_loop_tags[i]         # Tensor: [tag_token_0, tag_token_1...]
                valid_tags = current_tags[:valid_response_length]

                full_response_masks = data_item.batch["response_mask"]
                valid_gen_masks = full_response_masks[:valid_response_length]
                
                for p_str in current_probes:

                    is_format_valid = extra_info["is_format_valid"]
                    extra_info["is_format_valid"] = True

                    p_score = self.compute_score(
                        reward_type=self.process_reward_type,
                        data_source=data_source,
                        solution_str=f'<answer>{p_str.replace("<eval>", "").replace("</eval>", "").strip()}</answer>',
                        ground_truth=ground_truth,
                        extra_info=extra_info
                    )
                    extra_info["is_format_valid"] = is_format_valid

                    if isinstance(p_score, dict):
                        probe_scores.append(p_score["score"])
                    else:
                        probe_scores.append(p_score)
                    
                    probe_str.append(p_str)

                for k in range(1, len(probe_scores)):
                    delta_reward = probe_scores[k] - probe_scores[k-1]
                    target_tag = k - 1 

                    target_indices_mask = (valid_tags == target_tag) & (valid_gen_masks == 1)

                    indices = target_indices_mask.nonzero(as_tuple=True)[0]
                    
                    if len(indices) > 0:
                        last_token_idx = indices[-1]
                        reward_tensor[i, last_token_idx] += self.alpha * delta_reward
                        # reward_tensor[i, last_token_idx] += process_reward_coeff * delta_reward

            task_type_str = "MAIN" if is_main_task else "PROBE"
            print_key = (data_source, task_type_str)

            if print_key not in already_print_data_sources:
                already_print_data_sources[print_key] = 0

            # if already_print_data_sources[print_key] < self.num_examine:
            num_examine = self.num_examine if is_main_task else 10
            if already_print_data_sources[print_key] < num_examine:
                already_print_data_sources[print_key] += 1

                task_type_str = "MAIN" if is_main_task else "PROBE"
                print(f"--- [Role: {task_type_str}] ---")

                # print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                if isinstance(score, dict):
                    for key, value in score.items():
                        print(f"[{key}]", value)
                else:
                    print("[score]", score)

                if is_main_task and batch_probe_responses is not None and batch_loop_tags is not None:
                    for idx, (pstr, pscore) in enumerate(zip(probe_str, probe_scores)):
                        print(f"[probe {idx} answer]", pstr)
                        print(f"[probe {idx} score]", pscore)
                    # print("[reward_tensor]", reward_tensor.tolist()[i][:valid_response_length])

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor
