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
Metrics related to the PPO trainer.
"""

from collections import defaultdict
from functools import partial
from typing import Any, Callable

import numpy as np
import torch

from verl import DataProto
from verl.utils.import_utils import deprecated


@deprecated("verl.utils.metric.reduce_metrics")
def reduce_metrics(metrics: dict[str, list[Any]]) -> dict[str, Any]:
    """
    Reduces a dictionary of metric lists by computing the mean of each list.

    Args:
        metrics: A dictionary mapping metric names to lists of metric values.

    Returns:
        A dictionary with the same keys but with each list replaced by its mean value.

    Example:
        >>> metrics = {"loss": [1.0, 2.0, 3.0], "accuracy": [0.8, 0.9, 0.7]}
        >>> reduce_metrics(metrics)
        {"loss": 2.0, "accuracy": 0.8}
    """
    from verl.utils.metric import reduce_metrics

    return reduce_metrics(metrics)


def _compute_response_info(batch: DataProto) -> dict[str, Any]:
    """
    Computes information about prompts and responses from a batch.

    This is an internal helper function that extracts masks and lengths for prompts and responses.

    Args:
        batch: A DataProto object containing batch data with responses and attention masks.

    Returns:
        A dictionary containing:
            - response_mask: Attention mask for the response tokens
            - prompt_length: Tensor of prompt lengths for each item in the batch
            - response_length: Tensor of response lengths for each item in the batch
    """
    response_length = batch.batch["responses"].shape[-1]

    prompt_mask = batch.batch["attention_mask"][:, :-response_length]
    response_mask = batch.batch["attention_mask"][:, -response_length:]

    prompt_length = prompt_mask.sum(-1).float()
    response_length = response_mask.sum(-1).float()  # (batch_size,)

    return dict(
        response_mask=response_mask,
        prompt_length=prompt_length,
        response_length=response_length,
    )


def compute_data_metrics(batch: DataProto, use_critic: bool = True) -> dict[str, Any]:
    """
    Computes various metrics from a batch of data for PPO training, separating Main and Probe tasks.
    """
    
    # --- 1. 基础数据准备 ---
    sequence_score = batch.batch["token_level_scores"].sum(-1)
    sequence_reward = batch.batch["token_level_rewards"].sum(-1)
    advantages = batch.batch["advantages"]
    returns = batch.batch["returns"]
    
    token_level_rewards = batch.batch["token_level_rewards"]
    prompt_ids = batch.batch["prompts"]
    prompt_len = prompt_ids.shape[-1]
    attention_mask = batch.batch["attention_mask"]
    
    # 计算实际生成的 response 长度
    max_response_length = batch.batch["responses"].shape[-1]
    prompt_mask = batch.batch["attention_mask"][:, :-max_response_length].bool()
    response_mask = batch.batch["response_mask"].bool()
    max_prompt_length = prompt_mask.size(-1)

    response_info = _compute_response_info(batch)
    prompt_length = response_info["prompt_length"]   # Tensor
    response_length = response_info["response_length"] # Tensor

    # 计算 Last Token Reward
    valid_response_lengths = attention_mask[:, prompt_len:].sum(dim=-1)
    indices = (valid_response_lengths - 1).unsqueeze(1)
    indices = torch.clamp(indices, min=0) 
    last_token_rewards = token_level_rewards.gather(1, indices).squeeze(1)

    # --- 2. 构建 Masks (Main vs Probe, Aborted vs Valid) ---
    batch_size = len(batch)
    device = batch.batch['input_ids'].device
    
    # 获取 is_main 标记 (List -> Tensor/Numpy)
    is_main_list = batch.non_tensor_batch.get("is_main", [True] * batch_size)
    
    # Tensor Mask (用于 GPU 上的 Tensor 数据)
    is_main_mask = torch.tensor(is_main_list, dtype=torch.bool, device=device)
    is_probe_mask = ~is_main_mask
    
    # Numpy Mask (用于 non_tensor_batch 数据)
    is_main_numpy = np.array(is_main_list, dtype=bool)
    is_probe_numpy = ~is_main_numpy

    # Aborted Mask (用于过滤长度为0的样本)
    aborted_mask = (response_length == 0).bool()
    non_aborted_mask = ~aborted_mask # 总体非 Aborted

    # 组合 Masks
    # Main 且 非 Aborted
    main_non_aborted_mask = is_main_mask & non_aborted_mask
    # Probe 且 非 Aborted
    probe_non_aborted_mask = is_probe_mask & non_aborted_mask

    # --- 3. 辅助函数：快速添加统计信息 ---
    metrics = {}
    metrics["sample/count"] = int(batch_size)

    def _add_stats(name_prefix, data_tensor):
        """Helper to add mean/max/min/clip_ratio."""
        if data_tensor.numel() == 0:
            return
        
        # 将数据转为 float 以便计算
        if not data_tensor.is_floating_point():
            data_tensor = data_tensor.float()
            
        metrics[f"{name_prefix}/mean"] = torch.mean(data_tensor).detach().item()
        metrics[f"{name_prefix}/max"] = torch.max(data_tensor).detach().item()
        metrics[f"{name_prefix}/min"] = torch.min(data_tensor).detach().item()

    # --- 4. 计算 Critic 相关 (保持总体统计，也可以根据需要拆分) ---
    # 这里保持原逻辑，统计所有非 Aborted 的 Sequence Score/Reward
    non_aborted_sequence_score = sequence_score[non_aborted_mask]
    non_aborted_sequence_reward = sequence_reward[non_aborted_mask]
    
    _add_stats("critic/score", non_aborted_sequence_score)
    # _add_stats("critic/rewards", non_aborted_sequence_reward)

    valid_adv = torch.masked_select(advantages, response_mask)
    valid_returns = torch.masked_select(returns, response_mask)
    
    _add_stats("critic/advantages", valid_adv)
    _add_stats("critic/returns", valid_returns)

    if use_critic:
        values = batch.batch["values"]
        valid_values = torch.masked_select(values, response_mask)
        _add_stats("critic/values", valid_values)
        
        return_diff_var = torch.var(valid_returns - valid_values)
        return_var = torch.var(valid_returns)
        metrics["critic/vf_explained_var"] = (1.0 - return_diff_var / (return_var + 1e-5)).detach().item()

    # --- 5. 分开统计 Last Rewards (Main vs Probe) ---
    # Main
    _add_stats("train/last_rewards", last_token_rewards[main_non_aborted_mask])
    # Probe
    _add_stats("train/last_rewards_probe", last_token_rewards[probe_non_aborted_mask])

    # --- 6. 分开统计 Lengths (Response & Prompt) ---
    
    # 6.1 Response Length (只统计 Non-Aborted 的)
    # Main
    _add_stats("response_length", response_length[main_non_aborted_mask])
    # Probe
    _add_stats("response_length/probe", response_length[probe_non_aborted_mask])

    # Aborted Ratio (总体)
    metrics["response_length/aborted_ratio"] = torch.mean(aborted_mask.float()).detach().item()

    # Clip Ratio (总体)
    non_aborted_resp_len = response_length[non_aborted_mask]
    if non_aborted_resp_len.numel() > 0:
         metrics["response_length/clip_ratio"] = (
            torch.mean(torch.eq(non_aborted_resp_len, max_response_length).float()).detach().item()
        )

    # 6.2 Prompt Length
    _add_stats("prompt_length", prompt_length[is_main_mask])
    _add_stats("prompt_length/probe", prompt_length[is_probe_mask])

    # --- 7. Non-Tensor Batch (Num Turns, F1, EM) ---
    
    def _add_numpy_stats(name_prefix, data_array):
        if len(data_array) == 0:
            return
        metrics[f"{name_prefix}/min"] = data_array.min()
        metrics[f"{name_prefix}/max"] = data_array.max()
        metrics[f"{name_prefix}/mean"] = data_array.mean()

    # 7.1 Num Turns
    if "__num_turns__" in batch.non_tensor_batch:
        num_turns = np.array(batch.non_tensor_batch["__num_turns__"])
        main_num_turns = num_turns[is_main_numpy]
        _add_numpy_stats("train/num_turns", main_num_turns)

    # 7.2 F1 Score
    if "f1" in batch.non_tensor_batch:
        f1_scores = np.array(batch.non_tensor_batch["f1"])
        _add_numpy_stats("train/f1", f1_scores[is_main_numpy])
        _add_numpy_stats("train/f1_probe", f1_scores[is_probe_numpy])

        probe_pos = np.array(batch.non_tensor_batch.get("parent_step", [-1] * batch_size), dtype=int)

        for pos in range(4):
            pos_mask = is_probe_numpy & (probe_pos == pos)
            pos_scores = f1_scores[pos_mask]

            if len(pos_scores) != 0:
                metrics[f"train/f1_probe_pos_{pos}/mean"] = float(pos_scores.mean())

    # 7.3 EM Score
    if "em" in batch.non_tensor_batch:
        em_scores = np.array(batch.non_tensor_batch["em"])
        _add_numpy_stats("train/em", em_scores[is_main_numpy])
        _add_numpy_stats("train/em_probe", em_scores[is_probe_numpy])

        probe_pos = np.array(batch.non_tensor_batch.get("parent_step", [-1] * batch_size), dtype=int)

        for pos in range(4):
            pos_mask = is_probe_numpy & (probe_pos == pos)
            pos_scores = em_scores[pos_mask]

            if len(pos_scores) != 0:
                metrics[f"train/em_probe_pos_{pos}/mean"] = float(pos_scores.mean())

    return metrics


def compute_timing_metrics(batch: DataProto, timing_raw: dict[str, float]) -> dict[str, Any]:
    """
    Computes timing metrics for different processing stages in PPO training.

    This function calculates both raw timing metrics (in seconds) and per-token timing metrics
    (in milliseconds) for various processing stages like generation, reference computation,
    value computation, advantage computation, and model updates.

    Args:
        batch: A DataProto object containing batch data with responses and attention masks.
        timing_raw: A dictionary mapping stage names to their execution times in seconds.

    Returns:
        A dictionary containing:
            - timing_s/{name}: Raw timing in seconds for each stage
            - timing_per_token_ms/{name}: Per-token timing in milliseconds for each stage

    Note:
        Different stages use different token counts for normalization:
        - "gen" uses only response tokens
        - Other stages ("ref", "values", "adv", "update_critic", "update_actor") use all tokens
          (prompt + response)
    """
    response_info = _compute_response_info(batch)
    num_prompt_tokens = torch.sum(response_info["prompt_length"]).item()
    num_response_tokens = torch.sum(response_info["response_length"]).item()
    num_overall_tokens = num_prompt_tokens + num_response_tokens

    num_tokens_of_section = {
        "gen": num_response_tokens,
        **{name: num_overall_tokens for name in ["ref", "values", "adv", "update_critic", "update_actor"]},
    }

    return {
        **{f"timing_s/{name}": value for name, value in timing_raw.items()},
        **{
            f"timing_per_token_ms/{name}": timing_raw[name] * 1000 / num_tokens_of_section[name]
            for name in set(num_tokens_of_section.keys()) & set(timing_raw.keys())
        },
    }


def compute_throughout_metrics(batch: DataProto, timing_raw: dict[str, float], n_gpus: int) -> dict[str, Any]:
    """
    Computes throughput metrics for PPO training.

    This function calculates performance metrics related to token processing speed,
    including the total number of tokens processed, time per step, and throughput
    (tokens per second per GPU).

    Args:
        batch: A DataProto object containing batch data with meta information about token counts.
        timing_raw: A dictionary mapping stage names to their execution times in seconds.
                   Must contain a "step" key with the total step time.
        n_gpus: Number of GPUs used for training.

    Returns:
        A dictionary containing:
            - perf/total_num_tokens: Total number of tokens processed in the batch
            - perf/time_per_step: Time taken for the step in seconds
            - perf/throughput: Tokens processed per second per GPU

    Note:
        The throughput is calculated as total_tokens / (time * n_gpus) to normalize
        across different GPU counts.
    """
    total_num_tokens = sum(batch.meta_info["global_token_num"])
    time = timing_raw["step"]
    # estimated_flops, promised_flops = flops_function.estimate_flops(num_tokens, time)
    # f'Actual TFLOPs/s/GPU​': estimated_flops/(n_gpus),
    # f'Theoretical TFLOPs/s/GPU​': promised_flops,
    return {
        "perf/total_num_tokens": total_num_tokens,
        "perf/time_per_step": time,
        "perf/throughput": total_num_tokens / (time * n_gpus),
    }


def bootstrap_metric(
    data: list[Any],
    subset_size: int,
    reduce_fns: list[Callable[[np.ndarray], float]],
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> list[tuple[float, float]]:
    """
    Performs bootstrap resampling to estimate statistics of metrics.

    This function uses bootstrap resampling to estimate the mean and standard deviation
    of metrics computed by the provided reduction functions on random subsets of the data.

    Args:
        data: List of data points to bootstrap from.
        subset_size: Size of each bootstrap sample.
        reduce_fns: List of functions that compute a metric from a subset of data.
        n_bootstrap: Number of bootstrap iterations. Defaults to 1000.
        seed: Random seed for reproducibility. Defaults to 42.

    Returns:
        A list of tuples, where each tuple contains (mean, std) for a metric
        corresponding to each reduction function in reduce_fns.

    Example:
        >>> data = [1, 2, 3, 4, 5]
        >>> reduce_fns = [np.mean, np.max]
        >>> bootstrap_metric(data, 3, reduce_fns)
        [(3.0, 0.5), (4.5, 0.3)]  # Example values
    """
    np.random.seed(seed)

    bootstrap_metric_lsts = [[] for _ in range(len(reduce_fns))]
    for _ in range(n_bootstrap):
        bootstrap_idxs = np.random.choice(len(data), size=subset_size, replace=True)
        bootstrap_data = [data[i] for i in bootstrap_idxs]
        for i, reduce_fn in enumerate(reduce_fns):
            bootstrap_metric_lsts[i].append(reduce_fn(bootstrap_data))
    return [(np.mean(lst), np.std(lst)) for lst in bootstrap_metric_lsts]


def calc_maj_val(data: list[dict[str, Any]], vote_key: str, val_key: str) -> float:
    """
    Calculate a value based on majority voting.

    This function identifies the most common value for a specified vote key
    in the data, then returns the corresponding value for that majority vote.

    Args:
        data: List of dictionaries, where each dictionary contains both vote_key and val_key.
        vote_key: The key in each dictionary used for voting/counting.
        val_key: The key in each dictionary whose value will be returned for the majority vote.

    Returns:
        The value associated with the most common vote.

    Example:
        >>> data = [
        ...     {"pred": "A", "val": 0.9},
        ...     {"pred": "B", "val": 0.8},
        ...     {"pred": "A", "val": 0.7}
        ... ]
        >>> calc_maj_val(data, vote_key="pred", val_key="val")
        0.9  # Returns the first "val" for the majority vote "A"
    """
    vote2vals = defaultdict(list)
    for d in data:
        vote2vals[d[vote_key]].append(d[val_key])

    vote2cnt = {k: len(v) for k, v in vote2vals.items()}
    maj_vote = max(vote2cnt, key=vote2cnt.get)

    maj_val = vote2vals[maj_vote][0]

    return maj_val


def process_validation_metrics(
    data_sources: list[str], sample_uids: list[str], infos_dict: dict[str, list[Any]], seed: int = 42
) -> dict[str, dict[str, dict[str, float]]]:
    """
    Process validation metrics into a structured format with statistical analysis.

    This function organizes validation metrics by data source and prompt, then computes
    various statistical measures including means, standard deviations, best/worst values,
    and majority voting results. It also performs bootstrap sampling to estimate statistics
    for different sample sizes.

    Args:
        data_sources: List of data source identifiers for each sample.
        sample_uids: List of sample uids corresponding to each sample.
        infos_dict: Dictionary mapping variable names to lists of values for each sample.
        seed: Random seed for bootstrap sampling. Defaults to 42.

    Returns:
        A nested dictionary with the structure:
        {
            data_source: {
                variable_name: {
                    metric_name: value
                }
            }
        }

        Where metric_name includes:
        - "mean@N": Mean value across N samples
        - "std@N": Standard deviation across N samples
        - "best@N/mean": Mean of the best values in bootstrap samples of size N
        - "best@N/std": Standard deviation of the best values in bootstrap samples
        - "worst@N/mean": Mean of the worst values in bootstrap samples
        - "worst@N/std": Standard deviation of the worst values in bootstrap samples
        - "maj@N/mean": Mean of majority voting results in bootstrap samples (if "pred" exists)
        - "maj@N/std": Standard deviation of majority voting results (if "pred" exists)

    Example:
        >>> data_sources = ["source1", "source1", "source2"]
        >>> sample_uids = ["uid1", "uid1", "uid2"]
        >>> infos_dict = {"score": [0.8, 0.9, 0.7], "pred": ["A", "A", "B"]}
        >>> result = process_validation_metrics(data_sources, sample_uids, infos_dict)
        >>> # result will contain statistics for each data source and variable
    """
    # Group metrics by data source, prompt and variable
    data_src2uid2var2vals = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for sample_idx, data_source in enumerate(data_sources):
        uid = sample_uids[sample_idx]
        var2vals = data_src2uid2var2vals[data_source][uid]
        for var_name, var_vals in infos_dict.items():
            var2vals[var_name].append(var_vals[sample_idx])

    # Calculate metrics for each group
    data_src2uid2var2metric = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for data_source, uid2var2vals in data_src2uid2var2vals.items():
        for uid, var2vals in uid2var2vals.items():
            for var_name, var_vals in var2vals.items():
                if isinstance(var_vals[0], str):
                    continue

                metric = {}
                n_resps = len(var_vals)
                metric[f"mean@{n_resps}"] = np.mean(var_vals)

                if n_resps > 1:
                    metric[f"std@{n_resps}"] = np.std(var_vals)

                    ns = []
                    n = 2
                    while n < n_resps:
                        ns.append(n)
                        n *= 2
                    ns.append(n_resps)

                    for n in ns:
                        [(bon_mean, bon_std), (won_mean, won_std)] = bootstrap_metric(
                            data=var_vals, subset_size=n, reduce_fns=[np.max, np.min], seed=seed
                        )
                        metric[f"best@{n}/mean"], metric[f"best@{n}/std"] = bon_mean, bon_std
                        metric[f"worst@{n}/mean"], metric[f"worst@{n}/std"] = won_mean, won_std
                        if var2vals.get("pred", None) is not None:
                            vote_data = [
                                {"val": val, "pred": pred} for val, pred in zip(var_vals, var2vals["pred"], strict=True)
                            ]
                            [(maj_n_mean, maj_n_std)] = bootstrap_metric(
                                data=vote_data,
                                subset_size=n,
                                reduce_fns=[partial(calc_maj_val, vote_key="pred", val_key="val")],
                                seed=seed,
                            )
                            metric[f"maj@{n}/mean"], metric[f"maj@{n}/std"] = maj_n_mean, maj_n_std

                data_src2uid2var2metric[data_source][uid][var_name] = metric

    # Aggregate metrics across uids
    data_src2var2metric2uid_vals = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for data_source, uid2var2metric in data_src2uid2var2metric.items():
        for uid, var2metric in uid2var2metric.items():
            for var_name, metric in var2metric.items():
                for metric_name, metric_val in metric.items():
                    data_src2var2metric2uid_vals[data_source][var_name][metric_name].append(metric_val)

    data_src2var2metric2val = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    for data_source, var2metric2uid_vals in data_src2var2metric2uid_vals.items():
        for var_name, metric2uid_vals in var2metric2uid_vals.items():
            for metric_name, uid_vals in metric2uid_vals.items():
                data_src2var2metric2val[data_source][var_name][metric_name] = np.mean(uid_vals)

    return data_src2var2metric2val
