# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 Search-R1 Contributors
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
# Adapted from https://github.com/PeterGriffinJin/Search-R1/blob/main/verl/utils/reward_score/qa_em.py

import random
import re
import string
from collections import Counter


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def extract_solution(solution_str):
    answer_pattern = r"<answer>(.*?)</answer>"
    match = re.search(answer_pattern, solution_str, re.DOTALL)

    if match:
        return match.group(1).strip()
    else:
        return None

def em_check(prediction, golden_answers, format_score=0.0, score=1.0):
    if prediction is None:
        return format_score

    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    s = format_score
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer == normalized_prediction:
            s = score
            break
    return s

def f1_score(prediction: str, golden_answers: str, format_score=0.0, score=1.0) -> float:
    if prediction is None:
        return format_score
        
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    pred_tokens = normalize_answer(prediction).split()

    final_s = format_score
    for golden_answer in golden_answers:
        s = format_score

        gt_tokens = normalize_answer(golden_answer).split()
        
        common = Counter(pred_tokens) & Counter(gt_tokens)
        num_same = sum(common.values())

        if len(pred_tokens) == 0 or len(gt_tokens) == 0:
            s = score if pred_tokens == gt_tokens else format_score
        elif num_same == 0:
            s = format_score
        else:
            precision = num_same / len(pred_tokens)
            recall = num_same / len(gt_tokens)
            s = (2 * precision * recall) / (precision + recall)
            s *= score

        if s > final_s:
            final_s = s

    return final_s

def compute_score(solution_str, ground_truth, reward_type="em", method="strict", format_score=0.0, score=1.0, is_format_valid=True):
    """The scoring function for exact match (EM).

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    answer = extract_solution(solution_str=solution_str)
    # do_print = random.randint(1, 64) == 1
    do_print = False
    
    if do_print:
        print("--------------------------------")
        print(f"Golden answers: {ground_truth}")
        if answer is not None:
            print(f"Extracted answer is not None: {answer}")
        else:
            print("Extracted answer: None!")
        print(f"Solution string: {solution_str}")


    em = em_check(answer, ground_truth, format_score, score)
    f1 = f1_score(answer, ground_truth, format_score, score)

    if not is_format_valid:
        return {"em": em, "f1": f1, "score": -1}
    elif answer is None:
        return {"em": em, "f1": f1, "score": -1}
    else:
        return {
            "em": em, 
            "f1": f1, 
            "score": {"em": em, "f1": f1}.get(reward_type, format_score)
        }