# OASES: Outcome-Aligned Search-Evaluation Co-Training for Agentic Search

This repository contains the official code for the paper **OASES: Outcome-Aligned Search-Evaluation Co-Training for Agentic Search**.

## Abstract

Reinforcement learning with verifiable rewards has become a promising paradigm for training large language model agents on multi-step search tasks. However, agentic search remains challenging because sparse outcome rewards provide supervision only after the final answer is produced, making it difficult to credit intermediate search decisions. Recent process-reward methods provide denser feedback, but they often rely on proxy evaluators or auxiliary signals whose supervision can be difficult to verify, misaligned with the outcome objective, or unreliable as the search policy evolves. These limitations suggest that effective agentic search requires jointly improving both search behavior and the evaluation of intermediate search states.

To this end, we propose **OASES**, an **O**utcome-**A**ligned **S**earch-**E**valuation **S**upervision framework for agentic search. OASES derives process rewards from outcome improvements across intermediate search states, converting sparse outcome supervision into dense step-level feedback while keeping every intermediate reward tied to the same verifiable objective. Beyond reward construction, OASES performs search-evaluation co-training, enabling the agent to learn not only how to search but also how to assess whether its current search state is useful for solving the task. This search-evaluation co-training avoids a separate process reward model, reduces evaluator-policy mismatch, and keeps process supervision aligned with the evolving policy. Experiments on multi-hop question answering benchmarks show that OASES effectively mitigates reward sparsity and improves agentic search performance.

## Overview

OASES is designed for training and evaluating agentic search policies with dense, outcome-aligned supervision. The repository includes:

- training code for OASES-style agentic search optimization
- evaluation code for multi-hop question answering benchmarks
- search and intermediate-state evaluation components used in co-training
- scripts for environment setup, training, and evaluation

## Installation

```bash
conda create -n oases python=3.10 -y
conda activate oases
cd /oases
bash scripts/install_vllm_sglang_mcore_async.sh
pip3 install --no-deps -e .
```

## Quick Start

After installation, you can use the provided scripts as the main entry points:

```bash
bash run_train.sh
```

```bash
bash run_eval.sh
```

Please review the configuration values in these scripts before launching jobs, especially paths, cluster settings, and service endpoints.

## Repository Structure

The most relevant directories and files are:

- `run_train.sh`: example training entry for OASES
- `run_eval.sh`: example evaluation entry for OASES
- `verl/`: core training and rollout framework code
- `scripts/`: installation and utility scripts
- `data/`: dataset-related assets and prepared inputs

