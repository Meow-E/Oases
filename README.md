# OASES: Outcome-Aligned Search-Evaluation Co-Training for Agentic Search

This repository contains the official code for the paper **OASES: Outcome-Aligned Search-Evaluation Co-Training for Agentic Search**.

## Abstract

Agentic search allows language models to solve knowledge-intensive questions by adaptively acquiring external evidence over multiple steps. Reinforcement learning with verifiable rewards is a natural training paradigm for search agents, but outcome-only rewards are sparse and provide limited credit assignment for intermediate search actions. Existing process-reward methods introduce denser feedback through proxy signals, external evaluators, or likelihood-based information gain, yet these rewards can be misaligned with the outcome objective or inaccurate due to evaluator--policy mismatch. In this paper, we propose **OASES**, an **O**utcome-**A**ligned **Se**arch-**E**valuation **S**upervision framework for agentic search. OASES derives outcome-aligned process rewards by employing an evaluator to measure how well each intermediate search state supports answering the original question. OASES further co-trains a shared-parameter search policy and state evaluator, enabling the evaluator to adapt to the evolving search behavior and provide more accurate process rewards. Experiments on multi-hop QA benchmarks show that OASES consistently outperforms strong RL baselines, with analyses confirming the benefits of outcome-aligned process rewards and search--evaluation co-training.

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

