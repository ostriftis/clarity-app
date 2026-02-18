# CLARITY‑LLM: Detecting Ambiguity & Evasion in Political Discourse
_A full-stack ML system for the SemEval 2026 CLARITY Task_

## Overview
Political communication is often intentionally ambiguous. In high‑stakes settings such as 
presidential interviews and debates, politicians frequently use evasive strategies that obscure 
whether they actually answered the question. While this phenomenon—known as equivocation or evasion—is
well‑studied in political science, it has received limited attention in computational linguistics.

The SemEval 2026 CLARITY shared task addresses this gap by challenging participants to automatically 
classify the clarity of responses in question–answer (QA) pairs extracted from presidential interviews. 
The task is grounded in the taxonomy introduced by Thomas et al. (2024), presented at EMNLP 2024, which proposes
a two‑level hierarchy:

### Level 1 - Clarity Classification
Classify an answer as:
- Clear Reply
- Ambiguous
- Clear Non‑Reply

### Level 2 - Evasion Technique Classification
Classify an answer into one of 9 fine‑grained evasion strategies, capturing the rhetorical tools politicians 
use to avoid giving a direct answer.

This hierarchical framing enables deeper discourse analysis and has been shown to improve classification performance
when both levels are modeled jointly.

## Approach
This project implements a full ML pipeline for both CLARITY tasks, including training, evaluation, inference, and deployment.

## Model
- **Base model:** Mistral-7b-instruct (4bit)
- **Fine‑tuning**: LoRA (4‑bit quantization)
- **Library**: Unsloth
- **Training**: 3 epochs
- **Hardware‑efficient**: QLoRA + 4‑bit quantization

## Code structure
```
src/
│── models.py        # Model loading, LoRA integration
│── dataset.py       # Dataset preprocessing & loaders
│── evaluate.py      # Evaluation pipeline
│── predict.py       # Inference utilities
main.py              # CLI for train/eval/predict
models/
│── base/            # Cached pretrained model
│── lora/            # Fine‑tuned LoRA adapters
deploy/
│── Dockerfile       # Training/inference image
```

## Results


| Task                   |	Metric |	Score |
| ------                 | --------- | --------- |
| Clarity Classification |	Macro F1 |	0.5187 |
| Evasion Technique Classification |	Macro F1 |	0.3504 |

These results reflect the difficulty of the task—fine‑grained evasion detection is 
significantly more challenging due to subtle rhetorical cues and class imbalance.

## Full-stack Deployment
### Backend - FastAPI
- Loads the fine‑tuned model (base + LoRA or merged)
- Exposes REST endpoints for clarity and evasion predictions
- Handles preprocessing, inference, and post‑processing
- Designed for low‑latency, GPU‑accelerated serving

### Frontend - React / Next.js
- Clean UI for entering QA pairs
- Displays clarity and evasion technique prediction labels
- Communicates with the FastAPI backend via REST

### Containerization
Each component has its own Dockerfile:
- **deploy/Dockerfile** — training & CLI
- **deploy/Dockerfile.api** — FastAPI inference server
- **deploy/Dockerfile.frontend** — Next.js build

A top‑level docker-compose.yml orchestrates:
- API service
- Frontend service
- (Optional) training/inference service

This enables one‑command deployment of the entire system.

## Instructions
