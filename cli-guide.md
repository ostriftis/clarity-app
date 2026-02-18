# Command‑Line Interface Guide
The project provides a unified CLI through main.py with three main commands:
- **train** — fine‑tune the model
- **evaluate** — evaluate a trained model
- **inference** — run predictions on QA pairs or files

## Training
### Command
```bash
python main.py train --task <task> [--technique <technique>] [--output_dir <path>]
```

### Arguments

| Argument	| Required |	Description |
| ------- | ------ | ------- |
| --task	| Yes	| One of: clarity, evasion, multitask |
| --technique	| No	| Prompting strategy: default, persona, chain_of_thought, definition_aware |
| --output_dir |	No	| Where to save LoRA adapters (default: models/lora/) |

### Examples
```bash
python main.py train --task clarity
python main.py train --task evasion --technique chain_of_thought
python main.py train --task multitask --output_dir models/lora_multitask
```
### Config
In `configs/training.yaml` you can change training arguments.

## Evaluation
### Command
```bash
python main.py evaluate --task <task> [--model_path <path>] [--lora_path <path>] [--technique <technique>] [--subset_size <n>]
```
### Arguments

| Argument	| Required |	Description |
| ------- | ------ | ------- |
| --task	| Yes	| `clarity` or `evasion` |
| --model_path	| No	| 	Path to full model (merged or base) |
| --lora_path	| No	| Path to LoRA adapters |
| --technique	| No	| Prompting strategy (same as training) |
| --subset_size |	No	| Evaluate on a subset (default: -1 = full dataset, should be smaller than 308) |

### Examples
```bash
python main.py evaluate --task clarity --lora_path models/lora
python main.py evaluate --task evasion --model_path models/merged
python main.py evaluate --task clarity --lora_path models/lora --subset_size 500
```

## Inference / Prediction
```bash
python main.py inference --task <task> --model_path <path> [--lora_path <path>] [--input_path <file>] [--question <text>] [--answer <text>] [--pres <name>] [--date <date>]
```
### Arguments

| Argument	| Required |	Description |
| ------- | ------ | ------- |
| --task	| Yes	| `clarity` or `evasion` |
| --model_path	| No	| 	Path to full model (merged or base) |
| --lora_path	| No	| Path to LoRA adapters |
| --input_path	| No	| yaml file with QA pair (optionally add president and date) |
| --question	| Yes (if no input file exists)	| Question text (single‑sample inference) |
| --answer | No (if no input file exists) | Answer text |
| --pres |	No	|President name (metadata) |
| --date |	No	| Interview date (metadata) |

### Examples
```bash
python main.py inference \
    --task clarity \
    --model_path models/base \
    --lora_path models/lora \
    --question "Do you support the new policy?" \
    --answer "Well, what I can say is that our administration is working hard..."

python main.py inference \
    --task evasion \
    --model_path models/base \
    --lora_path models/lora \
    --question "Why did the budget increase?" \
    --answer "Let me first talk about our accomplishments this year."

python main.py inference \
    --task clarity \
    --model_path models/base \
    --lora_path models/lora \
    --input_path data/test.json
```


