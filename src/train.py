from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForSeq2Seq
from unsloth import is_bfloat16_supported

from trl import SFTTrainer
from transformers import TrainingArguments
from src.dataset import ClarityDataset
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
LORA_OUT = ROOT_DIR / "models" / "lora" / "new_lora"

def train_model(model, tokenizer, dataset, output_dir=LORA_OUT,
                training_args = TrainingArguments(
                per_device_train_batch_size = 2,
                gradient_accumulation_steps = 4,
                warmup_steps = 5,
                #num_train_epochs = 1, # Set this for 1 full training run.
                max_steps = 100,
                learning_rate = 2e-4,
                fp16 = not is_bfloat16_supported(),
                bf16 = is_bfloat16_supported(),
                logging_steps = 1,
                optim = "adamw_8bit",
                weight_decay = 0.01,
                lr_scheduler_type = "linear",
                seed = 21,
                output_dir = "outputs",
                report_to = "none", # Use this for WandB etc
            )):
    """
    Executes the training loop.
    Args:
        model: Pre-loaded Unsloth model object
        tokenizer: Pre-loaded Unsloth tokenizer
        dataset: Training dataset
        task: 'task_1_clarity' or 'task_2_evasion'
        output_dir: Where to save the adapter
        training_args: TrainingArguments(...) from transformers
    """
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        dataset_text_field = "text",
        max_seq_length = 2048,
        dataset_num_proc = 2,
        packing = False,
        args = training_args,
    )

    trainer.train()
    print(f"Saving adapter to {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)