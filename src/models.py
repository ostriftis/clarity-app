from unsloth import FastLanguageModel, AutoModelForCausalLM, AutoTokenizer
import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
BASE_MODEL = ROOT_DIR / "models" / "base"
LORA = ROOT_DIR / "models" / "lora" / "clarity-llm"

def get_model_and_tokenizer_training(model_name="unsloth/mistral-7b-instruct-v0.3-bnb-4bit", model_path= BASE_MODEL, max_seq_length=2048, 
                            dtype=None, load_in_4bit = True, output_dir = BASE_MODEL, save=False):
    """
    Loads the 4-bit Quantized Mistral model with Unsloth optimizations.
    Returns: (model, tokenizer) tuple
    """

    load_target = model_path if (model_path and os.path.exists(model_path)) else model_name
    print(f"Loading model from: {load_target}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = load_target,
        max_seq_length = max_seq_length,
        dtype = dtype,           # Auto-detect (Float16 for T4, Bfloat16 for Ampere)
        load_in_4bit = load_in_4bit,    # Force 4-bit loading
    )

    # Attach LoRA Adapters (Configuration)
    model = FastLanguageModel.get_peft_model(
        model,
        r = 16,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"],
        lora_alpha = 16,
        lora_dropout = 0,       # Optimized for Unsloth
        bias = "none",
        use_gradient_checkpointing = "unsloth",
        random_state = 3407,
        use_rslora = False,
        loqftq_config = None,
    )
    if save:
        print(f"Saving model to {output_dir}...")
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir) # Don't forget the tokenizer!
    return model, tokenizer

def get_model_and_tokenizer_inference(model_name="unsloth/mistral-7b-instruct-v0.3-bnb-4bit", model_path= BASE_MODEL, max_seq_length=2048, 
                            dtype=None, load_in_4bit = True, output_dir = BASE_MODEL, save=False):
    """
    Loads the 4-bit Quantized Mistral model with Unsloth optimizations.
    Returns: (model, tokenizer) tuple
    """
    load_target = model_path if (model_path and os.path.exists(model_path)) else model_name
    print(f"Loading model from: {load_target}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = load_target,
        max_seq_length = max_seq_length,
        dtype = dtype,           # Auto-detect (Float16 for T4, Bfloat16 for Ampere)
        load_in_4bit = load_in_4bit,    # Force 4-bit loading
    )

    FastLanguageModel.for_inference(model)
    
    if save:
        print(f"Saving model to {output_dir}...")
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
    return model, tokenizer

def load_lora(model, path=LORA):
    model.load_adapter(path, adapter_name="lora")
    model.set_adapter("lora")
    return