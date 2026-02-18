from datasets import load_dataset, ClassLabel, Value
from unsloth.chat_templates import get_chat_template
from unsloth import FastLanguageModel
from collections import Counter
import yaml
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
PROMPTS_PATH = ROOT_DIR / "configs" / "prompts.yaml"


def load_prompts(yaml_path=PROMPTS_PATH):
    with open(yaml_path, "r") as f:
        return yaml.safe_load(f)


def transform_dataset_task(dataset, task, technique, tokenizer, yaml_path=PROMPTS_PATH):
    """
    Transforms the dataset by applying the specific prompt technique from the YAML.
    
    Args:
        dataset (DatasetDict or Dataset): The Hugging Face dataset.
        task (str): 'task_1_clarity' or 'task_2_evasion'.
        technique (str): 'default', 'persona', 'chain_of_thought', 'definition_aware'.
        yaml_path (str): Path to prompts.yaml.
        
    Returns:
        Dataset: The dataset with a new 'text' column containing the formatted prompt.
    """
    prompts = load_prompts(yaml_path)

    # Validation
    if task not in prompts:
        raise ValueError(f"Task '{task}' not found in yaml. Available: {list(prompts.keys())}")
    if technique not in prompts[task]:
        raise ValueError(f"Technique '{technique}' not found for {task}. Available: {list(prompts[task].keys())}")
        
    config = prompts[task][technique]
    system_prompt = config['system']
    
    if task == 'task_1_clarity':
        label_column = "clarity_label"
        unique_labels = sorted(set(dataset["train"][label_column]))
    elif task == 'task_2_evasion':
        label_column = "evasion_label"
        unique_labels = sorted(set(dataset["train"][label_column]))
    
    # We do this to get the same distribution for the subset 
    dataset = dataset['train'].cast_column(
        label_column,
        ClassLabel(names=unique_labels)
    )
    # Step 2: Stratified split
    dataset = dataset.train_test_split(
        test_size=0.1,
        stratify_by_column=label_column,
        seed=42,
    )
    dataset = dataset['test']

    # Step 3: Convert labels BACK to strings
    def convert_label_back(example):
        idx = int(example[label_column])   # convert "4" → 4
        example[label_column] = unique_labels[idx]
        return example

    dataset = dataset.cast_column(label_column, Value("string"))
    dataset = dataset.map(convert_label_back)
        
    def format_row(row):
        # 1. Extract inputs
        q = row['interview_question']
        a = row['interview_answer']
        p = row['president']
        d = row['date']
        raw_label = row[label_column]
        
    
        # 2. Format the full string
        # Note: We strip whitespace to prevent tokenization issues at boundaries
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"President: {p}\nDate: {d}\n{q}\n{a}"}, #Q, A are already in the dataset before each Question and Answer respectively
            {"role": "assistant", "content": str(raw_label)}
            ]
        
        # 3. Apply Template
        # tokenize=False gives you the formatted string back so you can debug it
        # add_generation_prompt=False because we include the assistant response (training)
        formatted_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        
        return {"text": formatted_text}

    # Apply transformation
    # remove_columns is optional but recommended if you only need the 'text' for training
    return dataset.map(format_row, remove_columns=dataset.column_names)

def transform_multitask(train_dataset, techniques, tokenizer, yaml_path=PROMPTS_PATH):
    """
    Transforms the dataset by applying the specific prompt technique from the YAML.
    
    Args:
        dataset (Dataset): The Hugging Face dataset.
        techniques (dict): Techniques to be used on each task (default, chain_of_thought, etc).
        yaml_path (str): Path to prompts.yaml.
        
    Returns:
        Dataset: The dataset with a new 'text' column containing the formatted prompt.
    """
    prompts = load_prompts(yaml_path)
        
    task1 = prompts['task_1_clarity'][techniques['task_1_clarity']]
    task2 = prompts['task_2_evasion'][techniques['task_2_evasion']]
    system_prompt_1 = task1['system']
    system_prompt_2 = task2['system']

    # Check columns to remove
    if hasattr(train_dataset, "column_names"):
        cols_to_remove = train_dataset.column_names
    else:
        cols_to_remove = [] # Fallback
    
    def format_batch(batch):
        new_texts = []
        
        # Iterate through the batch
        for i in range(len(batch['interview_question'])):
            q = batch['interview_question'][i]
            a = batch['interview_answer'][i]
            pres = batch['president'][i]
            date = batch['date'][i]
            clarity = batch['clarity_label'][i]
            evasion = batch['evasion_label'][i]

            
            # FIXED: Cleaner context string formatting
            context_parts = []
            if pres is not None:
                context_parts.append(f"President {pres} answered this question")
            if date is not None:
                context_parts.append(f"on {date}")
            

            # 1. Process Clarity Task
            if clarity != "":       
                messages = [
                    {"role": "system", "content": system_prompt_1},
                    {"role": "user", "content": f"President: {pres}\nDate: {date}\n{q}\n{a}"},
                    {"role": "assistant", "content": str(clarity)}
                ]
                formatted_text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False
                )
                new_texts.append(formatted_text)

            # 2. Process Evasion Task
            if evasion != "":       
                messages = [
                    {"role": "system", "content": system_prompt_2},
                    {"role": "user", "content": f"President: {pres}\nDate: {date}\n{q}\n{a}"},
                    {"role": "assistant", "content": str(evasion)}
                ]
                formatted_text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False
                )
                new_texts.append(formatted_text)
            
                
        return {"text": new_texts}

    return train_dataset.map(format_batch, batched=True, remove_columns=cols_to_remove)
