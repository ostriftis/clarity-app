from dataset import load_prompts
from tqdm import tqdm

def predict_label(question, answer, president, date, task_type, technique, model, tokenizer):
    """Generates a classification label for a given question/answer pair."""
    
    # Set the correct system prompt based on the task
    prompts = load_prompts("../configs/prompts.yaml")
    
    sys_prompt = prompts[task_type][technique]
    messages = [
        {"role": "system", "content": sys_prompt['system']},
        {"role": "user", "content": f"President: {president}\nDate: {date}\n{question}\n{answer}"},
    ]

    # Tokenize and format with the chat template
    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048,
    ).to("cuda")
    # Generate the classification
    outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],  # CRITICAL: Pass the attention mask
            max_new_tokens=50,  # Reduced since we expect short labels
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    
    # Slice the output tensor to isolate only the newly generated tokens
    input_length = inputs["input_ids"].shape[1]
    new_tokens = outputs[0][input_length:]


    # Slice the output tensor to isolate only the newly generated tokens
    # This is much cleaner and less error-prone than string splitting!
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()