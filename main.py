# main.py
import argparse
from src.models import get_model_and_tokenizer_training, get_model_and_tokenizer_inference, load_lora
from src.train import train_model
from src.dataset import transform_dataset_task, transform_multitask
from src.evaluate import evaluate_model_on_dataset, evaluate_f1_scores
from src.predict import predict_label  # if you have this

from pathlib import Path
# ... other imports ...
from datasets import load_dataset
import sys
import yaml
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported



ROOT_DIR = Path(__file__).resolve()
LORA_OUT = ROOT_DIR / "models" / "lora" /"new_lora"
TRAIN_CONFIG = ROOT_DIR / "configs" / "training.yaml"
EVAL_CONFIG = ROOT_DIR / "configs" / "evaluation.yaml"
PROMPTS_CONFIG = ROOT_DIR / "configs" / "prompts.yaml"
INFER_CONFIG = ROOT_DIR / "configs" / "inference.yaml"

def load_yaml(yaml_path=""):
    with open(yaml_path, "r") as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")

    # -------------------------
    # TRAIN
    # -------------------------
    train_parser = subparsers.add_parser("train")
    # task: "clarity", "evasion", "multitask"
    train_parser.add_argument("--task", type=str, required=True)
    # technique: "default", "persona", "chain_of_thought", "definition_aware"
    train_parser.add_argument("--technique", type=str)
    train_parser.add_argument("--output_dir", type=str, default=str(LORA_OUT))
    #train_parser.add_argument("--subset_size", type=int, default=-1)

    # -------------------------
    # EVALUATE
    # -------------------------
    eval_parser = subparsers.add_parser("evaluate")
    eval_parser.add_argument("--task", type=str, required=True)
    eval_parser.add_argument("--model_path", type=str)
    eval_parser.add_argument("--subset_size", type=int, default=-1)
    eval_parser.add_argument("--technique", type=str)
    eval_parser.add_argument("--lora_path", type=str)
    
    

    # -------------------------
    # PREDICT / INFERENCE
    # -------------------------
    pred_parser = subparsers.add_parser("inference")
    pred_parser.add_argument("--model_path", type=str, required=True)
    pred_parser.add_argument("--lora_path", type=str)
    pred_parser.add_argument("--input_path", type=str)
    # QA args
    pred_parser.add_argument("--pres", type=str, default="")
    pred_parser.add_argument("--date", type=str, default="")
    pred_parser.add_argument("--question", type=str)
    pred_parser.add_argument("--answer", type=str)
    # task
    pred_parser.add_argument("--task", type=str, required=True)

    args = parser.parse_args()

    # -------------------------
    # COMMAND: TRAIN
    # -------------------------
    if args.command == "train":
        # check validity of args
        if args.task != "multitask" and args.technique is None:
            parser.print_help()
            sys.exit(1)

        # load training arguments
        training_config = load_yaml(TRAIN_CONFIG)
        dataset = load_dataset("ailsntua/QEvasion")

        techniques = training_config.get('techniques', {})
        task_map = training_config.get('task_mapping', {})
        technique = args.technique if args.technique is not None else techniques[task_map[args.task]]

        training_dataset = None
        model, tokenizer = get_model_and_tokenizer_training()
        if args.task == 'multi_task':
            training_dataset = transform_multitask(dataset, techniques, tokenizer)
        else:
            training_dataset = transform_dataset_task(dataset, task_map[args.task], technique,
                                         tokenizer)


        tr_args = training_config['training_args']
        
        #safely extract max_steps or epochs
        safe_max_steps = tr_args.get('max_steps', -1)
        safe_epochs = tr_args.get('num_train_epochs', 1)

        training_arguments = TrainingArguments(
                per_device_train_batch_size = tr_args['per_device_train_batch_size'],
                gradient_accumulation_steps = tr_args['gradient_accumulation_steps'],
                warmup_steps = tr_args['warmup_steps'],
                max_steps = safe_max_steps,
                num_train_epochs = safe_epochs, # Set this for 1 full training run.
                learning_rate = tr_args['learning_rate'],
                fp16 = not is_bfloat16_supported(),
                bf16 = is_bfloat16_supported(),
                logging_steps = tr_args['logging_steps'],
                optim = tr_args['optim'],
                weight_decay = tr_args['weight_decay'],
                lr_scheduler_type = tr_args['lr_scheduler_type'],
                seed = tr_args['seed'],
                output_dir = tr_args['output_dir'],
                report_to = tr_args['report_to'],
        )

        train_model(
            model=model,
            tokenizer=tokenizer,
            dataset=training_dataset,
            output_dir=args.output_dir,
            training_args=training_arguments
        )

    # -------------------------
    # COMMAND: EVALUATE
    # -------------------------
    elif args.command == "evaluate":
        dataset = load_dataset("ailsntua/QEvasion")
        test_dataset = dataset['test']
        if args.subset_size is not None: test_dataset = test_dataset.select(range(args.subset_size))

        # load model
        model, tokenizer = get_model_and_tokenizer_inference(model_path=args.model_path)

        # load lora if arg is provided:
        if args.lora_path is not None: load_lora(model, args.lora_path)
        
        # load prompt
        evaluation_config = load_yaml(EVAL_CONFIG)
        task_map = evaluation_config.get('task_mapping', {})
        technique = evaluation_config['evaluation'][task_map[args.task]] if args.technique is None else args.technique
        valid_labels = evaluation_config['valid_labels'].get(task_map[args.task], [])


        print("Begining evaluation on test dataset")
        results = evaluate_model_on_dataset(
            test_dataset = test_dataset,
            task_type=task_map[args.task],
            technique=technique,
            model=model,
            tokenizer=tokenizer,
            task=args.task
        )
        print(f"Evaluating {args.task} Task:")
        f1 = evaluate_f1_scores(results, valid_labels)

    # -------------------------
    # COMMAND: PREDICT
    # -------------------------
    elif args.command == "inference":
        model, tokenizer = get_model_and_tokenizer_inference(model_path=args.model_path)
        
        # load lora if arg is provided:
        if args.lora_path is not None: load_lora(model, args.lora_path)
        if args.input_path is None:
            if not args.question or not args.answer:
                pred_parser.error("If --input_path is not provided, both --question and --answer must be specified.")
            
            q, a, p, d = args.question, args.answer, args.pres, args.date
        
        else:
            data = load_yaml(args.input_path)
            q = data.get('question', args.question)
            a = data.get('answer', args.answer)
            p = data.get('pres', args.pres)
            d = data.get('date', args.date)
            if not q or not a:
                raise ValueError(f"The file {args.input_path} is missing 'question' or 'answer' keys.")
        
        q = "Q " + q
        a = "A " + a


        inference_config = load_yaml(INFER_CONFIG)
        task_map = inference_config['task_mapping']
        technique = inference_config[task_map[args.task]]

        prediction = predict_label(
            question=q,
            answer=a,
            president=p,
            date=d,
            technique=technique,
            task_type=task_map[args.task],
            model=model,
            tokenizer=tokenizer,
        )

        print("Prediction:", prediction)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
