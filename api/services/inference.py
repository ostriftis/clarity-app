import sys
import os

# 1. Get the absolute path of the current file's directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# 2. Go up two levels (from api/routers/ or api/services/ up to project-root/)
project_root = os.path.abspath(os.path.join(current_dir, "../../"))

# 3. Add the project root to Python's internal path
sys.path.append(project_root)

from src.models import get_model_and_tokenizer_inference, load_lora
from src.predict import predict_label
import yaml

#model, tokenizer = get_model_and_tokenizer_inference(model_path="../../models/base",save=True)
#load_lora(model, "../../models/lora/clarity-llm")

def run_inference(payload):
    with open("configs/inference.yaml", "r") as f:
        config = yaml.safe_load(f)

        technique = config.get(str(payload["task"]),"")
    return "Success"
    #return predict_label(
    #    question="Q " + payload["question"],
    #    answer="A " + payload["answer"],
    #    president=payload["president"],
    #    date=payload["date"],
    #    task_type=payload["task"],
    #    technique=technique,
    #    model=model,
    #    tokenizer=tokenizer
    #)
