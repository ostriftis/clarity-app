from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from api.services.inference import run_inference

router = APIRouter(prefix="/predict")

class PredictionRequest(BaseModel):
    question: str
    answer: str
    task: str
    president: str
    date: str

@router.post("/")
def predict(payload: PredictionRequest):
    try:
        # Pydantic validates the payload automatically
        result = run_inference(payload.dict())
        return {"label": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))