from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import Dict
import logging

from app.services.data_service import DataService
from src.predictions.compare_models_top1_quali import main as train_top1_quali
from src.predictions.compare_models_top1_pre_quali import main as train_top1_pre_quali
from src.predictions.compare_models_top3_quali import main as train_top3_quali
from src.predictions.compare_models_top3_pre_quali import main as train_top3_pre_quali

router = APIRouter()
data_service = DataService()
logger = logging.getLogger(__name__)

@router.post("/retrain")
async def retrain_models(background_tasks: BackgroundTasks):
    try:
        background_tasks.add_task(run_model_retraining)
        return {"status": "started", "message": "Model retraining started in background"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/update-data")
async def update_data():
    try:
        result = await data_service.update_data()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def run_model_retraining():
    try:
        logger.info("Starting model retraining...")
        
        train_top1_quali()
        logger.info("Top-1 quali training complete")
        
        train_top1_pre_quali()
        logger.info("Top-1 pre-quali training complete")
        
        train_top3_quali()
        logger.info("Top-3 quali training complete")
        
        train_top3_pre_quali()
        logger.info("Top-3 pre-quali training complete")
        
        logger.info("All model retraining complete")
        
    except Exception as e:
        logger.error(f"Training error: {e}", exc_info=True)