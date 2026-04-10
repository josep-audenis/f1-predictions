from fastapi import APIRouter, HTTPException
from typing import Dict, List
import json
from pathlib import Path

from app.core.config import settings

router = APIRouter()

@router.get("/performance/top1-quali")
async def get_top1_quali_performance():
    try:
        model_path = settings.model_performance_top1_quali
        
        if not model_path.exists():
            raise HTTPException(status_code=404, detail="Model performance file not found")
        
        with open(model_path, "r") as f:
            performance = json.load(f)
        
        return performance
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/performance/top1-pre-quali")
async def get_top1_pre_quali_performance():
    try:
        model_path = settings.model_performance_top1_pre_quali
        
        if not model_path.exists():
            raise HTTPException(status_code=404, detail="Model performance file not found")
        
        with open(model_path, "r") as f:
            performance = json.load(f)
        
        return performance
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# @router.get("/best")
# async def get_best_models():
#     try:
#         models_info = {}
        
#         if settings.model_performance_top1_quali.exists():
#             with open(settings.model_performance_top1_quali, "r") as f:
#                 perf = json.load(f)
#             best_year = max(perf.keys())
#             best_model = max(perf[best_year], key=lambda m: perf[best_year][m]["accuracy"])
#             models_info["top1_quali"] = {
#                 "model": best_model,
#                 "year": best_year,
#                 "accuracy": perf[best_year][best_model]["accuracy"]
#             }
        
#         if settings.model_performance_top1_pre_quali.exists():
#             with open(settings.model_performance_top1_pre_quali, "r") as f:
#                 perf = json.load(f)
#             best_year = max(perf.keys())
#             best_model = max(perf[best_year], key=lambda m: perf[best_year][m]["accuracy"])
#             models_info["top1_pre_quali"] = {
#                 "model": best_model,
#                 "year": best_year,
#                 "accuracy": perf[best_year][best_model]["accuracy"]
#             }
        
#         return models_info
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

@router.get("/available")
async def get_available_models():
    try:
        models = {
            "top1": {
                "quali": [],
                "pre-quali": []
            }
        }
        
        quali_path = settings.models_dir / "top1" / "quali"
        if quali_path.exists():
            models["top1"]["quali"] = [f.stem.replace("_model", "") for f in quali_path.glob("*.pkl")]
        
        pre_quali_path = settings.models_dir / "top1" / "pre-quali"
        if pre_quali_path.exists():
            models["top1"]["pre-quali"] = [f.stem.replace("_model", "") for f in pre_quali_path.glob("*.pkl")]
        
        return models
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))