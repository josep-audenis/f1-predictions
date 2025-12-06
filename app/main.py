from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import logging

from app.core.config import settings
from app.api.routes import predictions, training, races, models

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

app = FastAPI(
    title=settings.project_name,
    version=settings.version,
    openapi_url=f"{settings.api_prefix}/openapi.json"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(
    predictions.router,
    prefix=f"{settings.api_prefix}/predictions",
    tags=["predictions"]
)
app.include_router(
    training.router,
    prefix=f"{settings.api_prefix}/training",
    tags=["training"]
)
app.include_router(
    races.router,
    prefix=f"{settings.api_prefix}/races",
    tags=["races"]
)
app.include_router(
    models.router,
    prefix=f"{settings.api_prefix}/models",
    tags=["models"]
)

@app.get("/")
async def root():
    return {
        "message": "F1 Predictions API",
        "version": settings.version,
        "docs": f"{settings.api_prefix}/docs"
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)