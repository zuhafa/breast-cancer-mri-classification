"""
Breast Cancer MRI Classification API
====================================
FastAPI application for AI-powered breast cancer classification.

This application loads four deep learning models at startup:
- ResNet-based CNN
- DenseNet-based CNN
- EfficientNet-based CNN
- ConvNeXt (modern CNN architecture)

Usage:
    uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import logging
import os
import random

import numpy as np
import tensorflow as tf
from contextlib import asynccontextmanager

# --------------------------------------------------------------------------
# Set seeds for reproducibility
# --------------------------------------------------------------------------
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["PYTHONHASHSEED"] = "42"
tf.random.set_seed(42)
np.random.seed(42)
random.seed(42)
# --------------------------------------------------------------------------

from app.core.config import settings
from app.core.model_loader import initialize_models
from app.api.routes import api_router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler.
    Loads models at startup and cleanup at shutdown.
    """
    # Startup: Load all models
    logger.info("=" * 60)
    logger.info("Starting Breast Cancer MRI Classification API")
    logger.info("=" * 60)
    
    # Initialize and load models
    model_status = initialize_models()
    
    loaded_count = sum(1 for loaded in model_status.values() if loaded)
    logger.info(f"Server ready with {loaded_count} models loaded")
    logger.info("=" * 60)
    
    yield
    
    # Shutdown
    logger.info("Shutting down API server")


# Create FastAPI application
app = FastAPI(
    title=settings.API_TITLE,
    description=settings.API_DESCRIPTION,
    version=settings.API_VERSION,
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(api_router, prefix="/api")

# Mount static files (for frontend)
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
static_dir = os.path.join(base_dir, "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Serve frontend
frontend_dir = os.path.join(base_dir, "frontend", "dist")

@app.get("/")
async def serve_frontend():
    """Serve the frontend application."""
    if not os.path.exists(frontend_dir):
        return {"message": "API is running", "docs": "/docs", "status": "Frontend build not found"}
        
    index_path = os.path.join(frontend_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "Breast Cancer MRI Classification API", "docs": "/docs"}
    

@app.get("/{path:path}")
async def serve_frontend_routes(path: str):
    """Serve frontend routes."""
    index_path = os.path.join(frontend_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "Path not found", "path": path}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
