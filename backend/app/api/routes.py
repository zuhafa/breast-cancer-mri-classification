"""
API Routes
==========
FastAPI routes for the Breast Cancer MRI Classification API.
"""

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from typing import Optional, Dict, Any
import logging

from app.services.prediction_service import PredictionService
from app.services.image_processor import ImageProcessor
from app.core.model_loader import model_manager
from app.core.config import settings

logger = logging.getLogger(__name__)

# Create router
api_router = APIRouter(prefix="/api/v1")


@api_router.get("/")
async def root() -> Dict[str, Any]:
    """API root endpoint."""
    return {
        "message": "Breast Cancer MRI Classification API",
        "version": settings.API_VERSION,
        "documentation": "/docs"
    }


@api_router.get("/health")
async def health_check() -> Dict[str, Any]:
    """Health check endpoint."""
    loaded_models = model_manager.get_loaded_models()
    return {
        "status": "healthy",
        "models_loaded": len(loaded_models),
        "available_models": loaded_models
    }


@api_router.get("/models")
async def list_models() -> Dict[str, Any]:
    """List all available models and their status."""
    return PredictionService.get_available_models()


@api_router.post("/predict/{model_id}")
async def predict_single(
    model_id: str,
    file: UploadFile = File(...)
) -> Dict[str, Any]:
    """
    Run prediction with a specific model.
    
    Args:
        model_id: Model identifier (resnet, densenet, efficientnet, convnext)
        file: Uploaded MRI image file
        
    Returns:
        Prediction result with class and confidence
    """
    # Validate model ID
    if model_id not in settings.MODEL_PATHS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model ID. Available models: {list(settings.MODEL_PATHS.keys())}"
        )
    
    # Validate file
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Please upload an image file."
        )
    
    try:
        # Read image data
        image_data = await file.read()
        
        # Validate image
        is_valid, error_msg = ImageProcessor.validate_image(image_data)
        if not is_valid:
            raise HTTPException(status_code=400, detail=error_msg)
        
        # Run prediction
        result = PredictionService.predict_single(model_id, image_data)
        
        if not result.success:
            raise HTTPException(status_code=500, detail=result.error_message)
        
        return result.to_dict()
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@api_router.post("/predict")
async def predict_with_model(
    model: str = Form(...),
    file: UploadFile = File(...)
) -> Dict[str, Any]:
    """
    Alternative prediction endpoint using form data.
    
    Args:
        model: Model identifier
        file: Uploaded MRI image file
        
    Returns:
        Prediction result
    """
    return await predict_single(model, file)


@api_router.post("/compare")
async def compare_models(
    file: UploadFile = File(...),
    models: Optional[str] = Form(None)
) -> Dict[str, Any]:
    """
    Compare predictions from multiple models.
    
    Args:
        file: Uploaded MRI image file
        models: Comma-separated list of model IDs (default: all loaded models)
        
    Returns:
        Comparison results from all specified models
    """
    # Validate file
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Please upload an image file."
        )
    
    try:
        # Read image data
        image_data = await file.read()
        
        # Validate image
        is_valid, error_msg = ImageProcessor.validate_image(image_data)
        if not is_valid:
            raise HTTPException(status_code=400, detail=error_msg)
        
        # Parse model list if provided
        model_ids = None
        if models:
            model_ids = [m.strip() for m in models.split(",")]
            # Validate all model IDs
            invalid_models = [m for m in model_ids if m not in settings.MODEL_PATHS]
            if invalid_models:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid model IDs: {invalid_models}"
                )
        
        # Run comparison
        results = PredictionService.predict_all(image_data, model_ids)
        
        return results
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Comparison error: {e}")
        raise HTTPException(status_code=500, detail=f"Comparison failed: {str(e)}")


@api_router.post("/analyze")
async def analyze_image(
    file: UploadFile = File(...)
) -> Dict[str, Any]:
    """
    Comprehensive analysis with all models.
    
    Args:
        file: Uploaded MRI image file
        
    Returns:
        Complete analysis with all model predictions and consensus
    """
    return await compare_models(file, None)
