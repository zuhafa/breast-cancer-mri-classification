"""
Prediction Service
==================
High-level service for running predictions across all models.
Handles single model predictions and multi-model comparisons.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import time
import logging

from app.core.model_loader import model_manager
from app.services.image_processor import ImageProcessor

logger = logging.getLogger(__name__)


@dataclass
class PredictionResult:
    """Structured prediction result."""
    model_id: str
    model_name: str
    predicted_class: str
    confidence: float
    inference_time_ms: float
    success: bool
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "model_id": self.model_id,
            "model_name": self.model_name,
            "predicted_class": self.predicted_class,
            "confidence": f"{self.confidence * 100:.2f}%",  # Convert to percentage
            "inference_time_ms": round(self.inference_time_ms, 2),
            "success": self.success,
            "error_message": self.error_message
        }


class PredictionService:
    """
    Service for running predictions on MRI images.
    """
    
    @classmethod
    def predict_single(
        cls,
        model_id: str,
        image_data: bytes
    ) -> PredictionResult:
        """
        Run prediction with a single model.
        
        Args:
            model_id: Which model to use
            image_data: Raw image bytes
            
        Returns:
            PredictionResult with all details
        """
        from app.core.config import settings
        
        start_time = time.time()
        
        try:
            # Check if model is loaded
            if not model_manager.is_model_loaded(model_id):
                model_info = settings.MODEL_INFO.get(model_id, {})
                return PredictionResult(
                    model_id=model_id,
                    model_name=model_info.get('name', model_id),
                    predicted_class="",
                    confidence=0.0,
                    inference_time_ms=0.0,
                    success=False,
                    error_message=f"Model '{model_id}' is not loaded. Please ensure the model file is available."
                )
            
            # Preprocess image
            preprocessed = ImageProcessor.preprocess(image_data,model_id)
            
            # Run prediction
            predicted_class, confidence = model_manager.predict(model_id, preprocessed)
            
            confidence = round(float(confidence), 4)

            # Calculate inference time
            inference_time = (time.time() - start_time) * 1000  # Convert to ms
            
            model_info = settings.MODEL_INFO.get(model_id, {})
            
            return PredictionResult(
                model_id=model_id,
                model_name=model_info.get('name', model_id),
                predicted_class=predicted_class,
                confidence=confidence,
                inference_time_ms=inference_time,
                success=True
            )
            
        except Exception as e:
            logger.error(f"Prediction failed for model {model_id}: {e}")
            model_info = settings.MODEL_INFO.get(model_id, {})
            return PredictionResult(
                model_id=model_id,
                model_name=model_info.get('name', model_id),
                predicted_class="",
                confidence=0.0,
                inference_time_ms=0.0,
                success=False,
                error_message=str(e)
            )
    
    @classmethod
    def predict_all(
        cls,
        image_data: bytes,
        model_ids: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Run predictions with all available models for comparison.
        
        Args:
            image_data: Raw image bytes
            model_ids: Optional list of specific models to use (default: all loaded)
            
        Returns:
            Dictionary with all predictions and comparison data
        """
        start_time = time.time()
        
        # Determine which models to use
        if model_ids is None:
            model_ids = list(model_manager.get_loaded_models().keys())
        
        # Run predictions for all models
        results: List[PredictionResult] = []
        for model_id in model_ids:
            result = cls.predict_single(model_id, image_data)
            results.append(result)
        
        # Calculate aggregate statistics
        successful_predictions = [r for r in results if r.success]

        
        
        if successful_predictions:
            # Weighted ensemble voting for stable consensus
            weights = {
                "resnet": 0.20,
                "densenet": 0.30,
                "efficientnet": 0.35,
                "convnext": 0.15
            }

            malignant_score = 0
            benign_score = 0

            for r in successful_predictions:
                weight = weights.get(r.model_id, 0.25)  # Default weight if not specified
                if r.predicted_class == "Malignant":
                    malignant_score += weight * r.confidence
                else:  # Benign
                    benign_score += weight * r.confidence

            # Calculate weighted average confidence across all predictions
            total_confidence = sum(r.confidence for r in successful_predictions)
            avg_confidence = total_confidence / len(successful_predictions) if successful_predictions else 0.0

            # Determine consensus with a confidence margin
            if abs(malignant_score - benign_score) < 0.15:
                consensus_class = "Uncertain"
                # Use the higher score as confidence, or average them
                consensus_confidence = avg_confidence
            elif malignant_score > benign_score:
                consensus_class = "Malignant"
                consensus_confidence = avg_confidence
            else:
                consensus_class = "Benign"
                consensus_confidence = avg_confidence

           
            
            # Model agreement based on the dominance of the winning score
            total_score = malignant_score + benign_score
            agreement = (max(malignant_score, benign_score) / total_score * 100) if total_score > 0 else 0
        else:
            consensus_class = "Unknown"
            consensus_confidence = 0.0
            avg_confidence = 0.0
            agreement = 0.0
        
        total_time = (time.time() - start_time) * 1000
        
        return {
            "predictions": [r.to_dict() for r in results],
            "comparison": {
                "consensus_class": consensus_class,
                "consensus_confidence": f"{consensus_confidence * 100:.2f}%",
                "average_confidence": f"{avg_confidence * 100:.2f}%",
                "model_agreement": round(agreement, 1),
                "models_used": len(model_ids),
                "successful_predictions": len(successful_predictions)
            },
            "total_time_ms": round(total_time, 2)
        }
    
    @classmethod
    def get_available_models(cls) -> Dict[str, Any]:
        """Get information about all available models."""
        from app.core.config import settings
        
        models = []
        for model_id in settings.MODEL_PATHS.keys():
            info = settings.MODEL_INFO.get(model_id, {})
            models.append({
                "id": model_id,
                "name": info.get('name', model_id),
                "description": info.get('description', ''),
                "loaded": model_manager.is_model_loaded(model_id),
                "input_shape": info.get('input_shape', (224, 224, 3))
            })
        
        return {"models": models}
