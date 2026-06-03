"""
Model Loader Module
==================
Handles loading and caching of all trained deep learning models at server startup.
Models are loaded once and reused for all prediction requests.
"""

import tensorflow as tf
import numpy as np
from typing import Dict, Optional, Tuple, Any
import logging
from pathlib import Path

from app.core.config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelManager:
    """
    Singleton manager for all ML models.
    Ensures models are loaded once and reused across requests.
    """
    
    _instance: Optional['ModelManager'] = None
    _models: Dict[str, tf.keras.Model] = {}
    _model_status: Dict[str, bool] = {}
    
    def __new__(cls) -> 'ModelManager':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def load_all_models(self) -> Dict[str, bool]:
        """
        Load all configured models at server startup.
        
        Returns:
            Dictionary mapping model IDs to load success status
        """
        logger.info("=" * 60)
        logger.info("LOADING TRAINED MODELS")
        logger.info("=" * 60)
        
        for model_id, model_path in settings.MODEL_PATHS.items():
            self._load_single_model(model_id, model_path)
        
        # Print summary
        loaded_count = sum(self._model_status.values())
        total_count = len(settings.MODEL_PATHS)
        logger.info("-" * 60)
        logger.info(f"Model Loading Complete: {loaded_count}/{total_count} models loaded")
        logger.info("=" * 60)
        
        return self._model_status.copy()
    
    def _load_single_model(self, model_id: str, model_path: str) -> bool:
        """
        Load a single model from disk.
        
        Args:
            model_id: Unique identifier for the model
            model_path: Path to the .keras or .h5 model file
            
        Returns:
            True if loading succeeded, False otherwise
        """
        try:
            path = Path(model_path)
            
            # Check if file exists
            if not path.exists():
                logger.warning(f"[{model_id}] Model file not found: {model_path}")
                logger.warning(f"[{model_id}] Please place your trained model at this location")
                self._model_status[model_id] = False
                return False
            
            # Load the model
            logger.info(f"[{model_id}] Loading model from: {model_path}")
            model = tf.keras.models.load_model(model_path, compile=False, safe_mode=False)
            
            # Store model
            self._models[model_id] = model
            self._model_status[model_id] = True
            
            # Log model info
            model_info = settings.MODEL_INFO.get(model_id, {})
            logger.info(f"[{model_id}] ✓ Successfully loaded: {model_info.get('name', model_id)}")
            logger.info(f"[{model_id}]   Input shape: {model_info.get('input_shape', 'Unknown')}")
            logger.info(f"[{model_id}]   Parameters: {model.count_params():,}")
            
            return True
            
        except Exception as e:
            logger.error(f"[{model_id}] ✗ Failed to load model: {str(e)}")
            self._model_status[model_id] = False
            return False
    
    def get_model(self, model_id: str) -> Optional[tf.keras.Model]:
        """
        Retrieve a loaded model by ID.
        
        Args:
            model_id: The model identifier
            
        Returns:
            The Keras model if loaded, None otherwise
        """
        return self._models.get(model_id)
    
    def is_model_loaded(self, model_id: str) -> bool:
        """Check if a specific model is loaded and ready."""
        return self._model_status.get(model_id, False)
    
    def get_loaded_models(self) -> Dict[str, str]:
        """Get list of successfully loaded model IDs and their names."""
        return {
            model_id: settings.MODEL_INFO.get(model_id, {}).get('name', model_id)
            for model_id, loaded in self._model_status.items()
            if loaded
        }
    
    def get_model_status(self) -> Dict[str, Dict[str, Any]]:
        """Get detailed status of all models."""
        return {
            model_id: {
                "loaded": loaded,
                "name": settings.MODEL_INFO.get(model_id, {}).get('name', model_id),
                "description": settings.MODEL_INFO.get(model_id, {}).get('description', ''),
            }
            for model_id, loaded in self._model_status.items()
        }
    
    def predict(self, model_id: str, preprocessed_image: np.ndarray) -> Tuple[str, float]:
        """
        Run prediction on a preprocessed image, averaging multiple runs for stability.
        
        Args:
            model_id: Which model to use
            preprocessed_image: Preprocessed image array ready for inference
            
        Returns:
            Tuple of (predicted_class, confidence_score)
        """
        model = self.get_model(model_id)
        if model is None:
            raise ValueError(f"Model '{model_id}' is not loaded")
        
        # Run inference multiple times for stability
        preds = []
        for _ in range(5):  # run inference 5 times
            prediction = model(preprocessed_image, training=False).numpy()

            # Handle sigmoid output
            if prediction.shape[-1] == 1:
                p = float(prediction[0][0])
            # Handle softmax output
            else:
                p = float(prediction[0][1])

            preds.append(p)
            
        # Average the raw predictions
        confidence = float(np.clip(np.mean(preds), 0.0, 1.0))
        print(f"{model_id} raw predictions: {preds}")
        print(f"{model_id} averaged confidence: {confidence}")
        
       

        if confidence >= 0.40:
            predicted_class = "Malignant"
            final_confidence = confidence

        else:
            predicted_class = "Benign"
            final_confidence = 1.0 - confidence
        
        return predicted_class, float(final_confidence)

# Global model manager instance
model_manager = ModelManager()


def initialize_models() -> Dict[str, bool]:
    """
    Initialize and load all models at application startup.
    Call this function when the server starts.
    """
    return model_manager.load_all_models()
