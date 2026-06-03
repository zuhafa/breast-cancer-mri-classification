"""
Configuration settings for the Breast Cancer MRI Classification API.
Modify the model paths below to point to your trained model files.
"""

from pydantic_settings import BaseSettings
from typing import Dict, List, Set, Any
import os
from pathlib import Path

# Build paths inside the project like this: os.path.join(BASE_DIR, ...)
BASE_DIR = Path(__file__).resolve().parents[3]
MODELS_DIR = BASE_DIR / "models"


class Settings(BaseSettings):
    """Application settings with model configuration."""
    
    # API Configuration
    API_TITLE: str = "Breast Cancer MRI Classification API"
    API_DESCRIPTION: str = "AI-powered breast cancer classification from MRI scans using deep learning"
    API_VERSION: str = "1.0.0"
    
    # CORS Settings
    CORS_ORIGINS: List[str] = ["*"]
    
    # Upload Settings
    UPLOAD_DIR: str = "uploads"
    MAX_FILE_SIZE: int = 10 * 1024 * 1024  # 10MB
    ALLOWED_EXTENSIONS: Set[str] = {".jpg", ".jpeg", ".png", ".dcm"}
    
    # Image Processing Settings
    IMAGE_SIZE: int = 224  # Input size for models
    
    # ============================================================================
    # LOAD TRAINED MODELS HERE
    # ============================================================================
    # Update these paths to point to your trained .keras or .h5 model files
    # Models are loaded once at server startup for efficient inference
    # ============================================================================
    
    MODEL_PATHS: Dict[str, str] = {
        "resnet": str(MODELS_DIR / "resnet_model.keras"),
        "densenet": str(MODELS_DIR / "densenet_model.keras"),
        "efficientnet": str(MODELS_DIR / "efficientnet_model.keras"),
         "convnext": str(MODELS_DIR / "convnext_model.keras"),
    }
    
    # Model Metadata
    MODEL_INFO: Dict[str, Dict[str, Any]] = {
        "resnet": {
            "name": "ResNet-50",
            "description": "Residual Network with skip connections for deep feature extraction",
            "input_shape": (224, 224, 3),
            "version": "1.0"
        },
        "densenet": {
            "name": "DenseNet-121",
            "description": "Densely Connected Convolutional Network with feature reuse",
            "input_shape": (224, 224, 3),
            "version": "1.0"
        },
        "efficientnet": {
            "name": "EfficientNet-B0",
            "description": "Efficiently scaled network with compound scaling",
            "input_shape": (224, 224, 3),
            "version": "1.0"
        },
        "convnext": {
            "name": "ConvNeXt-Tiny",
           "description": "Modern CNN architecture inspired by Transformers",
           "input_shape": (224, 224, 3),
            "version": "1.0"
        }
    }
    
    # Classification Settings
    CLASS_NAMES: List[str] = ["Benign", "Malignant"]
    CONFIDENCE_THRESHOLD: float = 0.4
    
    class Config:
        env_file = ".env"


# Global settings instance
settings = Settings()
