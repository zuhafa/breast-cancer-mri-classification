"""
Image Preprocessing Service
===========================
Handles all image preprocessing for model inference.
Converts uploaded images to the format expected by the models.
"""
import numpy as np
from PIL import Image
import io
from typing import Tuple, Union
import logging
from app.core.config import settings

# Use tensorflow directly to prevent Pylance "unresolved import" squiggles
import tensorflow as tf

logger = logging.getLogger(__name__)

try:
    # Modern Pillow >= 9.1.0
    RESAMPLE_FILTER = Image.Resampling.LANCZOS
except AttributeError:
    # Older Pillow versions
    RESAMPLE_FILTER = Image.LANCZOS  # type: ignore

class ImageProcessor:
    """
    Image preprocessing pipeline for MRI scan classification.
    """
    
    TARGET_SIZE: Tuple[int, int] = (settings.IMAGE_SIZE, settings.IMAGE_SIZE)
    
    @classmethod
    def preprocess(cls, image_data: Union[bytes, Image.Image], model_id: str) -> np.ndarray:
        """
        Preprocess an image for model inference.
        
        Args:
            image_data: Raw image bytes or PIL Image
            
        Returns:
            Preprocessed image array of shape (1, 224, 224, 3)
        """
        # Load image if bytes provided
        if isinstance(image_data, bytes):
            image = cls._load_from_bytes(image_data)
        else:
            image = image_data
        
        # Apply preprocessing pipeline
        image = cls._convert_to_rgb(image)
        image = cls._resize(image)
        image_array = cls._normalize(image,model_id)
        image_array = cls._add_batch_dimension(image_array)
        
        return image_array
    
    @classmethod
    def _load_from_bytes(cls, image_bytes: bytes) -> Image.Image:
        """Load image from bytes."""
        try:
            return Image.open(io.BytesIO(image_bytes))
        except Exception as e:
            logger.error(f"Failed to load image: {e}")
            raise ValueError("Invalid image file. Please upload a valid image.")
    
    @classmethod
    def _convert_to_rgb(cls, image: Image.Image) -> Image.Image:
        """Convert image to RGB format."""
        if image.mode != 'RGB':
            # Convert grayscale or RGBA to RGB
            if image.mode == 'L':
                # Grayscale - convert to RGB
                image = image.convert('RGB')
            elif image.mode in ('RGBA', 'P'):
                # Handle transparency
                background = Image.new('RGB', image.size, (255, 255, 255))
                if image.mode == 'P':
                    image = image.convert('RGBA')
                background.paste(image, mask=image.split()[-1] if image.mode == 'RGBA' else None)
                image = background
            else:
                image = image.convert('RGB')
        return image
    
    @classmethod
    def _resize(cls, image: Image.Image) -> Image.Image:
        """Resize image to target size using high-quality resampling."""
        return image.resize(cls.TARGET_SIZE, RESAMPLE_FILTER)
    
    
    @classmethod
    def _normalize(cls, image: Image.Image, model_id: str) -> np.ndarray:
        image_array = np.array(image).astype(np.float32)

        if model_id == "resnet":
            image_array = tf.keras.applications.resnet50.preprocess_input(image_array)

        elif model_id == "densenet":
            image_array = tf.keras.applications.densenet.preprocess_input(image_array)

        elif model_id == "efficientnet":
            image_array = tf.keras.applications.efficientnet.preprocess_input(image_array)     

        elif model_id == "convnext":
            image_array = image_array / 255.0
        else:
            image_array = image_array / 255.0
        return image_array

    @classmethod
    def _add_batch_dimension(cls, image_array: np.ndarray) -> np.ndarray:
        """Add batch dimension for model inference."""
        return np.expand_dims(image_array, axis=0)
    
    @classmethod
    def get_preview(cls, image_data: Union[bytes, Image.Image], size: Tuple[int, int] = (300, 300)) -> Image.Image:
        """
        Generate a preview image for display.
        
        Args:
            image_data: Raw image bytes or PIL Image
            size: Preview size
            
        Returns:
            Resized preview image
        """
        if isinstance(image_data, bytes):
            image = cls._load_from_bytes(image_data)
        else:
            image = image_data
        
        # Convert to RGB for preview
        image = cls._convert_to_rgb(image)
        
        # Resize maintaining aspect ratio
        image.thumbnail(size, RESAMPLE_FILTER)
        
        return image
    
    @classmethod
    def validate_image(cls, image_data: bytes) -> Tuple[bool, str]:
        """
        Validate uploaded image.
        
        Args:
            image_data: Raw image bytes
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Check file size
            if len(image_data) > settings.MAX_FILE_SIZE:
                return False, f"File too large. Maximum size is {settings.MAX_FILE_SIZE / (1024*1024):.1f}MB"
            
            # Try to load image
            image = cls._load_from_bytes(image_data)
            
            # Check dimensions
            if image.width < 50 or image.height < 50:
                return False, "Image too small. Minimum dimensions are 50x50 pixels"
            
            return True, ""
            
        except Exception as e:
            return False, f"Invalid image file: {str(e)}"


# Convenience function
def preprocess_image(image_data: Union[bytes, Image.Image], model_id: str) -> np.ndarray:
    """Preprocess image for model inference."""
    return ImageProcessor.preprocess(image_data, model_id)
