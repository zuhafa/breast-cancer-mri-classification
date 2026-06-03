# Model Integration Guide

This guide explains how to integrate your trained deep learning models into the Breast Cancer MRI Classification platform.

## Overview

The platform is designed to load models once at server startup and reuse them for all predictions. This ensures efficient inference and minimal latency.

## Model Requirements

### Supported Formats
- `.keras` (Keras native format, recommended)
- `.h5` (HDF5 format)
- SavedModel format (directory)

### Input Specifications
All models must accept:
- **Shape**: `(batch_size, 224, 224, 3)`
- **Type**: RGB images as float32 tensors
- **Preprocessing**: ImageNet normalization applied automatically

### Output Specifications
Models should output:
- **Binary classification**: Single value (0-1) or two-class softmax
- **Class order**: `[Benign, Malignant]` or single probability for Malignant

## Integration Steps

### Step 1: Export Your Model

#### From TensorFlow/Keras

```python
import tensorflow as tf

# After training
model.save('models/resnet_model.keras')  # Recommended format
# or
model.save('models/resnet_model.h5')     # HDF5 format
```

#### Verify Model Output

```python
import numpy as np

# Test with dummy input
test_input = np.random.randn(1, 224, 224, 3).astype(np.float32)
prediction = model.predict(test_input)

print(f"Output shape: {prediction.shape}")
print(f"Output range: {prediction.min():.4f} to {prediction.max():.4f}")
```

### Step 2: Place Model Files

Copy your model files to the `backend/models/` directory:

```bash
# Create models directory if needed
mkdir -p backend/models

# Copy your models
cp /path/to/resnet_model.keras backend/models/
cp /path/to/densenet_model.keras backend/models/
cp /path/to/efficientnet_model.keras backend/models/
cp /path/to/convnext_model.keras backend/models/
```

### Step 3: Configure Model Paths

Edit `backend/app/core/config.py`:

```python
# LOAD TRAINED MODELS HERE
# ============================================================================
MODEL_PATHS = {
    "resnet": "models/resnet_model.keras",      # Your ResNet model
    "densenet": "models/densenet_model.keras",  # Your DenseNet model
    "efficientnet": "models/efficientnet_model.keras",  # Your EfficientNet
    "convnext": "models/convnext_model.keras",  # Your ConvNeXt model
}
# ============================================================================
```

### Step 4: Update Model Metadata

```python
MODEL_INFO = {
    "resnet": {
        "name": "ResNet-50",
        "description": "Your custom ResNet description",
        "input_shape": (224, 224, 3),
        "version": "1.0"
    },
    # ... other models
}
```

### Step 5: Test Model Loading

Start the server and check the logs:

```bash
cd backend
uvicorn app.main:app --reload
```

You should see:
```
============================================================
LOADING TRAINED MODELS
============================================================
[resnet] Loading model from: models/resnet_model.keras
[resnet] ✓ Successfully loaded: ResNet-50
[resnet]   Input shape: (224, 224, 3)
[resnet]   Parameters: 23,564,800
...
============================================================
Model Loading Complete: 4/4 models loaded
============================================================
```

## Adding New Models

To add a completely new model architecture:

### 1. Add Model Path

```python
MODEL_PATHS = {
    # ... existing models
    "custom_model": "models/custom_model.keras",
}
```

### 2. Add Model Info

```python
MODEL_INFO = {
    # ... existing models
    "custom_model": {
        "name": "Custom CNN",
        "description": "Your custom architecture",
        "input_shape": (224, 224, 3),
        "version": "1.0"
    },
}
```

### 3. Restart Server

The new model will be automatically loaded on startup.

## Model Output Handling

The platform handles different output formats automatically:

### Single Output (Sigmoid)
```python
# Model outputs: [[0.75]]
# Interpreted as: 75% probability of Malignant
```

### Two Outputs (Softmax)
```python
# Model outputs: [[0.25, 0.75]]
# Interpreted as: 25% Benign, 75% Malignant
```

### Custom Output Handling

If your model has a different output format, modify `app/core/model_loader.py`:

```python
def predict(self, model_id: str, preprocessed_image: np.ndarray):
    model = self.get_model(model_id)
    predictions = model.predict(preprocessed_image, verbose=0)
    
    # Custom interpretation logic
    # ... your code here
    
    return predicted_class, confidence
```

## Performance Optimization

### Model Quantization

Reduce model size and improve inference speed:

```python
import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
quantized_model = converter.convert()
```

### Mixed Precision

Enable mixed precision for faster inference:

```python
from tensorflow.keras import mixed_precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)
```

### Batch Inference

For multiple images, use batch prediction:

```python
# Stack multiple preprocessed images
batch = np.stack([img1, img2, img3])
predictions = model.predict(batch)
```

## Troubleshooting

### Model Loading Errors

**File not found**
```
Check that model files exist in backend/models/
Verify paths in config.py are correct
Use absolute paths if needed
```

**Incompatible format**
```
Ensure model is saved with compatible TensorFlow version
Try re-saving: model.save('new_format.keras')
Check for custom objects/layers
```

**Out of memory**
```
Reduce model size or use quantization
Enable memory growth: tf.config.experimental.set_memory_growth(gpu, True)
Use CPU inference if GPU memory is limited
```

### Prediction Errors

**Wrong output shape**
```python
# Check your model's output
print(model.output_shape)

# Adjust prediction logic in model_loader.py if needed
```

**Incorrect preprocessing**
```python
# Verify preprocessing matches training
# The platform uses ImageNet normalization by default
# Modify image_processor.py if your model needs different preprocessing
```

## Best Practices

1. **Version Control**: Include model version in filename
   ```
   resnet_v1.2.keras
   resnet_v2.0.keras
   ```

2. **Model Validation**: Test models before deployment
   ```python
   # validation script
   python scripts/validate_model.py --model models/resnet_model.keras
   ```

3. **Documentation**: Document model architecture and training details

4. **Monitoring**: Log prediction confidence and inference times

5. **Fallback**: Handle model loading failures gracefully

## Example: Complete Integration

```python
# 1. Train your model
import tensorflow as tf
from tensorflow.keras.applications import ResNet50

model = ResNet50(weights=None, classes=2, input_shape=(224, 224, 3))
# ... training code ...

# 2. Save model
model.save('backend/models/resnet_model.keras')

# 3. Update config.py
MODEL_PATHS["resnet"] = "models/resnet_model.keras"

# 4. Start server
# uvicorn app.main:app --reload

# 5. Test via API
# curl -X POST "http://localhost:8000/api/v1/predict/resnet" -F "file=@test.jpg"
```

## Support

For model integration issues:
- Check server logs for detailed error messages
- Verify TensorFlow version compatibility
- Test model loading in isolation
- Review model architecture requirements
