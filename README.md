# An Empirical Evaluation of Deep Learning Models for Breast MRI Classification

This project was developed as a third-year B.Tech Mini Project under the Department of Computer Science and Engineering at KMEA Engineering College. The objective was to explore and compare multiple deep learning architectures for breast MRI classification and integrate them into a web-based application.

**Note:** The trained model weights and dataset are not included in this repository. Users can integrate their own compatible models using the provided configuration and integration guide.


## Application Preview

<img width="1364" height="1536" alt="localhost_5173_ (2) (1)(1)" src="https://github.com/user-attachments/assets/761cbcbc-3a0f-4538-b112-952c798dde04" />


## Features

### Deep Learning Architectures

- **ResNet50**: Residual Network with skip connections
- **DenseNet121**: Densely Connected Convolutional Network
- **EfficientNetB0**: Efficiently scaled network architecture
- **ConvNeXt-Tiny**: Modern CNN inspired by Transformers

### Key Features
- Single model prediction with confidence scores
- Multi-model comparison and analysis
- Drag-and-drop image upload interface
- Real-time inference with image preprocessing
- Responsive web interface for desktop and mobile

## Application Screenshots

### Prediction Results
![Prediction Results](screenshots/results-page.png)

## Architecture

```
breast-cancer-mri-classifier/
├── backend/                 # FastAPI Python backend
│   ├── app/
│   │   ├── api/            # API routes
│   │   ├── core/           # Core configuration & model loader
│   │   ├── services/       # Business logic
│   │   └── main.py         # Application entry point
│   ├── models/             # Model directory (weights not included)
│   └── requirements.txt
├── frontend/               # React + TypeScript frontend
│   ├── src/
│   │   ├── components/     # UI components
│   │   ├── services/       # API integration
│   │   └── App.tsx         # Main application
│   └── package.json
└── docs/                   # Documentation
```

## Quick Start

### Prerequisites
- Python 3.9+
- Node.js 18+
- TensorFlow 2.15+

### 1. Clone and Setup

```bash
git clone <repository-url>
cd breast-cancer-mri-classifier
```

### 2. Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/macOS)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Attach Your Models

Place your trained model files in the `backend/models/` directory:

```bash
backend/models/
├── resnet_model.keras
├── densenet_model.keras
├── efficientnet_model.keras
└── convnext_model.keras
```

Update model paths in `backend/app/core/config.py` if needed:

```python
MODEL_PATHS = {
    "resnet": "models/resnet_model.keras",
    "densenet": "models/densenet_model.keras",
    "efficientnet": "models/efficientnet_model.keras",
    "convnext": "models/convnext_model.keras",
}
```

### 4. Start Backend Server

```bash
# From backend directory
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at `http://localhost:8000`
- API documentation: `http://localhost:8000/docs`
- Health check: `http://localhost:8000/api/v1/health`

### 5. Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

The frontend will be available at `http://localhost:5173`


## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/` | GET | API info |
| `/api/v1/health` | GET | Health check |
| `/api/v1/models` | GET | List available models |
| `/api/v1/predict/{model_id}` | POST | Single model prediction |
| `/api/v1/compare` | POST | Compare all models |
| `/api/v1/analyze` | POST | Comprehensive analysis |

### Example API Usage

```bash
# Single prediction
curl -X POST "http://localhost:8000/api/v1/predict/resnet" \
  -F "file=@mri_scan.jpg"

# Compare all models
curl -X POST "http://localhost:8000/api/v1/compare" \
  -F "file=@mri_scan.jpg"
```

## Model Requirements

### Input Specifications
- **Format**: RGB images (JPG, PNG)
- **Size**: 224x224 pixels (automatically resized)
- **Preprocessing**: ImageNet normalization

### Output Format
```json
{
  "model_id": "resnet",
  "model_name": "ResNet50",
  "predicted_class": "Benign",
  "confidence": 94.5,
  "inference_time_ms": 245,
  "success": true
}
```

## Configuration

### Backend Configuration

Edit `backend/app/core/config.py`:

```python
# Model paths
MODEL_PATHS = {
    "resnet": "models/your_resnet_model.keras",
    # ... other models
}

# Upload settings
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# Image processing
IMAGE_SIZE = 224
```

### Frontend Configuration

Create `frontend/.env`:

```env
VITE_API_URL=http://localhost:8000/api/v1
```

## Model Integration Guide

### Supported Models

- ResNet50
- DenseNet121
- EfficientNetB0
- ConvNeXt-Tiny

The examples in this guide use ResNet50 as the reference implementation. The same integration process can be adapted for the other supported architectures.

### Step 1: Prepare Your Model

Ensure your trained model:
- Accepts input shape: `(1, 224, 224, 3)`
- Outputs: Binary classification (Benign/Malignant)
- Is saved in `.keras` or `.h5` format

### Step 2: Place Model File

```bash
cp your_model.keras backend/models/resnet_model.keras
```

### Step 3: Update Configuration (if needed)

```python
# backend/app/core/config.py
MODEL_PATHS["resnet"] = "models/resnet_model.keras"
MODEL_INFO["resnet"] = {
    "name": "Your Model Name",
    "description": "Your model description",
    "input_shape": (224, 224, 3),
    "version": "1.0"
}
```

### Step 4: Restart Server

The model will be automatically loaded on startup.


### Common Issues

**Models not loading**
```
Check model file paths in config.py
Verify model files exist in backend/models/
Check TensorFlow version compatibility
```

**CORS errors**
```
Update CORS_ORIGINS in config.py
Ensure frontend URL is allowed
```

**Out of memory**
```
Reduce batch size
Use model quantization
Enable mixed precision

```

## Team Members and Contributions

This project was developed as a collaborative third-year B.Tech Mini Project under the Department of Computer Science and Engineering at KMEA Engineering College.

### Zuha Fathima

- Trained and evaluated the ResNet50 model.
- Developed and customized the React frontend.
- Implemented FastAPI backend services.
- Integrated deep learning models into the application.
- Developed image preprocessing and prediction workflows.
- Conducted application testing and debugging.

### Adhila Mansoor

- Trained and evaluated the DenseNet121 model.

### Shivapriya Sreejith

- Trained and evaluated the EfficientNetB0 model.

### Elsi C Pate

- Trained and evaluated the ConvNeXt-Tiny model.

### Collaborative Activities

The following activities were completed collaboratively by all team members:

- Dataset selection and preparation
- Literature review
- Model evaluation and comparison
- Project planning and review
- Abstract preparation
- Technical report writing
- Documentation
- PPT and presentation preparation
- Result analysis and discussion
- Mini project submission and review

## License

This project is for research and educational purposes. See LICENSE for details.

## Disclaimer

**Important**: This AI-powered tool is designed for research and educational purposes only. It is not intended to provide medical diagnosis. Always consult with qualified healthcare professionals for medical advice and diagnosis.


## Acknowledgments

- TensorFlow Team
- FastAPI Framework
- React Community
- Medical imaging researchers
