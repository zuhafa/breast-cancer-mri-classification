# Quick Start Guide

## Breast Cancer MRI Classification Platform

## Project Structure

```
breast-cancer-mri-classifier/
├── backend/              # FastAPI Python backend
│   ├── app/
│   │   ├── api/         # API routes
│   │   ├── core/        # Config & model loader
│   │   ├── services/    # Business logic
│   │   └── main.py      # Entry point
│   ├── models/          # Place your .keras/.h5 files here
│   └── requirements.txt
├── frontend/            # React + TypeScript frontend
│   ├── src/            # Source code
│   └── dist/           # Built files
├── docs/               # Documentation
├── README.md           # Full documentation
└── setup.sh            # Automated setup script
```

---

## Step 1: Attach Your Models

Place your trained model files in `backend/models/`:

```bash
backend/models/
├── resnet_model.keras
├── densenet_model.keras
├── efficientnet_model.keras
└── convnext_model.keras
```

### Model Configuration

Edit `backend/app/core/config.py` to update paths:

```python
# LOAD TRAINED MODELS HERE
MODEL_PATHS = {
    "resnet": "models/resnet_model.keras",
    "densenet": "models/densenet_model.keras",
    "efficientnet": "models/efficientnet_model.keras",
    "convnext": "models/convnext_model.keras",
}
```

---

## Step 2: Start Backend

```bash
cd backend

# Create virtual environment
python -m venv venv

# Activate (Linux/Mac)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start server
uvicorn app.main:app --reload
```

Backend will be available at: **http://localhost:8000**

API Documentation: **http://localhost:8000/docs**

---

## Step 3: Start Frontend (Development)

```bash
cd frontend

# Install dependencies (if not already done)
npm install

# Start dev server
npm run dev
```

Frontend will be available at: **http://localhost:5173**

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/health` | GET | Health check |
| `/api/v1/models` | GET | List models |
| `/api/v1/predict/{model_id}` | POST | Single prediction |
| `/api/v1/compare` | POST | Compare all models |
| `/api/v1/analyze` | POST | Full analysis |

### Example Request

```bash
curl -X POST "http://localhost:8000/api/v1/predict/resnet" \
  -F "file@mri_scan.jpg"
```

---
---

## Model Requirements

- **Input**: 224x224 RGB images
- **Format**: `.keras` or `.h5`
- **Output**: Binary classification (Benign/Malignant)
- **Preprocessing**: ImageNet normalization (handled automatically)

---

## Troubleshooting

### Models not loading
- Check file paths in `config.py`
- Verify model files exist in `backend/models/`
- Check server logs for errors

### CORS errors
- Update `CORS_ORIGINS` in `config.py`
- Ensure frontend URL is allowed

### Out of memory
- Use smaller batch sizes
- Enable model quantization
- Use CPU inference

---

## Support

See `README.md` for detailed documentation.
See `docs/MODEL_INTEGRATION.md` for model integration guide.
