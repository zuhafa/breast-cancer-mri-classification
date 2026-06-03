// API Service for communicating with the backend

import type { PredictionResult, ComparisonResult, ModelInfo } from '@/types';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000/api/v1';

class ApiService {
  private baseUrl: string;

  constructor(baseUrl: string = API_BASE_URL) {
    this.baseUrl = baseUrl;
  }

  async getModels(): Promise<ModelInfo[]> {
    const response = await fetch(`${this.baseUrl}/models`);
    if (!response.ok) {
      throw new Error('Failed to fetch models');
    }
    const data = await response.json();
    return data.models;
  }

  async predict(modelId: string, imageFile: File): Promise<PredictionResult> {
    const formData = new FormData();
    formData.append('file', imageFile);

    const response = await fetch(`${this.baseUrl}/predict/${modelId}`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Prediction failed');
    }

    return await response.json();
  }

  async compare(imageFile: File, modelIds?: string[]): Promise<ComparisonResult> {
    const formData = new FormData();
    formData.append('file', imageFile);
    
    if (modelIds && modelIds.length > 0) {
      formData.append('models', modelIds.join(','));
    }

    const response = await fetch(`${this.baseUrl}/compare`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Comparison failed');
    }

    return await response.json();
  }

  async analyze(imageFile: File): Promise<ComparisonResult> {
    const formData = new FormData();
    formData.append('file', imageFile);

    const response = await fetch(`${this.baseUrl}/analyze`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Analysis failed');
    }

    return await response.json();
  }

  async healthCheck(): Promise<{ status: string; models_loaded: number }> {
    const response = await fetch(`${this.baseUrl}/health`);
    if (!response.ok) {
      throw new Error('Health check failed');
    }
    return await response.json();
  }
}

// Mock API service for development/demo without backend
export class MockApiService extends ApiService {
  private mockDelay = 1500;

  async getModels(): Promise<ModelInfo[]> {
    await new Promise(resolve => setTimeout(resolve, 300));
    return [
      {
        id: 'resnet',
        name: 'ResNet-50',
        description: 'Residual Network with skip connections for deep feature extraction',
        loaded: true,
        input_shape: [224, 224, 3]
      },
      {
        id: 'densenet',
        name: 'DenseNet-121',
        description: 'Densely Connected Convolutional Network with feature reuse',
        loaded: true,
        input_shape: [224, 224, 3]
      },
      {
        id: 'efficientnet',
        name: 'EfficientNet-B0',
        description: 'Efficiently scaled network with compound scaling',
        loaded: true,
        input_shape: [224, 224, 3]
      },
      {
        id: 'convnext',
        name: 'ConvNeXt-Tiny',
        description: 'Modern CNN architecture inspired by Transformers',
        loaded: true,
        input_shape: [224, 224, 3]
      }
    ];
  }

  async predict(modelId: string): Promise<PredictionResult> {
    await new Promise(resolve => setTimeout(resolve, this.mockDelay));
    
    const modelNames: Record<string, string> = {
      resnet: 'ResNet-50',
      densenet: 'DenseNet-121',
      efficientnet: 'EfficientNet-B0',
      convnext: 'ConvNeXt-Tiny'
    };

    // Simulate prediction with random result
    const isMalignant = Math.random() > 0.5;
    const confidence = 0.7 + Math.random() * 0.25;

    return {
      model_id: modelId,
      model_name: modelNames[modelId] || modelId,
      predicted_class: isMalignant ? 'Malignant' : 'Benign',
      confidence: Math.round(confidence * 100),
      inference_time_ms: 245 + Math.random() * 100,
      success: true
    };
  }

  async compare(): Promise<ComparisonResult> {
    await new Promise(resolve => setTimeout(resolve, this.mockDelay * 1.5));

    const allModelIds = ['resnet', 'densenet', 'efficientnet', 'convnext'];
    const predictions: PredictionResult[] = [];
    
    let benignCount = 0;
    let malignantCount = 0;
    let totalConfidence = 0;

    for (const id of allModelIds) {
      const result = await this.predict(id);
      predictions.push(result);
      
      if (result.predicted_class === 'Benign') {
        benignCount++;
      } else {
        malignantCount++;
      }
      totalConfidence += result.confidence;
    }

    const consensusClass = malignantCount > benignCount ? 'Malignant' : 'Benign';
    const consensusCount = consensusClass === 'Malignant' ? malignantCount : benignCount;

    return {
      predictions,
      comparison: {
        consensus_class: consensusClass,
        consensus_confidence: Math.round((totalConfidence / predictions.length) * 100) / 100,
        average_confidence: Math.round((totalConfidence / predictions.length) * 100) / 100,
        model_agreement: Math.round((consensusCount / predictions.length) * 100),
        models_used: predictions.length,
        successful_predictions: predictions.length
      },
      total_time_ms: 1200
    };
  }

  async analyze(): Promise<ComparisonResult> {
    return this.compare();
  }

  async healthCheck(): Promise<{ status: string; models_loaded: number }> {
    await new Promise(resolve => setTimeout(resolve, 200));
    return { status: 'healthy', models_loaded: 4 };
  }
}

// Export singleton instance
export const api = new ApiService();;
