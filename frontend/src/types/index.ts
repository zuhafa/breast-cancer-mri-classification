// Type definitions for the Breast Cancer MRI Classifier

export interface ModelInfo {
  id: string;
  name: string;
  description: string;
  loaded: boolean;
  input_shape: [number, number, number];
}

export interface PredictionResult {
  model_id: string;
  model_name: string;
  predicted_class: 'Benign' | 'Malignant' | string;
  confidence: number;
  inference_time_ms: number;
  success: boolean;
  error_message?: string;
}

export interface ComparisonData {
  consensus_class: string;
  consensus_confidence: number;
  average_confidence: number;
  model_agreement: number;
  models_used: number;
  successful_predictions: number;
}

export interface ComparisonResult {
  predictions: PredictionResult[];
  comparison: ComparisonData;
  total_time_ms: number;
}

export interface UploadedImage {
  file: File;
  preview: string;
}

export type ModelType = 'resnet' | 'densenet' | 'efficientnet' | 'convnext' | 'all';

export interface AppState {
  uploadedImage: UploadedImage | null;
  selectedModel: ModelType;
  isLoading: boolean;
  result: PredictionResult | ComparisonResult | null;
  error: string | null;
}
