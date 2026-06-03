import { useState, useCallback, useEffect } from 'react';
import { 
  Upload, 
  Brain, 
  Activity, 
  Shield, 
  AlertCircle, 
  CheckCircle2, 
  XCircle,
  BarChart3,
  Microscope,
  Stethoscope,
  Info,
  RefreshCw,
  FileImage,
  Zap
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Progress } from '@/components/ui/progress';
import { Badge } from '@/components/ui/badge';
import { Separator } from '@/components/ui/separator';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { api } from '@/services/api';
import type { PredictionResult, ComparisonResult, ModelInfo, ModelType } from '@/types';
import './App.css';

function App() {
  // State
  const [uploadedImage, setUploadedImage] = useState<{ file: File; preview: string } | null>(null);
  const [selectedModel, setSelectedModel] = useState<ModelType>('resnet');
  const [isLoading, setIsLoading] = useState(false);
  const [singleResult, setSingleResult] = useState<PredictionResult | null>(null);
  const [compareResult, setCompareResult] = useState<ComparisonResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [isDragging, setIsDragging] = useState(false);
  const [activeTab, setActiveTab] = useState('single');

  // Load models on mount
  useEffect(() => {
    loadModels();
  }, []);

  const loadModels = async () => {
    try {
      const modelList = await api.getModels();
      setModels(modelList);
    } catch (err) {
      setError('Failed to load models. Please check your connection.');
    }
  };

  // Drag and drop handlers
  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
      handleFileSelect(files[0]);
    }
  }, []);

  const handleFileSelect = (file: File) => {
    if (!file.type.startsWith('image/')) {
      setError('Please upload a valid image file (JPG, PNG)');
      return;
    }
    
    if (file.size > 10 * 1024 * 1024) {
      setError('File size too large. Maximum 10MB allowed.');
      return;
    }

    const reader = new FileReader();
    reader.onload = (e) => {
      setUploadedImage({
        file,
        preview: e.target?.result as string
      });
      setError(null);
      setSingleResult(null);
      setCompareResult(null);
    };
    reader.readAsDataURL(file);
  };

  const handlePredict = async () => {
    if (!uploadedImage) {
      setError('Please upload an MRI image first');
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      if (activeTab === 'single') {
        const result = await api.predict(selectedModel);
        setSingleResult(result);
      } else {
        const result = await api.compare();
        setCompareResult(result);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Prediction failed');
    } finally {
      setIsLoading(false);
    }
  };

  const clearImage = () => {
    setUploadedImage(null);
    setSingleResult(null);
    setCompareResult(null);
    setError(null);
  };

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 90) return 'bg-emerald-500';
    if (confidence >= 70) return 'bg-blue-500';
    if (confidence >= 50) return 'bg-amber-500';
    return 'bg-red-500';
  };

  const getResultBadge = (result: string) => {
    if (result === 'Benign') {
      return (
        <Badge className="bg-emerald-100 text-emerald-700 border-emerald-200 px-4 py-2 text-lg">
          <CheckCircle2 className="w-5 h-5 mr-2" />
          Benign
        </Badge>
      );
    }
    return (
      <Badge className="bg-rose-100 text-rose-700 border-rose-200 px-4 py-2 text-lg">
        <AlertCircle className="w-5 h-5 mr-2" />
        Malignant
      </Badge>
    );
  };

  return (
    <div className="min-h-screen gradient-medical">
      {/* Header */}
      <header className="glass sticky top-0 z-50 border-b border-slate-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="w-12 h-12 rounded-xl gradient-primary flex items-center justify-center shadow-lg">
                <Brain className="w-7 h-7 text-white" />
              </div>
              <div>
                <h1 className="text-2xl font-bold text-slate-800 tracking-tight">
                  OncoScan AI
                </h1>
                <p className="text-sm text-slate-500">Breast Cancer MRI Classification</p>
              </div>
            </div>
            <div className="flex items-center space-x-4">
              <Badge variant="outline" className="bg-white/50">
                <Shield className="w-4 h-4 mr-1 text-emerald-500" />
                {models.filter(m => m.loaded).length} Models Ready
              </Badge>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Hero Section */}
        <div className="text-center mb-10 animate-fadeIn">
          <h2 className="text-4xl font-bold text-slate-800 mb-4">
            AI-Powered Breast Cancer Detection
          </h2>
          <p className="text-lg text-slate-600 max-w-2xl mx-auto">
            Upload an MRI scan and our ensemble of deep learning models will analyze it 
            to assist in breast cancer classification.
          </p>
        </div>

        {/* Model Cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-10">
          {models.map((model, index) => (
            <Card 
              key={model.id} 
              className={`card-hover border-slate-200 animate-fadeIn stagger-${index + 1} opacity-0`}
              style={{ animationFillMode: 'forwards' }}
            >
              <CardContent className="p-4">
                <div className="flex items-start space-x-3">
                  <div className={`w-10 h-10 rounded-lg flex items-center justify-center ${
                    model.loaded ? 'bg-emerald-100' : 'bg-slate-100'
                  }`}>
                    <Activity className={`w-5 h-5 ${model.loaded ? 'text-emerald-600' : 'text-slate-400'}`} />
                  </div>
                  <div className="flex-1 min-w-0">
                    <h4 className="font-semibold text-slate-800 text-sm truncate">{model.name}</h4>
                    <p className="text-xs text-slate-500 mt-1 line-clamp-2">{model.description}</p>
                    {model.loaded && (
                      <Badge variant="outline" className="mt-2 text-xs bg-emerald-50 text-emerald-600 border-emerald-200">
                        Ready
                      </Badge>
                    )}
                  </div>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>

        {/* Main Interface */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Left Column - Upload */}
          <div className="space-y-6">
            <Card className="border-slate-200 shadow-lg">
              <CardHeader>
                <CardTitle className="flex items-center text-slate-800">
                  <FileImage className="w-5 h-5 mr-2 text-medical-primary" />
                  Upload MRI Scan
                </CardTitle>
                <CardDescription>
                  Drag and drop or click to select an image
                </CardDescription>
              </CardHeader>
              <CardContent>
                {!uploadedImage ? (
                  <div
                    onDragOver={handleDragOver}
                    onDragLeave={handleDragLeave}
                    onDrop={handleDrop}
                    className={`
                      drop-zone rounded-xl p-12 text-center cursor-pointer
                      ${isDragging ? 'drag-over' : 'bg-slate-50'}
                    `}
                    onClick={() => document.getElementById('file-input')?.click()}
                  >
                    <input
                      type="file"
                      id="file-input"
                      className="hidden"
                      accept="image/*"
                      onChange={(e) => e.target.files?.[0] && handleFileSelect(e.target.files[0])}
                    />
                    <div className="w-20 h-20 mx-auto mb-4 rounded-full bg-medical-secondary flex items-center justify-center">
                      <Upload className="w-10 h-10 text-medical-primary" />
                    </div>
                    <p className="text-slate-600 font-medium mb-2">
                      Drop your MRI image here
                    </p>
                    <p className="text-sm text-slate-400">
                      or click to browse (JPG, PNG, max 10MB)
                    </p>
                  </div>
                ) : (
                  <div className="relative">
                    <img
                      src={uploadedImage.preview}
                      alt="Uploaded MRI"
                      className="w-full rounded-xl shadow-md"
                    />
                    <button
                      onClick={clearImage}
                      className="absolute top-2 right-2 w-8 h-8 bg-white rounded-full shadow-lg flex items-center justify-center hover:bg-slate-100 transition-colors"
                    >
                      <XCircle className="w-5 h-5 text-slate-500" />
                    </button>
                    <div className="absolute bottom-2 left-2 bg-black/50 text-white text-xs px-3 py-1 rounded-full">
                      {uploadedImage.file.name}
                    </div>
                  </div>
                )}
              </CardContent>
            </Card>

            {/* Configuration */}
            {uploadedImage && (
              <Card className="border-slate-200 shadow-lg animate-fadeIn">
                <CardHeader>
                  <CardTitle className="flex items-center text-slate-800">
                    <Microscope className="w-5 h-5 mr-2 text-medical-primary" />
                    Analysis Configuration
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <Tabs value={activeTab} onValueChange={setActiveTab}>
                    <TabsList className="grid w-full grid-cols-2">
                      <TabsTrigger value="single">Single Model</TabsTrigger>
                      <TabsTrigger value="compare">Compare All</TabsTrigger>
                    </TabsList>
                    
                    <TabsContent value="single" className="mt-4">
                      <label className="text-sm font-medium text-slate-700 mb-2 block">
                        Select Model
                      </label>
                      <Select value={selectedModel} onValueChange={(v) => setSelectedModel(v as ModelType)}>
                        <SelectTrigger>
                          <SelectValue placeholder="Choose a model" />
                        </SelectTrigger>
                        <SelectContent>
                          {models.filter(m => m.loaded).map(model => (
                            <SelectItem key={model.id} value={model.id}>
                              {model.name}
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    </TabsContent>
                    
                    <TabsContent value="compare" className="mt-4">
                      <div className="p-4 bg-slate-50 rounded-lg">
                        <p className="text-sm text-slate-600">
                          Run all {models.filter(m => m.loaded).length} models simultaneously and compare their predictions.
                        </p>
                      </div>
                    </TabsContent>
                  </Tabs>

                  <Button
                    onClick={handlePredict}
                    disabled={isLoading}
                    className="w-full gradient-primary hover:opacity-90 text-white"
                  >
                    {isLoading ? (
                      <>
                        <RefreshCw className="w-5 h-5 mr-2 animate-spin" />
                        Analyzing...
                      </>
                    ) : (
                      <>
                        <Zap className="w-5 h-5 mr-2" />
                        {activeTab === 'single' ? 'Run Analysis' : 'Compare All Models'}
                      </>
                    )}
                  </Button>
                </CardContent>
              </Card>
            )}
          </div>

          {/* Right Column - Results */}
          <div className="space-y-6">
            {/* Error Alert */}
            {error && (
              <Alert variant="destructive" className="animate-fadeIn">
                <AlertCircle className="w-4 h-4" />
                <AlertDescription>{error}</AlertDescription>
              </Alert>
            )}

            {/* Single Model Result */}
            {singleResult && activeTab === 'single' && (
              <Card className="border-slate-200 shadow-lg animate-fadeIn">
                <CardHeader>
                  <CardTitle className="flex items-center text-slate-800">
                    <Stethoscope className="w-5 h-5 mr-2 text-medical-primary" />
                    Analysis Result
                  </CardTitle>
                  <CardDescription>
                    Prediction from {singleResult.model_name}
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-6">
                  {/* Result Badge */}
                  <div className="flex justify-center py-4">
                    {getResultBadge(singleResult.predicted_class)}
                  </div>

                  {/* Confidence Meter */}
                  <div className="space-y-2">
                    <div className="flex justify-between text-sm">
                      <span className="text-slate-600">Confidence</span>
                      <span className="font-semibold text-slate-800">
                        {singleResult.confidence}%
                      </span>
                    </div>
                    <div className="h-4 bg-slate-100 rounded-full overflow-hidden">
                      <div
                        className={`h-full ${getConfidenceColor(singleResult.confidence)} confidence-bar rounded-full`}
                        style={{ width: `${singleResult.confidence}%` }}
                      />
                    </div>
                  </div>

                  {/* Stats */}
                  <div className="grid grid-cols-2 gap-4">
                    <div className="bg-slate-50 rounded-lg p-3 text-center">
                      <p className="text-xs text-slate-500 mb-1">Inference Time</p>
                      <p className="text-lg font-semibold text-slate-800">
                        {singleResult.inference_time_ms.toFixed(0)}ms
                      </p>
                    </div>
                    <div className="bg-slate-50 rounded-lg p-3 text-center">
                      <p className="text-xs text-slate-500 mb-1">Model</p>
                      <p className="text-lg font-semibold text-slate-800">
                        {singleResult.model_name}
                      </p>
                    </div>
                  </div>
                </CardContent>
              </Card>
            )}

            {/* Comparison Results */}
            {compareResult && activeTab === 'compare' && (
              <Card className="border-slate-200 shadow-lg animate-fadeIn">
                <CardHeader>
                  <CardTitle className="flex items-center text-slate-800">
                    <BarChart3 className="w-5 h-5 mr-2 text-medical-primary" />
                    Model Comparison
                  </CardTitle>
                  <CardDescription>
                    Consensus from {compareResult.comparison.models_used} models
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-6">
                  {/* Consensus */}
                  <div className="bg-gradient-to-r from-slate-50 to-slate-100 rounded-xl p-6 text-center">
                    <p className="text-sm text-slate-500 mb-2">Consensus Prediction</p>
                    <div className="flex justify-center mb-2">
                      {getResultBadge(compareResult.comparison.consensus_class)}
                    </div>
                    <p className="text-3xl font-bold text-slate-800">
                      {compareResult.comparison.consensus_confidence}%
                    </p>
                    <p className="text-sm text-slate-500">confidence</p>
                  </div>

                  {/* Model Agreement */}
                  <div className="space-y-2">
                    <div className="flex justify-between text-sm">
                      <span className="text-slate-600">Model Agreement</span>
                      <span className="font-semibold text-slate-800">
                        {compareResult.comparison.model_agreement}%
                      </span>
                    </div>
                    <Progress value={compareResult.comparison.model_agreement} className="h-2" />
                  </div>

                  <Separator />

                  {/* Individual Model Results */}
                  <div className="space-y-3">
                    <h4 className="font-semibold text-slate-800">Individual Predictions</h4>
                    {compareResult.predictions.map((pred, idx) => (
                      <div 
                        key={pred.model_id}
                        className="flex items-center justify-between p-3 bg-slate-50 rounded-lg animate-slideIn"
                        style={{ animationDelay: `${idx * 0.1}s` }}
                      >
                        <div className="flex items-center space-x-3">
                          <div className={`w-3 h-3 rounded-full ${
                            pred.predicted_class === 'Benign' ? 'bg-emerald-500' : 'bg-rose-500'
                          }`} />
                          <span className="font-medium text-slate-700">{pred.model_name}</span>
                        </div>
                        <div className="flex items-center space-x-4">
                          <span className={`text-sm font-semibold ${
                            pred.predicted_class === 'Benign' ? 'text-emerald-600' : 'text-rose-600'
                          }`}>
                            {pred.predicted_class}
                          </span>
                          <span className="text-sm text-slate-500 w-16 text-right">
                            {pred.confidence}%
                          </span>
                        </div>
                      </div>
                    ))}
                  </div>

                  {/* Total Time */}
                  <div className="text-center text-sm text-slate-500">
                    Total analysis time: {compareResult.total_time_ms.toFixed(0)}ms
                  </div>
                </CardContent>
              </Card>
            )}

            {/* Empty State */}
            {!singleResult && !compareResult && !isLoading && (
              <Card className="border-slate-200 border-dashed">
                <CardContent className="p-12 text-center">
                  <div className="w-16 h-16 mx-auto mb-4 rounded-full bg-slate-100 flex items-center justify-center">
                    <Activity className="w-8 h-8 text-slate-300" />
                  </div>
                  <p className="text-slate-500">Upload an image and run analysis to see results</p>
                </CardContent>
              </Card>
            )}

            {/* Loading State */}
            {isLoading && (
              <Card className="border-slate-200">
                <CardContent className="p-12 text-center">
                  <div className="w-16 h-16 mx-auto mb-4 rounded-full bg-medical-secondary flex items-center justify-center animate-pulse">
                    <RefreshCw className="w-8 h-8 text-medical-primary animate-spin" />
                  </div>
                  <p className="text-slate-600 font-medium">Analyzing MRI scan...</p>
                  <p className="text-sm text-slate-400 mt-1">This may take a few seconds</p>
                </CardContent>
              </Card>
            )}
          </div>
        </div>

        {/* Disclaimer */}
        <Alert className="mt-10 bg-amber-50 border-amber-200">
          <Info className="w-5 h-5 text-amber-600" />
          <AlertDescription className="text-amber-800">
            <strong>Important Disclaimer:</strong> This AI-powered tool is designed for research 
            and educational purposes only. It is not intended to provide medical diagnosis. 
            Always consult with qualified healthcare professionals for medical advice and diagnosis.
          </AlertDescription>
        </Alert>
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-200 mt-16 py-8 bg-white/50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex flex-col md:flex-row justify-between items-center">
            <div className="flex items-center space-x-2 mb-4 md:mb-0">
              <Brain className="w-5 h-5 text-medical-primary" />
              <span className="font-semibold text-slate-700">OncoScan AI</span>
            </div>
            <div className="flex items-center space-x-6 text-sm text-slate-500">
              <span>Powered by TensorFlow & FastAPI</span>
              <Separator orientation="vertical" className="h-4" />
              <span>v1.0.0</span>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}

export default App;
