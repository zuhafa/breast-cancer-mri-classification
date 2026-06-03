import numpy as np
from PIL import Image
import keras
import click

@click.command()
@click.argument('image_path', type=click.Path(exists=True))
@click.option('--model_path', default='model.h5', help='Path to your trained Keras/HDF5 model.')
def predict(image_path: str, model_path: str) -> None:
    """Loads an MRI image and predicts the likelihood of breast cancer."""
    
    # 1. Load the model (using Keras/h5py)
    print(f"Loading model from {model_path}...")
    model = keras.saving.load_model(model_path)
    
    # 2. Load and preprocess the image (using PIL/Pillow)
    # Assuming your model expects 224x224 grayscale images
    img = Image.open(image_path).convert('L')
    img = img.resize((224, 224))
    
    # 3. Convert to numerical array and normalize (using Numpy)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0) # Add batch dimension
    img_array = np.expand_dims(img_array, axis=-1) # Add channel dimension
    
    # 4. Run the prediction
    prediction = model.predict(img_array)
    print(f"Prediction Output: {prediction[0]}")

if __name__ == '__main__':
    predict()