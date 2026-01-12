"""
FastAPI Application for Pneumonia Classification
Provides /predict endpoint for inference
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import io
from pathlib import Path
import uvicorn

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Paths
BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "model.pth"
IMAGE_SIZE = 224

# Initialize FastAPI app
app = FastAPI(
    title="Pneumonia Classification API",
    description="API for classifying chest X-ray images as Pneumonia or Normal",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model variable
model = None


def load_model():
    """Load the trained model"""
    global model
    
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Please train the model first.")
    
    # Create model architecture (same as training)
    model = models.resnet18(weights=None)
    num_features = model.fc.in_features
    model.fc = torch.nn.Sequential(
        torch.nn.Dropout(0.5),
        torch.nn.Linear(num_features, 128),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.3),
        torch.nn.Linear(128, 2)
    )
    
    # Load weights
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded successfully on {device}")


@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    try:
        load_model()
        print("API started successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise


def preprocess_image(image_bytes: bytes) -> torch.Tensor:
    """Preprocess image for model input"""
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        input_tensor = transform(image).unsqueeze(0)
        return input_tensor
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Pneumonia Classification API",
        "version": "1.0.0",
        "endpoints": {
            "/predict": "POST - Upload chest X-ray image for classification",
            "/health": "GET - Health check endpoint"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device)
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Predict pneumonia from chest X-ray image
    
    Args:
        file: Uploaded image file (JPEG, PNG, etc.)
    
    Returns:
        JSON response with prediction, confidence, and class probabilities
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read image
        image_bytes = await file.read()
        
        # Preprocess
        input_tensor = preprocess_image(image_bytes)
        input_tensor = input_tensor.to(device)
        
        # Predict
        with torch.no_grad():
            output = model(input_tensor)
            probs = F.softmax(output, dim=1)
            pred_class = output.argmax(dim=1).item()
            confidence = probs[0][pred_class].item()
        
        # Get class probabilities
        normal_prob = probs[0][0].item()
        pneumonia_prob = probs[0][1].item()
        
        prediction = "PNEUMONIA" if pred_class == 1 else "NORMAL"
        
        return JSONResponse({
            "prediction": prediction,
            "confidence": round(confidence, 4),
            "probabilities": {
                "NORMAL": round(normal_prob, 4),
                "PNEUMONIA": round(pneumonia_prob, 4)
            },
            "message": f"Image classified as {prediction} with {confidence:.2%} confidence"
        })
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during prediction: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)



