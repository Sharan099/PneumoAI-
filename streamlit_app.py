"""
Streamlit Web Application for Pneumonia Classification
Features:
- X-ray image upload and prediction
- Grad-CAM visualization
- RAG chatbot for clinical explanations
"""

import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import numpy as np
from pathlib import Path
import io
import os
from typing import Optional
import tempfile

# OpenCV import with fallback for headless environments
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    try:
        # Try headless version
        import cv2.cv2 as cv2
        CV2_AVAILABLE = True
    except ImportError:
        CV2_AVAILABLE = False
        st.warning("OpenCV not available. Grad-CAM visualization will be limited.")

# Import from our modules
import sys
sys.path.append(str(Path(__file__).parent))

from rag_chatbot import ClinicalChatbot, PredictionResult
import torch.nn as nn

# Grad-CAM class (inline to avoid import issues)
class GradCAM:
    """Grad-CAM implementation for ResNet18"""
    
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        self.activations = output
    
    def save_gradient(self, module, grad_input, grad_output):
        if grad_output[0] is not None:
            self.gradients = grad_output[0]
    
    def generate_cam(self, input_image, class_idx=None):
        """Generate Class Activation Map"""
        self.model.eval()
        
        # Ensure input requires gradient
        input_image = input_image.requires_grad_(True)
        
        # Forward pass
        output = self.model(input_image)
        
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        
        # Backward pass
        self.model.zero_grad()
        score = output[0, class_idx]
        score.backward()
        
        # Check if gradients were captured
        if self.gradients is None:
            raise RuntimeError("Gradients not captured. Check hook registration.")
        
        # Get gradients and activations
        gradients = self.gradients[0].cpu().data.numpy()
        activations = self.activations[0].cpu().data.numpy()
        
        # Calculate weights (global average pooling of gradients)
        weights = np.mean(gradients, axis=(1, 2))
        
        # Generate CAM
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]
        
        # Apply ReLU
        cam = np.maximum(cam, 0)
        
        # Normalize
        if cam.max() > 0:
            cam = cam / (cam.max() + 1e-8)
        
        return cam, output

# Set page config
st.set_page_config(
    page_title="Pneumonia Classification System",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .pneumonia {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
    }
    .normal {
        background-color: #e8f5e9;
        border-left: 5px solid #4caf50;
    }
    .stButton>button {
        width: 100%;
    }
    </style>
""", unsafe_allow_html=True)

# Paths - Handle both local and Streamlit Cloud environments
BASE_DIR = Path(__file__).parent.absolute()
MODEL_PATH = BASE_DIR / "model.pth"
PDF_DIR = BASE_DIR / "who_pdfs"
IMAGE_SIZE = 224

# Helper function to find model.pth in multiple locations
def find_model_path():
    """Find model.pth in common locations (for Streamlit Cloud compatibility)"""
    possible_paths = [
        BASE_DIR / "model.pth",           # Same directory as script
        Path("model.pth"),                # Current working directory
        Path.cwd() / "model.pth",         # Explicit current directory
        Path(__file__).parent / "model.pth",  # Script parent directory
    ]
    
    # Also check parent directories (in case script is in subdirectory)
    for depth in range(3):
        parent_path = Path(__file__).parent
        for _ in range(depth):
            parent_path = parent_path.parent
        possible_paths.append(parent_path / "model.pth")
    
    for path in possible_paths:
        if path.exists() and path.is_file():
            return path
    
    return None

# Debug: Print paths for troubleshooting
import os
if os.getenv("STREAMLIT_DEBUG", "false").lower() == "true":
    print(f"BASE_DIR: {BASE_DIR}")
    print(f"MODEL_PATH: {MODEL_PATH}")
    print(f"MODEL_PATH exists: {MODEL_PATH.exists()}")
    print(f"MODEL_PATH absolute: {MODEL_PATH.absolute()}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Files in BASE_DIR: {list(BASE_DIR.glob('*.pth'))}")

# Initialize session state
if 'chatbot' not in st.session_state:
    st.session_state.chatbot = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'current_prediction' not in st.session_state:
    st.session_state.current_prediction = None
if 'current_image' not in st.session_state:
    st.session_state.current_image = None


@st.cache_resource
def load_model():
    """Load the trained pneumonia model"""
    if not MODEL_PATH.exists():
        st.error(f"Model not found at {MODEL_PATH}. Please train the model first using train.py")
        return None
    
    model = models.resnet18(weights=None)
    num_features = model.fc.in_features
    model.fc = torch.nn.Sequential(
        torch.nn.Dropout(0.5),
        torch.nn.Linear(num_features, 128),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.3),
        torch.nn.Linear(128, 2)
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    return model, device


@st.cache_resource
def initialize_chatbot():
    """Initialize the RAG chatbot"""
    try:
        # Find model using helper function
        model_path = find_model_path()
        
        if model_path is None:
            st.error("Model not found. Cannot initialize chatbot.")
            return None
        
        chatbot = ClinicalChatbot(model_path, PDF_DIR, use_claude=True)
        return chatbot
    except FileNotFoundError as e:
        st.error(f"Model file not found: {e}")
        return None
    except Exception as e:
        st.warning(f"Chatbot initialization warning: {e}")
        return None


def preprocess_image(image: Image.Image) -> torch.Tensor:
    """Preprocess image for model input"""
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)


def predict_pneumonia(model, image_tensor, device):
    """Predict pneumonia from image"""
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        output = model(image_tensor)
        probs = F.softmax(output, dim=1)
        pred_class = output.argmax(dim=1).item()
        confidence = probs[0][pred_class].item()
    
    prediction = "PNEUMONIA" if pred_class == 1 else "NORMAL"
    return prediction, confidence, probs[0].cpu().numpy()


def generate_gradcam(model, image_tensor, device, target_layer=None):
    """Generate Grad-CAM visualization"""
    if target_layer is None:
        # Get the last convolutional layer
        target_layer = model.layer4[-1].conv2
    
    # Initialize Grad-CAM
    grad_cam = GradCAM(model, target_layer)
    
    # Generate CAM
    cam, output = grad_cam.generate_cam(image_tensor)
    
    return cam, output


def overlay_heatmap(image, heatmap, alpha=0.4):
    """Overlay heatmap on original image"""
    if not CV2_AVAILABLE:
        # Fallback: simple overlay without cv2
        if isinstance(image, Image.Image):
            image = np.array(image)
        # Simple heatmap overlay using numpy
        heatmap_resized = np.array(Image.fromarray((heatmap * 255).astype(np.uint8)).resize((image.shape[1], image.shape[0])))
        heatmap_3d = np.stack([heatmap_resized] * 3, axis=-1)
        overlayed = (image * (1 - alpha) + heatmap_3d * alpha).astype(np.uint8)
        return overlayed, heatmap_3d
    
    # Convert PIL to numpy
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Resize heatmap to match image size
    heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    heatmap_resized = np.uint8(255 * heatmap_resized)
    
    # Apply colormap
    heatmap_colored = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Overlay
    overlayed = cv2.addWeighted(image, 1 - alpha, heatmap_colored, alpha, 0)
    
    return overlayed, heatmap_colored


def get_concise_explanation(chatbot, pred_result, image_path=None):
    """Get concise explanation using chatbot (optimized for token usage)"""
    if chatbot is None:
        return pred_result.gradcam_note
    
    # Create concise prompt to minimize tokens (optimized for Claude)
    prompt = f"""Brief summary (50 words max):
Pred: {pred_result.prediction} ({pred_result.confidence:.0%}). {pred_result.gradcam_note}
Provide: 1) Meaning 2) Next steps 3) Safety note."""
    
    try:
        explanation = chatbot._query_llm(prompt)
        return explanation
    except:
        return pred_result.gradcam_note


def main():
    # Header
    st.markdown('<h1 class="main-header">🫁 Pneumonia Classification System</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        st.info("Upload an X-ray image to get started")
        
        # Model status
        model_path = find_model_path()
        if model_path:
            st.success(f"✓ Model found at: `{model_path.name}`")
        else:
            st.error("✗ Model not found")
            with st.expander("🔍 Debug Info"):
                st.code(f"""
Script location: {Path(__file__).parent.absolute()}
Current directory: {os.getcwd()}
Base directory: {BASE_DIR}

Searched locations:
- {BASE_DIR / 'model.pth'} (exists: {(BASE_DIR / 'model.pth').exists()})
- {Path('model.pth').absolute()} (exists: {Path('model.pth').exists()})
- {Path.cwd() / 'model.pth'} (exists: {(Path.cwd() / 'model.pth').exists()})
                """)
        
        # Chatbot status
        if st.session_state.chatbot is None:
            with st.spinner("Initializing chatbot..."):
                st.session_state.chatbot = initialize_chatbot()
        
        if st.session_state.chatbot:
            st.success("✓ Chatbot ready")
        else:
            st.warning("⚠ Chatbot unavailable")
        
        st.markdown("---")
        st.markdown("### 📖 About")
        st.markdown("""
        This system uses:
        - **ResNet18** for classification
        - **Grad-CAM** for explainability
        - **Claude AI** for clinical insights
        - **RAG** from WHO guidelines
        """)
    
    # Main content
    tab1, tab2 = st.tabs(["🔍 Image Analysis", "💬 Chat Assistant"])
    
    with tab1:
        st.header("Upload X-Ray Image")
        
        uploaded_file = st.file_uploader(
            "Choose an X-ray image...",
            type=['jpg', 'jpeg', 'png'],
            help="Upload a chest X-ray image in JPEG or PNG format"
        )
        
        if uploaded_file is not None:
            # Load and display image
            image = Image.open(uploaded_file).convert('RGB')
            st.session_state.current_image = image
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Original Image")
                st.image(image, use_container_width=True)
            
            # Load model
            model_data = load_model()
            if model_data is None:
                st.stop()
            
            model, device = model_data
            
            # Preprocess and predict
            with st.spinner("Analyzing image..."):
                image_tensor = preprocess_image(image)
                prediction, confidence, probs = predict_pneumonia(model, image_tensor, device)
                
                # Store prediction
                pred_result = PredictionResult(
                    prediction=prediction,
                    confidence=confidence,
                    gradcam_note="The highlighted regions indicate important areas for the diagnosis."
                )
                st.session_state.current_prediction = pred_result
                
                # Generate Grad-CAM
                try:
                    cam, _ = generate_gradcam(model, image_tensor, device)
                    overlayed, heatmap_colored = overlay_heatmap(image, cam)
                    
                    with col2:
                        st.subheader("Grad-CAM Visualization")
                        st.image(overlayed, use_container_width=True, caption="Heatmap overlay showing important regions")
                except Exception as e:
                    st.warning(f"Could not generate Grad-CAM: {e}")
                    overlayed = None
            
            # Display prediction results
            st.markdown("---")
            st.header("📊 Prediction Results")
            
            # Prediction box
            pred_class = "pneumonia" if prediction == "PNEUMONIA" else "normal"
            st.markdown(f'<div class="prediction-box {pred_class}">', unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Prediction", prediction)
            with col2:
                st.metric("Confidence", f"{confidence:.2%}")
            with col3:
                normal_prob = probs[0]
                pneumonia_prob = probs[1]
                st.metric("Normal Probability", f"{normal_prob:.2%}")
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Probabilities bar chart
            st.subheader("Probability Distribution")
            prob_data = {
                "NORMAL": normal_prob,
                "PNEUMONIA": pneumonia_prob
            }
            st.bar_chart(prob_data)
            
            # Clinical explanation
            st.markdown("---")
            st.header("🔬 Clinical Explanation")
            
            if st.session_state.chatbot:
                with st.spinner("Generating clinical explanation..."):
                    explanation = get_concise_explanation(
                        st.session_state.chatbot, 
                        pred_result,
                        uploaded_file
                    )
                    st.info(explanation)
            else:
                st.info(pred_result.gradcam_note)
                st.warning("Chatbot unavailable. Full explanation requires chatbot initialization.")
    
    with tab2:
        st.header("💬 Clinical Chat Assistant")
        st.markdown("Ask questions about pneumonia diagnosis, X-ray results, or medical guidelines.")
        
        # Display chat history
        if st.session_state.chat_history:
            st.subheader("Chat History")
            for i, (role, message) in enumerate(st.session_state.chat_history):
                if role == "user":
                    with st.chat_message("user"):
                        st.write(message)
                else:
                    with st.chat_message("assistant"):
                        st.write(message)
        
        # Chat input
        user_query = st.chat_input("Ask a question about pneumonia, X-ray results, or guidelines...")
        
        if user_query:
            # Add user message to history
            st.session_state.chat_history.append(("user", user_query))
            
            # Display user message
            with st.chat_message("user"):
                st.write(user_query)
            
            # Get response
            if st.session_state.chatbot:
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        # Include current prediction context if available (optimized for tokens)
                        context_prompt = user_query
                        if st.session_state.current_prediction:
                            # Concise context to save tokens
                            context = f"X-ray: {st.session_state.current_prediction.prediction} ({st.session_state.current_prediction.confidence:.0%}). "
                            context_prompt = context + user_query
                        
                        # Get response (optimized for token usage - Claude will summarize)
                        response = st.session_state.chatbot.chat(context_prompt)
                        st.write(response)
                        st.session_state.chat_history.append(("assistant", response))
            else:
                with st.chat_message("assistant"):
                    st.error("Chatbot is not available. Please check your Anthropic API key.")
                    st.session_state.chat_history.append(("assistant", "Chatbot unavailable."))
        
        # Quick action buttons
        st.markdown("---")
        st.subheader("Quick Actions")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📋 Explain Current Result"):
                if st.session_state.current_prediction:
                    query = f"Explain the X-ray result: {st.session_state.current_prediction.prediction} with {st.session_state.current_prediction.confidence:.2f} confidence"
                    st.session_state.chat_history.append(("user", query))
                    if st.session_state.chatbot:
                        response = st.session_state.chatbot.chat(query)
                        st.session_state.chat_history.append(("assistant", response))
                        st.rerun()
        
        with col2:
            if st.button("❓ What is Pneumonia?"):
                query = "What is pneumonia? Explain briefly."
                st.session_state.chat_history.append(("user", query))
                if st.session_state.chatbot:
                    response = st.session_state.chatbot.chat(query)
                    st.session_state.chat_history.append(("assistant", response))
                    st.rerun()
        
        with col3:
            if st.button("🔄 Clear Chat"):
                st.session_state.chat_history = []
                st.rerun()
        
        # Show current prediction context
        if st.session_state.current_prediction:
            with st.expander("📊 Current X-ray Context"):
                st.write(f"**Prediction:** {st.session_state.current_prediction.prediction}")
                st.write(f"**Confidence:** {st.session_state.current_prediction.confidence:.2%}")
                st.write(f"**Note:** {st.session_state.current_prediction.gradcam_note}")


if __name__ == "__main__":
    main()

