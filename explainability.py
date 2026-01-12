"""
Grad-CAM Explainability Script for Pneumonia Classification
Generates heatmaps showing important regions for predictions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, models
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
from tqdm import tqdm

# OpenCV import with fallback
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    try:
        import cv2.cv2 as cv2
        CV2_AVAILABLE = True
    except ImportError:
        CV2_AVAILABLE = False
        print("Warning: OpenCV not available. Some visualization features may be limited.")

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Paths
BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "model.pth"
TEST_DIR = BASE_DIR / "chest-xray-pneumonia" / "chest_xray" / "chest_xray" / "test"
OUTPUT_DIR = BASE_DIR / "gradcam_results"
OUTPUT_DIR.mkdir(exist_ok=True)

IMAGE_SIZE = 224
NUM_SAMPLES = 10  # Number of test samples to process


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


def create_model(num_classes=2):
    """Create ResNet18 model (same as training script)"""
    model = models.resnet18(weights=None)  # We'll load our trained weights
    
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, 128),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(128, num_classes)
    )
    
    return model


def load_model(model_path, device):
    """Load trained model"""
    model = create_model()
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    return model


def preprocess_image(image_path):
    """Preprocess image for model input"""
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    original_image = np.array(image)
    input_tensor = transform(image).unsqueeze(0)
    
    return input_tensor, original_image


def overlay_heatmap(image, heatmap, alpha=0.4):
    """Overlay heatmap on original image"""
    if not CV2_AVAILABLE:
        # Fallback: simple overlay without cv2
        heatmap_resized = np.array(Image.fromarray((heatmap * 255).astype(np.uint8)).resize((image.shape[1], image.shape[0])))
        heatmap_3d = np.stack([heatmap_resized] * 3, axis=-1)
        overlayed = (image * (1 - alpha) + heatmap_3d * alpha).astype(np.uint8)
        return overlayed, heatmap_3d
    
    # Resize heatmap to match image size
    heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    heatmap_resized = np.uint8(255 * heatmap_resized)
    
    # Apply colormap
    heatmap_colored = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Overlay
    overlayed = cv2.addWeighted(image, 1 - alpha, heatmap_colored, alpha, 0)
    
    return overlayed, heatmap_colored


def get_test_samples(test_dir, num_samples=10):
    """Get random test samples from both classes"""
    test_path = Path(test_dir)
    samples = []
    
    for class_name in ['NORMAL', 'PNEUMONIA']:
        class_dir = test_path / class_name
        if not class_dir.exists():
            continue
        
        label = 0 if class_name == 'NORMAL' else 1
        image_files = list(class_dir.glob('*.jpeg'))
        
        # Randomly select samples
        np.random.seed(42)
        selected = np.random.choice(len(image_files), 
                                   min(num_samples // 2, len(image_files)), 
                                   replace=False)
        
        for idx in selected:
            samples.append((image_files[idx], label, class_name))
    
    return samples


def visualize_gradcam(model, image_path, true_label, class_name, output_dir, device):
    """Generate and save Grad-CAM visualization"""
    # Get target layer (last convolutional layer in ResNet18)
    target_layer = model.layer4[-1].conv2
    
    # Initialize Grad-CAM
    grad_cam = GradCAM(model, target_layer)
    
    # Preprocess image
    input_tensor, original_image = preprocess_image(image_path)
    input_tensor = input_tensor.to(device)
    
    # Generate CAM
    cam, output = grad_cam.generate_cam(input_tensor)
    
    # Get prediction
    probs = F.softmax(output, dim=1)
    pred_class = output.argmax(dim=1).item()
    confidence = probs[0][pred_class].item()
    
    # Resize CAM to match original image
    if CV2_AVAILABLE:
        cam_resized = cv2.resize(cam, (original_image.shape[1], original_image.shape[0]))
    else:
        # Fallback using PIL
        cam_resized = np.array(Image.fromarray((cam * 255).astype(np.uint8)).resize((original_image.shape[1], original_image.shape[0]))) / 255.0
    
    # Overlay heatmap
    overlayed, heatmap_colored = overlay_heatmap(original_image, cam_resized)
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Original image
    axes[0].imshow(original_image)
    axes[0].set_title(f'Original Image\nTrue Label: {class_name}', fontsize=12)
    axes[0].axis('off')
    
    # Heatmap
    axes[1].imshow(heatmap_colored)
    axes[1].set_title('Grad-CAM Heatmap', fontsize=12)
    axes[1].axis('off')
    
    # Overlay
    pred_label = 'PNEUMONIA' if pred_class == 1 else 'NORMAL'
    axes[2].imshow(overlayed)
    axes[2].set_title(f'Overlay\nPrediction: {pred_label} (Confidence: {confidence:.2f})', 
                     fontsize=12)
    axes[2].axis('off')
    
    plt.tight_layout()
    
    # Save figure
    image_name = Path(image_path).stem
    output_path = output_dir / f"{class_name}_{image_name}_gradcam.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return pred_class, confidence, output_path


def main():
    print("=" * 50)
    print("Grad-CAM Explainability Analysis")
    print("=" * 50)
    
    # Check if model exists
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}. Please train the model first.")
    
    # Check test directory
    if not TEST_DIR.exists():
        raise FileNotFoundError(f"Test directory not found: {TEST_DIR}")
    
    # Load model
    print("\nLoading model...")
    model = load_model(MODEL_PATH, device)
    print("Model loaded successfully!")
    
    # Get test samples
    print(f"\nSelecting {NUM_SAMPLES} test samples...")
    test_samples = get_test_samples(TEST_DIR, NUM_SAMPLES)
    print(f"Found {len(test_samples)} samples")
    
    # Process each sample
    print("\nGenerating Grad-CAM visualizations...")
    results = []
    
    for image_path, true_label, class_name in tqdm(test_samples):
        try:
            pred_class, confidence, output_path = visualize_gradcam(
                model, image_path, true_label, class_name, OUTPUT_DIR, device
            )
            
            results.append({
                'image': str(image_path),
                'true_label': class_name,
                'predicted_label': 'PNEUMONIA' if pred_class == 1 else 'NORMAL',
                'confidence': float(confidence),
                'correct': (pred_class == true_label),
                'output_path': str(output_path)
            })
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            continue
    
    # Print summary
    print("\n" + "=" * 50)
    print("Summary:")
    print("=" * 50)
    correct = sum(1 for r in results if r['correct'])
    print(f"Processed: {len(results)} samples")
    print(f"Correct predictions: {correct}/{len(results)} ({100*correct/len(results):.1f}%)")
    print(f"\nResults saved to: {OUTPUT_DIR}")
    
    # Save results summary
    import json
    with open(OUTPUT_DIR / "gradcam_summary.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nGrad-CAM analysis completed!")


if __name__ == "__main__":
    main()

