"""
Pneumonia Classification Training Script
Uses ResNet18 with frozen backbone for transfer learning
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
import os
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import json

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Training parameters
EPOCHS = 7  # Between 5-8 as specified
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
NUM_CLASSES = 2
IMAGE_SIZE = 224

# Data paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "chest-xray-pneumonia" / "chest_xray" / "chest_xray"
TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "val"
TEST_DIR = DATA_DIR / "test"
MODEL_SAVE_PATH = BASE_DIR / "model.pth"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)


class PneumoniaDataset(Dataset):
    """Custom dataset for pneumonia classification"""
    
    def __init__(self, data_dir, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.images = []
        self.labels = []
        
        # Load images and labels
        for class_name in ['NORMAL', 'PNEUMONIA']:
            class_dir = self.data_dir / class_name
            if not class_dir.exists():
                continue
                
            label = 0 if class_name == 'NORMAL' else 1
            
            for img_file in class_dir.glob('*.jpeg'):
                self.images.append(str(img_file))
                self.labels.append(label)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def get_transforms():
    """Define image preprocessing transforms"""
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet stats
                           std=[0.229, 0.224, 0.225])
    ])
    
    val_test_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_test_transform


def create_model(num_classes=2):
    """Create ResNet18 model with frozen backbone"""
    # Load pretrained ResNet18
    model = models.resnet18(weights='IMAGENET1K_V1')
    
    # Freeze backbone (Speed Trick)
    for param in model.parameters():
        param.requires_grad = False
    
    # Replace the final fully connected layer
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, 128),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(128, num_classes)
    )
    
    # Unfreeze only the classifier
    for param in model.fc.parameters():
        param.requires_grad = True
    
    return model


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(train_loader, desc='Training')
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        pbar.set_postfix({'loss': loss.item()})
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = accuracy_score(all_labels, all_preds)
    
    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc='Validating')
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            pbar.set_postfix({'loss': loss.item()})
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    return epoch_loss, epoch_acc, precision, recall, f1


def test_model(model, test_loader, device):
    """Test the model"""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        pbar = tqdm(test_loader, desc='Testing')
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    return acc, precision, recall, f1, cm, all_probs


def plot_training_history(history, save_path):
    """Plot training history"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    axes[0].plot(history['train_loss'], label='Train Loss', marker='o')
    axes[0].plot(history['val_loss'], label='Val Loss', marker='s')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Accuracy plot
    axes[1].plot(history['train_acc'], label='Train Acc', marker='o')
    axes[1].plot(history['val_acc'], label='Val Acc', marker='s')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_confusion_matrix(cm, save_path):
    """Plot confusion matrix"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['NORMAL', 'PNEUMONIA'],
                yticklabels=['NORMAL', 'PNEUMONIA'])
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    print("=" * 50)
    print("Pneumonia Classification Training")
    print("=" * 50)
    
    # Check data directories
    if not TRAIN_DIR.exists():
        raise FileNotFoundError(f"Training directory not found: {TRAIN_DIR}")
    if not VAL_DIR.exists():
        raise FileNotFoundError(f"Validation directory not found: {VAL_DIR}")
    if not TEST_DIR.exists():
        raise FileNotFoundError(f"Test directory not found: {TEST_DIR}")
    
    # Get transforms
    train_transform, val_test_transform = get_transforms()
    
    # Create datasets
    print("\nLoading datasets...")
    train_dataset = PneumoniaDataset(TRAIN_DIR, transform=train_transform)
    val_dataset = PneumoniaDataset(VAL_DIR, transform=val_test_transform)
    test_dataset = PneumoniaDataset(TEST_DIR, transform=val_test_transform)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, 
                             shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, 
                           shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, 
                            shuffle=False, num_workers=4, pin_memory=True)
    
    # Create model
    print("\nCreating model...")
    model = create_model(NUM_CLASSES)
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Frozen parameters: {total_params - trainable_params:,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Training history
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'val_precision': [], 'val_recall': [], 'val_f1': []
    }
    
    best_val_acc = 0.0
    patience = 3
    patience_counter = 0
    
    print("\nStarting training...")
    print(f"Epochs: {EPOCHS}, Batch Size: {BATCH_SIZE}, LR: {LEARNING_RATE}")
    print("-" * 50)
    
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc, val_precision, val_recall, val_f1 = validate(
            model, val_loader, criterion, device
        )
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_precision'].append(val_precision)
        history['val_recall'].append(val_recall)
        history['val_f1'].append(val_f1)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        print(f"Val Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1: {val_f1:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'history': history
            }, MODEL_SAVE_PATH)
            print(f"✓ Saved best model (Val Acc: {val_acc:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Early stopping check
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    # Load best model for testing
    print("\nLoading best model for testing...")
    checkpoint = torch.load(MODEL_SAVE_PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Test
    print("\nTesting model...")
    test_acc, test_precision, test_recall, test_f1, cm, test_probs = test_model(
        model, test_loader, device
    )
    
    print("\n" + "=" * 50)
    print("Test Results:")
    print("=" * 50)
    print(f"Accuracy: {test_acc:.4f}")
    print(f"Precision: {test_precision:.4f}")
    print(f"Recall: {test_recall:.4f}")
    print(f"F1-Score: {test_f1:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    
    # Save results
    results = {
        'test_accuracy': float(test_acc),
        'test_precision': float(test_precision),
        'test_recall': float(test_recall),
        'test_f1': float(test_f1),
        'confusion_matrix': cm.tolist(),
        'training_history': history,
        'best_val_accuracy': float(best_val_acc)
    }
    
    with open(RESULTS_DIR / "training_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    # Plot training history
    plot_training_history(history, RESULTS_DIR / "training_history.png")
    
    # Plot confusion matrix
    plot_confusion_matrix(cm, RESULTS_DIR / "confusion_matrix.png")
    
    print(f"\nModel saved to: {MODEL_SAVE_PATH}")
    print(f"Results saved to: {RESULTS_DIR}")
    print("\nTraining completed!")


if __name__ == "__main__":
    main()



