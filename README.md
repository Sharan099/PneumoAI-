# Pneumonia Classification using Deep Learning

A comprehensive deep learning system for classifying chest X-ray images as **Pneumonia** or **Normal** using transfer learning with ResNet18, Grad-CAM explainability, and a RAG-based clinical decision support chatbot.

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Pneumonia Classification System            │
└─────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        ▼                     ▼                     ▼
┌───────────────┐    ┌───────────────┐    ┌───────────────┐
│   Training    │    │ Explainability│    │ RAG Chatbot   │
│   (train.py)  │    │(explainability│    │(rag_chatbot.py│
│               │    │     .py)      │    │      )        │
└───────────────┘    └───────────────┘    └───────────────┘
        │                     │                     │
        ▼                     ▼                     ▼
┌───────────────┐    ┌───────────────┐    ┌───────────────┐
│  ResNet18     │    │   Grad-CAM    │    │  FAISS Vector │
│  (Frozen      │    │   Heatmaps    │    │  Store + LLM   │
│  Backbone)    │    │               │    │               │
└───────────────┘    └───────────────┘    └───────────────┘
        │                     │                     │
        └─────────────────────┼─────────────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │  FastAPI Server  │
                    │   (/predict)     │
                    └──────────────────┘
```

## 📋 Features

- ✅ **Transfer Learning**: ResNet18 with frozen backbone for fast training
- ✅ **Image Preprocessing**: Standard ImageNet normalization with data augmentation
- ✅ **Grad-CAM Explainability**: Visual heatmaps showing important regions
- ✅ **RAG Chatbot**: Clinical decision support using WHO guidelines
- ✅ **FastAPI Endpoint**: RESTful API for predictions
- ✅ **Docker Support**: Containerized deployment

## 📁 Project Structure

```
Pneumonia_classification/
├── train.py                 # Training script
├── explainability.py        # Grad-CAM explainability
├── rag_chatbot.py          # RAG chatbot for clinical explanations
├── app.py                   # FastAPI application
├── requirements.txt         # Python dependencies
├── Dockerfile              # Docker configuration
├── README.md               # This file
├── model.pth               # Trained model (generated after training)
├── chest-xray-pneumonia/   # Dataset directory
│   └── chest_xray/
│       └── chest_xray/
│           ├── train/
│           │   ├── NORMAL/
│           │   └── PNEUMONIA/
│           ├── val/
│           │   ├── NORMAL/
│           │   └── PNEUMONIA/
│           └── test/
│               ├── NORMAL/
│               └── PNEUMONIA/
├── results/                 # Training results (generated)
├── gradcam_results/        # Grad-CAM visualizations (generated)
├── vector_store/           # FAISS index for RAG (generated)
└── who_pdfs/              # WHO PDFs for RAG (add your PDFs here)
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Dataset

Ensure your dataset is organized as:
```
chest-xray-pneumonia/chest_xray/chest_xray/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── val/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
```

### 3. Train the Model

```bash
python train.py
```

**Training Parameters:**
- Epochs: 5-8 (default: 7)
- Batch Size: 32
- Optimizer: Adam
- Learning Rate: 1e-3
- Loss: CrossEntropy

The model will be saved as `model.pth`.

### 4. Generate Grad-CAM Explanations

```bash
python explainability.py
```

This will:
- Process test samples
- Generate Grad-CAM heatmaps
- Save visualizations to `gradcam_results/`

### 5. Set Up RAG Chatbot

1. Add WHO PDFs to `who_pdfs/` directory
2. (Optional) Set OpenAI API key:
   ```bash
   export OPENAI_API_KEY="your-api-key"
   ```

3. Run the chatbot:
   ```bash
   python rag_chatbot.py
   ```

### 6. Run FastAPI Server

```bash
python app.py
```

Or using uvicorn:
```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

**API Endpoints:**
- `GET /`: API information
- `GET /health`: Health check
- `POST /predict`: Upload image for prediction

**Example Request:**
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@test_image.jpeg"
```

## 🐳 Docker Deployment

### Build Image

```bash
docker build -t pneumonia-classifier .
```

### Run Container

```bash
docker run -p 8000:8000 pneumonia-classifier
```

### With GPU Support

```bash
docker run --gpus all -p 8000:8000 pneumonia-classifier
```

## 📊 Model Architecture

### ResNet18 with Frozen Backbone

- **Backbone**: ResNet18 (pretrained on ImageNet, frozen)
- **Classifier Head**:
  - Dropout (0.5)
  - Linear (512 → 128)
  - ReLU
  - Dropout (0.3)
  - Linear (128 → 2)

### Transfer Learning Strategy

- **Frozen Backbone**: All ResNet18 layers frozen (speed trick)
- **Trainable Parameters**: Only classifier head (~16K parameters)
- **Benefits**: Fast training, reduced overfitting, stable inference

## 🔍 Explainability

### Grad-CAM

Grad-CAM (Gradient-weighted Class Activation Mapping) highlights important regions in the image that contribute to the prediction.

**Output:**
- Original image
- Heatmap visualization
- Overlay showing highlighted regions

Saved to `gradcam_results/` directory.

## 💬 RAG Chatbot

### Workflow

1. **Extract Text from PDFs**: Processes WHO guidelines PDFs
2. **Chunk Text**: Splits into overlapping chunks (500 tokens, 50 overlap)
3. **Store in FAISS**: Creates vector embeddings using Sentence Transformers
4. **Query with Context**:
   - Model prediction
   - Confidence score
   - Grad-CAM notes
   - Retrieved relevant guidelines

### Example Prompt

```
You are a clinical decision support assistant.

The AI model predicts: PNEUMONIA (confidence: 0.91).

The highlighted regions indicate lung opacity areas.

Based on medical guidelines, explain:
1. Possible clinical interpretation
2. When further tests are recommended
3. Safety disclaimer
```

## 📈 Training Results

Results are saved to `results/` directory:
- `training_results.json`: Metrics and history
- `training_history.png`: Loss and accuracy plots
- `confusion_matrix.png`: Confusion matrix visualization

## ⚙️ Configuration

### Training Parameters

Edit `train.py` to modify:
- `EPOCHS`: Number of training epochs (default: 7)
- `BATCH_SIZE`: Batch size (default: 32)
- `LEARNING_RATE`: Learning rate (default: 1e-3)
- `IMAGE_SIZE`: Input image size (default: 224)

### RAG Configuration

Edit `rag_chatbot.py` to modify:
- `chunk_size`: Text chunk size (default: 500)
- `overlap`: Chunk overlap (default: 50)
- `embedding_model`: Sentence transformer model

## 🔒 Safety & Disclaimer

⚠️ **Important**: This AI tool is for **assistance only** and should **not replace professional medical judgment**. Always consult with qualified healthcare providers for diagnosis and treatment decisions.

## 📝 Requirements Checklist

Before applying, ensure:

- ✅ Dataset documented
- ✅ Transfer learning explained
- ✅ Grad-CAM images saved
- ✅ FastAPI /predict endpoint
- ✅ Dockerfile
- ✅ README with architecture diagram

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is for educational and research purposes.

## 🙏 Acknowledgments

- Dataset: Chest X-Ray Images (Pneumonia) from Kaggle
- Model: ResNet18 from torchvision
- Grad-CAM: Based on the original paper by Selvaraju et al.



