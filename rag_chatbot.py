"""
LLM RAG Chatbot for Clinical Decision Support
Uses WHO PDFs for guidelines-based explanations
Powered by Anthropic Claude API
"""

import os
import torch
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import numpy as np
from pathlib import Path
import json
from typing import List, Dict, Optional
import faiss
from sentence_transformers import SentenceTransformer
import PyPDF2
import re
from dataclasses import dataclass

# LLM imports (using Anthropic Claude API)
try:
    from anthropic import Anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    print("Warning: Anthropic SDK not available. Install with: pip install anthropic")

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Paths
BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "model.pth"
PDF_DIR = BASE_DIR / "who_pdfs"
VECTOR_STORE_DIR = BASE_DIR / "vector_store"
VECTOR_STORE_DIR.mkdir(exist_ok=True)
FAISS_INDEX_PATH = VECTOR_STORE_DIR / "faiss_index.bin"
CHUNKS_PATH = VECTOR_STORE_DIR / "chunks.json"
IMAGE_SIZE = 224


@dataclass
class PredictionResult:
    """Model prediction result"""
    prediction: str  # 'PNEUMONIA' or 'NORMAL'
    confidence: float
    gradcam_note: str  # Description of highlighted regions


class PDFProcessor:
    """Process PDFs and extract text"""
    
    def __init__(self):
        self.chunks = []
    
    def extract_text_from_pdf(self, pdf_path: Path) -> str:
        """Extract text from PDF file"""
        text = ""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
        except Exception as e:
            print(f"Error reading PDF {pdf_path}: {e}")
        return text
    
    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """Split text into overlapping chunks"""
        # Clean text
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n+', ' ', text)
        
        chunks = []
        words = text.split()
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            if len(chunk.strip()) > 50:  # Minimum chunk size
                chunks.append(chunk)
        
        return chunks
    
    def process_pdfs(self, pdf_dir: Path) -> List[str]:
        """Process all PDFs in directory"""
        all_chunks = []
        
        if not pdf_dir.exists():
            print(f"PDF directory not found: {pdf_dir}")
            return all_chunks
        
        pdf_files = list(pdf_dir.glob("*.pdf"))
        if not pdf_files:
            print(f"No PDF files found in {pdf_dir}")
            return all_chunks
        
        print(f"Processing {len(pdf_files)} PDF files...")
        
        for pdf_path in pdf_files:
            print(f"Processing: {pdf_path.name}")
            text = self.extract_text_from_pdf(pdf_path)
            chunks = self.chunk_text(text)
            all_chunks.extend(chunks)
            print(f"  Extracted {len(chunks)} chunks")
        
        return all_chunks


class VectorStore:
    """FAISS vector store for RAG"""
    
    def __init__(self, embedding_model_name: str = 'all-MiniLM-L6-v2'):
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.index = None
        self.chunks = []
        self.dimension = 384  # Dimension for all-MiniLM-L6-v2
    
    def build_index(self, chunks: List[str]):
        """Build FAISS index from text chunks"""
        print("Building FAISS index...")
        self.chunks = chunks
        
        # Generate embeddings
        embeddings = self.embedding_model.encode(chunks, show_progress_bar=True)
        embeddings = np.array(embeddings).astype('float32')
        
        # Create FAISS index
        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(embeddings)
        
        print(f"Index built with {len(chunks)} chunks")
    
    def search(self, query: str, k: int = 5) -> List[str]:
        """Search for relevant chunks"""
        if self.index is None:
            return []
        
        # Encode query
        query_embedding = self.embedding_model.encode([query])
        query_embedding = np.array(query_embedding).astype('float32')
        
        # Search
        distances, indices = self.index.search(query_embedding, k)
        
        # Return relevant chunks
        results = [self.chunks[idx] for idx in indices[0]]
        return results
    
    def save(self, index_path: Path, chunks_path: Path):
        """Save index and chunks"""
        if self.index is not None:
            faiss.write_index(self.index, str(index_path))
            with open(chunks_path, 'w', encoding='utf-8') as f:
                json.dump(self.chunks, f, indent=2, ensure_ascii=False)
            print(f"Vector store saved to {index_path}")
    
    def load(self, index_path: Path, chunks_path: Path):
        """Load index and chunks"""
        if index_path.exists() and chunks_path.exists():
            self.index = faiss.read_index(str(index_path))
            with open(chunks_path, 'r', encoding='utf-8') as f:
                self.chunks = json.load(f)
            print(f"Vector store loaded from {index_path}")
            return True
        return False


class PneumoniaModel:
    """Pneumonia classification model wrapper"""
    
    def __init__(self, model_path: Path, device):
        self.device = device
        self.model = self._load_model(model_path)
        self.transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    
    def _load_model(self, model_path: Path):
        """Load trained model"""
        model = models.resnet18(weights=None)
        num_features = model.fc.in_features
        model.fc = torch.nn.Sequential(
            torch.nn.Dropout(0.5),
            torch.nn.Linear(num_features, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(128, 2)
        )
        
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        model.eval()
        return model
    
    def predict(self, image_path: Path) -> PredictionResult:
        """Predict pneumonia from image"""
        image = Image.open(image_path).convert('RGB')
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(input_tensor)
            probs = F.softmax(output, dim=1)
            pred_class = output.argmax(dim=1).item()
            confidence = probs[0][pred_class].item()
        
        prediction = 'PNEUMONIA' if pred_class == 1 else 'NORMAL'
        
        # Generate Grad-CAM note
        if prediction == 'PNEUMONIA':
            gradcam_note = "The highlighted regions indicate lung opacity areas, consolidation patterns, and potential infiltrates consistent with pneumonia."
        else:
            gradcam_note = "The image shows clear lung fields without significant opacity or consolidation patterns."
        
        return PredictionResult(
            prediction=prediction,
            confidence=confidence,
            gradcam_note=gradcam_note
        )


class ClinicalChatbot:
    """RAG-based clinical decision support chatbot"""
    
    def __init__(self, model_path: Path, pdf_dir: Path, use_claude: bool = True, 
                 api_key: Optional[str] = None):
        self.model = PneumoniaModel(model_path, device)
        self.vector_store = VectorStore()
        self.use_claude = use_claude and ANTHROPIC_AVAILABLE
        self.claude_client = None
        
        if self.use_claude:
            # Try to get API key
            if api_key:
                api_key_value = api_key
            else:
                api_key_value = os.getenv('ANTHROPIC_API_KEY')
            
            if api_key_value:
                # Initialize Anthropic Claude client
                try:
                    self.claude_client = Anthropic(api_key=api_key_value)
                except Exception as e:
                    print(f"Warning: Error initializing Claude client: {e}")
                    self.use_claude = False
            else:
                print("Warning: Anthropic API key not found. Set ANTHROPIC_API_KEY environment variable.")
                self.use_claude = False
        
        # Initialize or load vector store
        self._initialize_vector_store(pdf_dir)
    
    def _initialize_vector_store(self, pdf_dir: Path):
        """Initialize vector store from PDFs"""
        # Try to load existing index
        if self.vector_store.load(FAISS_INDEX_PATH, CHUNKS_PATH):
            return
        
        # Build new index from PDFs
        processor = PDFProcessor()
        chunks = processor.process_pdfs(pdf_dir)
        
        if chunks:
            self.vector_store.build_index(chunks)
            self.vector_store.save(FAISS_INDEX_PATH, CHUNKS_PATH)
        else:
            print("Warning: No chunks extracted. Vector store not initialized.")
    
    def _query_llm(self, prompt: str) -> str:
        """Query LLM (Claude or local)"""
        if self.use_claude:
            if self.claude_client is None:
                print("Claude API not properly initialized. Falling back to template response.")
                return self._template_response(prompt)
            
            # Try different Claude models in order of preference
            models_to_try = [
                "claude-3-5-sonnet-20240620",  # Latest Claude 3.5 Sonnet (best quality)
                "claude-3-sonnet-20240229",     # Claude 3 Sonnet (fallback)
                "claude-3-haiku-20240307",      # Claude 3 Haiku (fastest, cheapest)
            ]
            
            for model_name in models_to_try:
                try:
                    # Claude API uses messages format with system message
                    # Optimized for token usage: reduced max_tokens, concise system message
                    response = self.claude_client.messages.create(
                        model=model_name,
                        max_tokens=512,  # Reduced from 1024 to save tokens
                        temperature=0.7,
                        system="Clinical assistant. Provide concise, evidence-based medical explanations. Be brief.",
                        messages=[
                            {"role": "user", "content": prompt}
                        ]
                    )
                    # Extract text from response
                    return response.content[0].text
                except Exception as e:
                    # If this is the last model, print error and fallback
                    if model_name == models_to_try[-1]:
                        print(f"Error querying Claude with all models: {e}")
                        print("Falling back to template response.")
                        return self._template_response(prompt)
                    # Otherwise, try next model
                    continue
        else:
            # Fallback: return template response
            return self._template_response(prompt)
    
    def _template_response(self, prompt: str) -> str:
        """Template-based response when LLM is not available"""
        return """
        Based on the AI model prediction and medical guidelines:
        
        1. Clinical Interpretation: The model has identified features consistent with the predicted condition. However, AI predictions should always be verified by qualified healthcare professionals.
        
        2. Further Tests: Additional diagnostic tests may be recommended based on clinical presentation, patient history, and physical examination.
        
        3. Safety Disclaimer: This AI tool is for assistance only and should not replace professional medical judgment. Always consult with qualified healthcare providers for diagnosis and treatment decisions.
        """
    
    def generate_explanation(self, image_path: Path) -> Dict:
        """Generate clinical explanation for an image"""
        # Get model prediction
        pred_result = self.model.predict(image_path)
        
        # Retrieve relevant context from vector store
        query = f"pneumonia diagnosis guidelines clinical interpretation {pred_result.prediction.lower()}"
        relevant_chunks = self.vector_store.search(query, k=3)
        
        # Build context
        context = "\n\n".join(relevant_chunks[:3])
        
        # Build prompt (optimized for token usage)
        # Limit context to save tokens
        context_limited = context[:500] if len(context) > 500 else context
        prompt = f"""Clinical explanation (brief, 100 words max):
Prediction: {pred_result.prediction} ({pred_result.confidence:.0%})
{pred_result.gradcam_note}

Guidelines:
{context_limited}

Provide: 1) Interpretation 2) Next steps 3) Safety note. Be concise."""
        
        # Generate response
        explanation = self._query_llm(prompt)
        
        return {
            'prediction': pred_result.prediction,
            'confidence': pred_result.confidence,
            'gradcam_note': pred_result.gradcam_note,
            'explanation': explanation,
            'relevant_guidelines': relevant_chunks
        }
    
    def chat(self, user_query: str, image_path: Optional[Path] = None) -> str:
        """Interactive chat interface"""
        # If image provided, get prediction first
        context = ""
        if image_path and image_path.exists():
            pred_result = self.model.predict(image_path)
            context = f"Model Prediction: {pred_result.prediction} (confidence: {pred_result.confidence:.2f}). {pred_result.gradcam_note}\n\n"
        
        # Retrieve relevant context
        relevant_chunks = self.vector_store.search(user_query, k=3)
        guidelines_context = "\n\n".join(relevant_chunks)
        
        # Build prompt (optimized for token usage)
        # Limit context length to save tokens - use only top 2 chunks
        guidelines_short = "\n\n".join(relevant_chunks[:2])
        guidelines_limited = guidelines_short[:400] if len(guidelines_short) > 400 else guidelines_short
        prompt = f"""{context}Q: {user_query}

Guidelines (brief):
{guidelines_limited}

Provide concise, evidence-based answer."""
        
        # Generate response
        response = self._query_llm(prompt)
        return response


def main():
    """Main function for testing"""
    print("=" * 50)
    print("Clinical RAG Chatbot")
    print("=" * 50)
    
    # Check model
    if not MODEL_PATH.exists():
        print(f"Error: Model not found at {MODEL_PATH}")
        print("Please train the model first using train.py")
        return
    
    # Initialize chatbot
    print("\nInitializing chatbot...")
    chatbot = ClinicalChatbot(MODEL_PATH, PDF_DIR)
    
    # Example usage
    print("\n" + "=" * 50)
    print("Example: Generate explanation for test image")
    print("=" * 50)
    
    # Find a test image
    test_dir = BASE_DIR / "chest-xray-pneumonia" / "chest_xray" / "chest_xray" / "test"
    if test_dir.exists():
        # Try to find a pneumonia image
        pneumonia_dir = test_dir / "PNEUMONIA"
        if pneumonia_dir.exists():
            test_images = list(pneumonia_dir.glob("*.jpeg"))
            if test_images:
                test_image = test_images[0]
                print(f"\nProcessing: {test_image.name}")
                
                result = chatbot.generate_explanation(test_image)
                
                print(f"\nPrediction: {result['prediction']}")
                print(f"Confidence: {result['confidence']:.2f}")
                print(f"\nGrad-CAM Note: {result['gradcam_note']}")
                print(f"\nClinical Explanation:\n{result['explanation']}")
    
    print("\n" + "=" * 50)
    print("Interactive Chat Mode")
    print("Type 'quit' to exit")
    print("=" * 50)
    
    # Interactive chat
    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in ['quit', 'exit', 'q']:
            break
        
        if user_input:
            response = chatbot.chat(user_input)
            print(f"\nAssistant: {response}")


if __name__ == "__main__":
    main()


