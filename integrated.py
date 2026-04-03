# integrated.py - General Purpose Classifier for Dynamic Folder Matching
import pickle
import math
import numpy as np
import collections
import chromadb
import torch
import time
import os
from sklearn.neighbors import KNeighborsClassifier
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer

# =========================
# DEVICE
# =========================

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# =========================
# MODELS
# =========================

tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-mnli")

nli_model = AutoModelForSequenceClassification.from_pretrained(
    "facebook/bart-large-mnli",
    output_attentions=True
).to(device)

nli_model.eval()

embedder = SentenceTransformer('all-MiniLM-L6-v2')
embedder = embedder.to(device)

# KNN model for similarity matching
knn_model = KNeighborsClassifier(n_neighbors=3, metric='cosine')

# =========================
# DATABASE
# =========================

chroma_client = chromadb.PersistentClient(path="./chroma_knn2")
collection = chroma_client.get_or_create_collection(name="archivistknn2")

# =========================
# STORAGE
# =========================

cached_embeddings = []
cached_labels = []
knn_fitted = False

# =========================
# KNN PERSISTENCE
# =========================

def load_knn_data():
    global cached_embeddings, cached_labels, knn_fitted
    if os.path.exists("knn_data.pkl"):
        try:
            with open("knn_data.pkl", "rb") as f:
                cached_embeddings, cached_labels = pickle.load(f)
            print(f"Loaded KNN data: {len(cached_labels)} samples")
            if len(set(cached_labels)) >= 2:
                knn_model.fit(np.array(cached_embeddings), cached_labels)
                knn_fitted = True
                print("KNN model restored")
        except:
            print("Could not load KNN data")

def save_knn_data():
    with open("knn_data.pkl", "wb") as f:
        pickle.dump((cached_embeddings, cached_labels), f)

load_knn_data()

# =========================
# UTILITIES
# =========================

HYPOTHESIS = "This document is about {}."

def get_entropy(probs):
    return -sum(p * math.log2(p) for p in probs if p > 0)

def get_top2_gap(probs):
    sorted_probs = np.sort(probs)[::-1]
    if len(sorted_probs) < 2:
        return 0.0
    return sorted_probs[0] - sorted_probs[1]

def dlts(logits, text):
    wc = len(text.split())
    T = max(1.0, 1.5 - (wc / 400))
    return torch.softmax(logits / T, dim=-1).cpu().numpy()[0]

def saliency_score(outputs, idx, raw_prob, gamma=0.12):
    if not hasattr(outputs, "attentions") or outputs.attentions is None:
        return raw_prob
    try:
        attn = outputs.attentions[-1][idx]
        avg = torch.mean(attn, dim=0)[0, :].detach().cpu().numpy()
        avg = avg / (np.sum(avg) + 1e-9)
        h = -np.sum(avg * np.log(avg + 1e-9))
        norm = h / np.log(len(avg))
        return raw_prob * (1 - gamma * norm)
    except Exception:
        return raw_prob

# =========================
# TRACKERS
# =========================

class KNNTracker:
    def __init__(self):
        self.history = collections.deque(maxlen=50)

    def update(self, correct):
        self.history.append(correct)

    def get_threshold(self):
        if len(self.history) < 10:
            return 0.35
        acc = sum(self.history) / len(self.history)
        if acc < 0.7:
            return min(0.55, 0.35 + (0.7 - acc) * 0.4)
        else:
            return max(0.25, 0.35 - (acc - 0.7) * 0.3)

class NLITracker:
    def __init__(self):
        self.history = collections.deque(maxlen=50)
        self.base_thresh = 0.30

    def update(self, correct):
        self.history.append(correct)

    def get_threshold(self):
        if len(self.history) < 20:
            return self.base_thresh
        acc = sum(self.history) / len(self.history)
        if acc < 0.65:
            return min(0.50, self.base_thresh + 0.2 * (0.65 - acc))
        else:
            return max(0.20, self.base_thresh - 0.1 * (acc - 0.65))

    def get_entropy_ceiling(self, n):
        base = math.log2(n)
        if len(self.history) < 20:
            return base * 1.1
        acc = sum(self.history) / len(self.history)
        if acc < 0.65:
            return base * 0.95
        else:
            return base * 1.05

knn_tracker = KNNTracker()
nli_tracker = NLITracker()

# =========================
# CORE FUNCTION - PREDICT AGAINST FOLDER NAMES
# =========================

def predict_against_folders(text, folder_names, use_hierarchy=True, use_ssp=True, use_dlts=True):
    """
    Predict which folder the document belongs to based on folder names
    Returns: (best_folder, confidence, source, needs_hitl)
    """
    global knn_fitted
    
    text = str(text)[:2000]
    
    # Get document embedding
    doc_embedding = embedder.encode(text, convert_to_numpy=True)
    
    # Try KNN first if we have enough data
    if knn_fitted and len(cached_labels) >= 10:
        try:
            # Get KNN prediction
            distances, indices = knn_model.kneighbors([doc_embedding])
            neighbor_labels = [cached_labels[i] for i in indices[0]]
            
            # Count occurrences
            from collections import Counter
            label_counts = Counter(neighbor_labels)
            most_common = label_counts.most_common(1)[0]
            
            # Check if KNN is confident
            if most_common[1] >= 2:  # At least 2 neighbors agree
                prediction = most_common[0]
                
                # Calculate confidence based on distance
                avg_distance = np.mean(distances[0])
                confidence = max(0.3, 1.0 - avg_distance)
                
                if confidence > 0.6:
                    return prediction, confidence, "KNN", False
        except Exception as e:
            print(f"KNN prediction failed: {e}")
    
    # Try NLI against folder names
    if folder_names:
        try:
            # Create hypotheses for each folder
            hypotheses = [HYPOTHESIS.format(name) for name in folder_names]
            
            # Batch process
            inputs = tokenizer(
                [text] * len(hypotheses),
                hypotheses,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )
            
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = nli_model(**inputs)
                logits = outputs.logits[:, 2].unsqueeze(0)  # Entailment logits
            
            # Apply DLTS if enabled
            if use_dlts:
                probs = dlts(logits, text)
            else:
                probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
            
            # Get best prediction
            best_idx = np.argmax(probs)
            raw_confidence = probs[best_idx]
            
            # Apply saliency if enabled
            if use_ssp:
                confidence = saliency_score(outputs, best_idx, raw_confidence)
            else:
                confidence = raw_confidence
            
            entropy = get_entropy(probs)
            gap = get_top2_gap(probs)
            
            threshold = nli_tracker.get_threshold()
            entropy_ceiling = nli_tracker.get_entropy_ceiling(len(folder_names))
            
            # Check if confident
            if confidence > threshold and entropy < entropy_ceiling and gap > 0.05:
                prediction = folder_names[best_idx]
                return prediction, confidence, "NLI", False
            
        except Exception as e:
            print(f"NLI prediction failed: {e}")
    
    # Not confident - need HITL
    return None, 0.0, "None", True

def update_model(folder_name, doc_embedding):
    """Update KNN model with new classification"""
    global knn_fitted
    
    # Add to cache
    cached_embeddings.append(doc_embedding)
    cached_labels.append(folder_name)
    
    # Retrain KNN if we have enough samples
    if len(set(cached_labels)) >= 2 and len(cached_labels) >= 5:
        try:
            knn_model.fit(np.array(cached_embeddings), cached_labels)
            knn_fitted = True
            print(f"KNN model updated with {len(cached_labels)} samples")
            
            # Save periodically
            if len(cached_labels) % 10 == 0:
                save_knn_data()
        except Exception as e:
            print(f"KNN update failed: {e}")

def process_sample(text, true_label=None, use_hierarchy=True, use_ssp=True, use_dlts=True):
    """
    Legacy function for backward compatibility
    """
    # This is kept for compatibility with existing code
    return {
        "prediction": true_label or "Unknown",
        "confidence": 0.5,
        "source": "Legacy",
        "needs_hitl": True
    }