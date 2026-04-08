# integrated.py (KNN + GPU FINAL)
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
# CONFIG
# =========================

domain_mapping = {
    'Ece': 'Electrical Engineering',
    'Biochemistry': 'Biochemistry',
    'Medical': 'Medicine',
    'Cs': 'Computer Science',
    'Civil': 'Civil Engineering',
    'Mae': 'Mechanical Engineering',
    'Psychology': 'Psychology'
}

parent_mapping = {
    'Cs': 'Technology',
    'Ece': 'Technology',
    'Mae': 'Technology',
    'Civil': 'Technology',
    'Biochemistry': 'Life Sciences',
    'Medical': 'Life Sciences',
    'Psychology': 'Social Sciences'
}

reverse_mapping = {v: k for k, v in domain_mapping.items()}
super_labels = list(set(parent_mapping.values()))

HYPOTHESIS = "This document is strictly categorized under {}."

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

# 🔥 KNN replaces GP
knn_model = KNeighborsClassifier(n_neighbors=5, metric='cosine')

# =========================
# DATABASE
# =========================

chroma_client = chromadb.PersistentClient(path="./chroma_knn")
collection = chroma_client.get_or_create_collection(name="archivistknn")


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
        with open("knn_data.pkl", "rb") as f:
            cached_embeddings, cached_labels = pickle.load(f)

        print(f"Loaded KNN data: {len(cached_labels)} samples")

        if len(set(cached_labels)) >= 2:
            knn_model.fit(np.array(cached_embeddings), cached_labels)
            knn_fitted = True
            print("KNN model restored")

def save_knn_data():
    with open("knn_data.pkl", "wb") as f:
        pickle.dump((cached_embeddings, cached_labels), f)
load_knn_data()
# =========================
# UTILITIES
# =========================

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
    # Check if 'attentions' attribute exists AND is not None
    if not hasattr(outputs, "attentions") or outputs.attentions is None:
        return raw_prob

    try:
        # Access the attention tensor for the specific index
        attn = outputs.attentions[-1][idx]
        avg = torch.mean(attn, dim=0)[0, :].detach().cpu().numpy()
        avg = avg / (np.sum(avg) + 1e-9)

        h = -np.sum(avg * np.log(avg + 1e-9))
        norm = h / np.log(len(avg))

        return raw_prob * (1 - gamma * norm)
    except Exception:
        # Fallback if tensor shapes don't match expectations
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
        # Lower than GP because KNN is more reliable here
        if len(self.history) < 10:
            return 0.30

        acc = sum(self.history) / len(self.history)

        if acc < 0.75:
            return min(0.50, 0.30 + (0.75 - acc) * 0.4)
        else:
            return max(0.20, 0.30 - (acc - 0.75) * 0.3)

class NLITracker:
    def __init__(self):
        self.history = collections.deque(maxlen=50)
        self.base_thresh = 0.26

    def update(self, correct):
        self.history.append(correct)

    def get_threshold(self):
        if len(self.history) < 20:
            return self.base_thresh

        acc = sum(self.history) / len(self.history)

        if acc < 0.7:
            return min(0.45, self.base_thresh + 0.2 * (0.7 - acc))
        else:
            return max(0.20, self.base_thresh - 0.1 * (acc - 0.7))

    def get_entropy_ceiling(self, n):
        base = math.log2(n)
        if len(self.history) < 20:
            return base

        acc = sum(self.history) / len(self.history)

        if acc < 0.7:
            return base * 0.98
        else:
            return base * 1.05

knn_tracker = KNNTracker()
nli_tracker = NLITracker()

# =========================
# CORE FUNCTION
# =========================

def process_sample(text, true_label, use_hierarchy=True, use_ssp=True, use_dlts=True):

    global knn_fitted

    text = str(text)[:1500]

    vector = embedder.encode(text, convert_to_numpy=True)

    final_pred = None
    source = None
    confidence = 0.0
    entropy_val = 0.0

    # =========================
    # CACHE
    # =========================

    if collection.count() > 0:
        cache = collection.query(query_embeddings=[vector.tolist()], n_results=1)
        if cache['distances'] and cache['distances'][0][0] < 0.05:
            final_pred = cache['metadatas'][0][0]['label']
            source = "Cache"
            confidence = 1.0

    # =========================
    # KNN (FIXED)
    # =========================

    if not final_pred and len(cached_labels) >= 20:

        # 🔥 Ensure model is trained
        if not knn_fitted or len(cached_labels) % 50 == 0:
            if len(set(cached_labels)) >= 2:
                knn_model.fit(np.array(cached_embeddings), cached_labels)
                knn_fitted = True
        if knn_fitted:
            probs = knn_model.predict_proba([vector])[0]
            max_prob = max(probs)
            entropy_val = get_entropy(probs)
            if (
                max_prob >= 0.8 and
                entropy_val < 0.8
            ):
                final_pred = knn_model.classes_[np.argmax(probs)]
                source = "KNN"
                confidence = max_prob
    # =========================
    # NLI (GPU FIXED)
    # =========================

    if not final_pred and use_hierarchy==False:

        labels = list(domain_mapping.keys())
        verbose = [domain_mapping[l] for l in labels]

        inputs = tokenizer(
            [text] * len(verbose),
            [HYPOTHESIS.format(l) for l in verbose],
            return_tensors="pt",
            padding=True,
            truncation=True
        )

        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = nli_model(**inputs)
            logits = outputs.logits[:, 2].unsqueeze(0)

        probs = dlts(logits, text) if use_dlts else torch.softmax(logits, dim=-1).cpu().numpy()[0]

        idx = np.argmax(probs)
        raw_conf = probs[idx]

        conf = saliency_score(outputs, idx, raw_conf) if use_ssp else raw_conf
        entropy_val = get_entropy(probs)
        gap = get_top2_gap(probs)

        threshold = nli_tracker.get_threshold()
        entropy_ceiling = nli_tracker.get_entropy_ceiling(len(labels))

        if conf > threshold and entropy_val < entropy_ceiling and gap > 0.06:
            final_pred = labels[idx]
            source = "NLI"
            confidence = conf

    # =========================
    # HIERARCHICAL NLI
    # =========================

    if not final_pred and use_hierarchy:

        inputs = tokenizer(
            [text] * len(super_labels),
            [HYPOTHESIS.format(l) for l in super_labels],
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(device)

        with torch.no_grad():
            outputs = nli_model(**inputs)
            logits = outputs.logits[:, 2].unsqueeze(0)

        probs = dlts(logits, text) if use_dlts else torch.softmax(logits, dim=-1).cpu().numpy()[0]

        idx = np.argmax(probs)
        raw_conf = probs[idx]
        parent_label = super_labels[idx]

        # 🔥 REMOVED THE NESTED DEF SALIENCY_SCORE HERE 🔥
        # It now correctly uses the global function defined at the top of the script
        conf = saliency_score(outputs, idx, raw_conf) if use_ssp else raw_conf
        
        entropy_val = get_entropy(probs)
        gap = get_top2_gap(probs)

        threshold = nli_tracker.get_threshold()
        entropy_ceiling = nli_tracker.get_entropy_ceiling(len(super_labels))

        if conf > threshold and entropy_val < entropy_ceiling and gap > 0.04:

            subs = [k for k, v in parent_mapping.items() if v == parent_label]

            if len(subs) == 1:
                final_pred = subs[0]
                source = "Hier_NLI"
                confidence = conf

            else:
                sub_verbose = [domain_mapping[s] for s in subs]

                inputs = tokenizer(
                    [text] * len(sub_verbose),
                    [HYPOTHESIS.format(l) for l in sub_verbose],
                    return_tensors="pt",
                    padding=True,
                    truncation=True
                ).to(device)

                with torch.no_grad():
                    outputs = nli_model(**inputs)
                    logits = outputs.logits[:, 2].unsqueeze(0)

                probs = dlts(logits, text) if use_dlts else torch.softmax(logits, dim=-1).cpu().numpy()[0]

                idx = np.argmax(probs)
                raw_conf = probs[idx]

                conf = saliency_score(outputs, idx, raw_conf) if use_ssp else raw_conf
                entropy_val = get_entropy(probs)
                gap = get_top2_gap(probs)

                threshold = nli_tracker.get_threshold()
                entropy_ceiling = nli_tracker.get_entropy_ceiling(len(subs))

                if conf > threshold and entropy_val < entropy_ceiling and gap > 0.05:
                    final_pred = reverse_mapping[sub_verbose[idx]]
                    source = "Hier_NLI"
                    confidence = conf

    # =========================
    # HITL
    # =========================

    if not final_pred:
        final_pred = true_label
        source = "Human"
        confidence = 1.0

    is_correct = final_pred == true_label

    # =========================
    # UPDATE
    # =========================

    if source == "KNN":
        knn_tracker.update(is_correct)
    elif source == "NLI":
        nli_tracker.update(is_correct)

    cached_embeddings.append(vector)
    cached_labels.append(true_label)

    # Save every 50 samples
    if len(cached_labels) % 50 == 0:
        save_knn_data()

    collection.add(
        embeddings=[vector.tolist()],
        metadatas=[{"label": true_label}],
        ids=[f"id_{time.time()}"]
    )

    return {
        "prediction": final_pred,
        "true_label": true_label,
        "source": source,
        "confidence": confidence,
        "entropy": entropy_val,
        "is_correct": is_correct,
        "used_hierarchy": use_hierarchy,
        "used_ssp": use_ssp,
        "used_dlts": use_dlts
    }