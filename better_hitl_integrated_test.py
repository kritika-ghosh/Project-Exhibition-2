import pandas as pd
import numpy as np
import math
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import DotProduct
from transformers import pipeline
from sentence_transformers import SentenceTransformer

# --- INITIALIZATION ---
print("Loading AI Models into memory...")
# 1. NLI Model (The Base Logic)
nli_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=-1)
# 2. Vector Embedder (The Memory Creator)
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# 3. Gaussian Process (The Active Learner)
gp_model = GaussianProcessClassifier(kernel=DotProduct(), random_state=42)
gp_memory_vectors = []
gp_memory_labels = []

def calculate_entropy(probabilities):
    """Calculates Shannon's entropy for a probability distribution."""
    entropy = 0.0
    for p in probabilities:
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy

def evaluate_hitl_simulation(file_path, sample_size=100):
    print(f"\nLoading WOS dataset from {file_path}...")
    try:
        df = pd.read_excel(file_path) 
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return

    # Clean the dataset
    df = df[['Abstract', 'Domain']].dropna()
    df['Domain'] = df['Domain'].str.strip().str.title()
    test_df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
    unique_domains = test_df['Domain'].unique().tolist()
    
    print(f"Testing against domains: {unique_domains}")
    print(f"Simulating HITL pipeline on {sample_size} documents...\n")

    # Map acronyms to real words for the NLI model
    domain_mapping = {
        'Ece': 'Electrical Engineering',
        'Biochemistry': 'Biochemistry',
        'Medical': 'Medicine',
        'Cs': 'Computer Science',
        'Civil': 'Civil Engineering',
        'Mae': 'Mechanical Engineering',
        'Psychology': 'Psychology'
    }
    nli_labels_to_guess = list(domain_mapping.values())

    # --- TRACKERS ---
    y_true = []      # Ground truth
    y_pred = []      # What the system (AI or Human) chose
    y_source = []    # Tracking WHO made the choice (GP, NLI, or Human)
    
    stats = {"GP_Used": 0, "NLI_Used": 0, "Human_Interventions": 0}

    for index, row in tqdm(test_df.iterrows(), total=sample_size, desc="Processing Archivist Pipeline"):
        text = str(row['Abstract'])
        true_label = row['Domain']
        text_chunk = text[:2500] 
        
        # 1. Generate Vector
        vector = embedder.encode(text_chunk)
        
        final_prediction = None
        source_of_decision = None
        
        # Check if GP is ready
        gp_trained = len(gp_memory_labels) >= 15 and len(set(gp_memory_labels)) >= 3
        
        # --- PHASE 1: GAUSSIAN PROCESS (MEMORY) ---
        if gp_trained:
            gp_probs = gp_model.predict_proba([vector])[0]
            gp_conf = max(gp_probs)
            
            if gp_conf > 0.45:
                final_prediction = gp_model.classes_[np.argmax(gp_probs)]
                source_of_decision = "GP"
                stats["GP_Used"] += 1

        # --- PHASE 2: NLI (ZERO-SHOT LOGIC) ---
        if not final_prediction:
            nli_result = nli_classifier(text_chunk, nli_labels_to_guess)
            nli_conf = nli_result['scores'][0]
            nli_top_cat_full_word = nli_result['labels'][0]
            
            # Convert full word back to acronym
            try:
                nli_top_cat = list(domain_mapping.keys())[list(domain_mapping.values()).index(nli_top_cat_full_word)]
            except ValueError:
                nli_top_cat = "Unknown"
            
            # --- PHASE 3: HUMAN ORACLE INTERVENES ---
            if nli_conf < 0.35:
                final_prediction = true_label  # Human provides ground truth
                source_of_decision = "Human"
                stats["Human_Interventions"] += 1
            else:
                final_prediction = nli_top_cat
                source_of_decision = "NLI"
                stats["NLI_Used"] += 1

            # --- ACTIVE LEARNING UPDATE ---
            # GP learns from the human or NLI result to get smarter
            gp_memory_vectors.append(vector)
            gp_memory_labels.append(true_label)
            
            if len(gp_memory_labels) >= 15 and len(set(gp_memory_labels)) >= 3:
                gp_model.fit(gp_memory_vectors, gp_memory_labels)

        # Save all results for filtering later
        y_true.append(true_label)
        y_pred.append(final_prediction)
        y_source.append(source_of_decision)

    # --- FILTERING OUT HUMAN INTERVENTIONS FOR METRICS ---
    # We only care about how well the GP and NLI performed on their own
    y_true_ai = [t for t, s in zip(y_true, y_source) if s != "Human"]
    y_pred_ai = [p for p, s in zip(y_pred, y_source) if s != "Human"]

    # --- CALCULATE AND PRINT METRICS ---
    print("\n" + "="*60)
    print(" AI-ONLY PERFORMANCE METRICS (Excluding Human Corrections)")
    print("="*60)
    
    if len(y_true_ai) > 0:
        ai_acc = accuracy_score(y_true_ai, y_pred_ai)
        ai_f1 = f1_score(y_true_ai, y_pred_ai, average='weighted')
        
        print(f"AI Model Accuracy (NLI + GP): {ai_acc * 100:.2f}%")
        print(f"AI Model F1 Score:            {ai_f1:.4f}")
        
        print("\n--- AI-ONLY CLASSIFICATION REPORT ---")
        print(classification_report(y_true_ai, y_pred_ai, zero_division=0))
        
        # --- VISUAL CONFUSION MATRIX (AI ONLY) ---
        cm = confusion_matrix(y_true_ai, y_pred_ai, labels=unique_domains)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=unique_domains, yticklabels=unique_domains)
        plt.title('AI-Only Confusion Matrix\n(Predictions by GP & NLI Only)')
        plt.ylabel('Actual Domain (Ground Truth)')
        plt.xlabel('AI Prediction')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('ai_only_confusion_matrix.png')
        print("\nVisual confusion matrix saved as 'ai_only_confusion_matrix.png'!")
        plt.show()
    else:
        print("Warning: No AI predictions were recorded. Check thresholds.")

    # --- SYSTEM EFFICIENCY STATS ---
    print("\n--- WORKLOAD DISTRIBUTION ---")
    print(f"Files Processed by GP Memory:  {stats['GP_Used']}")
    print(f"Files Processed by Base NLI:   {stats['NLI_Used']}")
    print(f"Files Requiring Human Review:  {stats['Human_Interventions']}")
    
    automation_rate = ((sample_size - stats['Human_Interventions']) / sample_size) * 100
    print(f"Total Automation Rate:         {automation_rate:.1f}%")

# --- RUN THE SIMULATION ---
# Update this path to your actual file location
evaluate_hitl_simulation("D:/Desktop/Smart digital archivist/dump/Meta-data/Data.xlsx", sample_size=100)