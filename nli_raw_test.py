import pandas as pd
import csv
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from transformers import pipeline

print("Loading NLI model for WOS evaluation...")
# device=0 uses your GPU. If you don't have one or get an error, change it to device=-1 (CPU)
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=0)

def evaluate_wos_dataset(csv_path, sample_size=100):
    """
    Loads the Web of Science metadata, runs zero-shot classification 
    on the abstracts, and outputs academic metrics.
    """
    print(f"Loading WOS dataset from {csv_path}...")
    
    # 1. Load the dataset
    # WOS metadata is sometimes saved as Excel or CSV. Adjust read_csv if needed!
    try:
        df = pd.read_excel(csv_path)
        # Or pd.read_excel(csv_path) if it's an .xlsx file
        #, encoding='latin1', engine="python",quoting=csv.QUOTE_MINIMAL
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return

    # 2. Clean and prep the data
    # We only care about the text (Abstract) and the true label (Domain)
    df = df[['Abstract', 'Domain']].dropna()
    
    # Clean up the domain strings (the README shows a typo: "majaor domain", "biochemistry" lowercase)
    df['Domain'] = df['Domain'].str.strip().str.title() 
    
    # 3. Subsample for testing
    # NLI is slow. We take a random sample to prove the code works first.
    print(f"Dataset loaded. Sampling {sample_size} random papers for this test run...")
    test_df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
    
    unique_domains = test_df['Domain'].unique().tolist()
    print(f"Testing against these {len(unique_domains)} domains: {unique_domains}\n")

    y_true = []
    y_pred = []

    # 4. The Evaluation Loop
    for index, row in test_df.iterrows():
        text = str(row['Abstract'])
        true_label = row['Domain']
        
        # We slice to 2500 chars to avoid token limits, just like our archivist
        text_chunk = text[:2500] 
        
        # NLI Prediction
        result = classifier(text_chunk, unique_domains)
        predicted_label = result['labels'][0]
        
        y_true.append(true_label)
        y_pred.append(predicted_label)
        
        if (index + 1) % 10 == 0:
            print(f"Processed {index + 1}/{sample_size} abstracts...")

    # 5. Metrics & Output
    print("\n" + "="*40)
    print(" WEB OF SCIENCE EVALUATION RESULTS")
    print("="*40)
    
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    print(f"Overall Accuracy:  {acc * 100:.2f}%")
    print(f"Weighted F1 Score: {f1:.4f}\n")
    
    print("Classification Report:")
    print(classification_report(y_true, y_pred, zero_division=0))
    
    # 6. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=unique_domains)
    plt.figure(figsize=(12, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=unique_domains, yticklabels=unique_domains)
    plt.title('Web of Science (Domain Level) - NLI Zero-Shot Confusion Matrix')
    plt.ylabel('Actual Domain (Ground Truth)')
    plt.xlabel('Predicted Domain (Archivist NLI)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('wos_evaluation_matrix.png')
    print("\nSaved confusion matrix visual as 'wos_evaluation_matrix.png'!")
    plt.show()

# --- RUN IT ---
# Change this path to point to your WOS Meta-data file!
evaluate_wos_dataset("D:/Desktop/Smart digital archivist/dump/Meta-data/Data.xlsx", sample_size=1000)