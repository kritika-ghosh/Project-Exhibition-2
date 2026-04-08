import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, classification_report, confusion_matrix

def perform_analysis(file_path):
    # Load data
    df = pd.read_csv(file_path)
    
    # --- ADDED FILTERING HERE ---
    # Remove rows where source is "cache"
    df = df[df['source'] != 'Cache'].copy()
    
    if df.empty:
        print("Warning: No data remaining after filtering out 'cache' sources.")
        return
    # ----------------------------

    # 1. Basic Metrics
    accuracy = accuracy_score(df['true_label'], df['prediction'])
    precision = precision_score(df['true_label'], df['prediction'], average='weighted')
    report = classification_report(df['true_label'], df['prediction'])
    
    print(f"Overall Accuracy (Excluding Cache): {accuracy:.4f}")
    print(f"Weighted Precision (Excluding Cache): {precision:.4f}")
    print("\nClassification Report:\n", report)
    
    # 2. Confusion Matrix Plot
    labels = sorted(df['true_label'].unique())
    cm = confusion_matrix(df['true_label'], df['prediction'], labels=labels)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap='Blues')
    plt.title('Confusion Matrix (Excluding Cache)')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig('confusion_matrix.png')
    
    # 3. Growth of Accuracy (Cumulative)
    df = df.sort_values('row_index').reset_index(drop=True)
    df['cumulative_correct'] = df['is_correct'].cumsum()
    df['cumulative_accuracy'] = df['cumulative_correct'] / (df.index + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(df['row_index'], df['cumulative_accuracy'], color='tab:blue')
    plt.title('Growth of Accuracy Over Records (Excluding Cache)')
    plt.xlabel('Number of Records (row_index)')
    plt.ylabel('Cumulative Accuracy')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig('accuracy_growth.png')
    
    # 4. Distribution of Sources Through Time
    df['bin'] = pd.cut(df['row_index'], bins=20)
    source_dist = df.groupby(['bin', 'source'], observed=False).size().unstack().fillna(0)
    source_dist_pct = source_dist.div(source_dist.sum(axis=1), axis=0)
    
    source_dist_pct.plot(kind='bar', stacked=True, figsize=(12, 6))
    plt.title('Distribution of Sources Through Time (Excluding Cache)')
    plt.xlabel('Record Range (row_index bins)')
    plt.ylabel('Proportion of Sources')
    plt.legend(title='Source', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('source_distribution_time.png')

if __name__ == "__main__":
    perform_analysis('results_knn.csv')