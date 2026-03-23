# Classification Performance and Active Learning Optimization

## 1. Existing Literature Summary
The following table summarizes the performance of various machine learning and deep learning models on document classification tasks, highlighting the datasets used and the accuracies achieved.

| Paper / Study | Model / Approach | Dataset / Class | Metric | Score |
| :--- | :--- | :--- | :--- | :--- |
| **Comparison of ML Models for Digital Dev.** | Traditional Multiclass (Logistic Regression) | Digital Development (General) | Weighted Avg F1-Score | 0.53 |
| | One-vs-Rest (OvR) SGD | Digital Dev. (Child Protection) | F1-Score | 0.86 |
| | One-vs-Rest (OvR) Logistic Regression | Digital Dev. (Digital Finance) | F1-Score | 0.80 |
| **On Dataless Hierarchical Text Classification** | Dataless + Bootstrapping (ESA) | 20 Newsgroups | Micro-F1 Score | 0.837 |
| **Documents Classification Based On Deep Learning** | Convolutional Neural Network (CNN) | 20 Newsgroups | Accuracy | 94.88% |
| | Modified TF-IDF LDA | 20 Newsgroups | Accuracy | 74.40% |
| | Traditional LDA | 20 Newsgroups | Accuracy | 60.80% |
| **Hierarchical Text Classification Using LLMs** | Traditional HPT | Web of Science | Accuracy | 0.826 |
| | Best LLM Prompt Strategy (GPT-4o-mini) | Web of Science | Accuracy | 0.713 |
| | LLM Direct Hierarchical 5-shot (GPT-4o-mini) | Amazon Product Reviews | Accuracy | 0.868 |
| | Traditional HPT | Amazon Product Reviews | Accuracy | 0.823 |

---

## 2. Baseline Performance: Zero-Shot NLI Only
Initial classification on the Web of Science (WoS) dataset using a pure Zero-Shot Natural Language Inference (NLI) approach yielded a heavily skewed baseline.

**Overall Metrics:**
* **Overall Accuracy:** 25.00%
* **Weighted F1 Score:** 0.2666

| Class | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| **Biochemistry** | 0.33 | 0.11 | 0.17 | 9 |
| **Civil** | 0.50 | 0.25 | 0.33 | 8 |
| **CS** | 0.14 | 0.67 | 0.23 | 15 |
| **ECE** | 0.00 | 0.00 | 0.00 | 9 |
| **MAE** | 0.00 | 0.00 | 0.00 | 4 |
| **Medical** | 0.92 | 0.26 | 0.40 | 43 |
| **Psychology** | 1.00 | 0.08 | 0.15 | 12 |

---

## 3. Active Learning System Performance (Including HITL)
Integrating a Gaussian Process (GP) memory system and Human-in-the-Loop (HITL) corrections dynamically shifted the system's performance. The GP learns from human corrections, resulting in a massive accuracy boost for the final dataset output.

**System Performance & Workload:**
* **Overall System Accuracy:** 92.00% *(Includes human corrections)*
* **Total Automation Rate:** 45.00%

| Class | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| **Biochemistry** | 1.00 | 0.89 | 0.94 | 9 |
| **Civil** | 1.00 | 0.75 | 0.86 | 8 |
| **CS** | 0.83 | 1.00 | 0.91 | 15 |
| **ECE** | 1.00 | 0.78 | 0.88 | 9 |
| **MAE** | 0.60 | 0.75 | 0.67 | 4 |
| **Medical** | 0.93 | 1.00 | 0.97 | 43 |
| **Psychology** | 1.00 | 0.83 | 0.91 | 12 |

---

## 4. True AI-Only Performance (NLI + GP, Excluding HITL)
By isolating the 45% of documents that the AI system (NLI + GP) processed entirely on its own without routing to a human reviewer, we can evaluate the true learning state of the model.

**AI-Only Metrics (Evaluated on 45 Automated Documents):**
* **AI Model Accuracy:** 82.22%
* **AI Model F1 Score:** 0.7785

| Class | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| **Biochemistry** | 0.00 | 0.00 | 0.00 | 1 |
| **Civil** | 0.00 | 0.00 | 0.00 | 2 |
| **CS** | 0.77 | 1.00 | 0.87 | 10 |
| **ECE** | 0.00 | 0.00 | 0.00 | 2 |
| **MAE** | 0.00 | 0.00 | 0.00 | 1 |
| **Medical** | 0.89 | 1.00 | 0.94 | 24 |
| **Psychology** | 1.00 | 0.60 | 0.75 | 5 |

---

## 5. System Diagnostics (The 45% Automation Bottleneck)
While the AI maintains an impressive 82.22% accuracy on the files it chooses to automate, the isolated AI-Only metrics reveal exactly why the automation rate is stuck at 45%:

* **The Low-Support Blindspot:** The AI-Only table shows F1-scores of 0.00 for Biochemistry, Civil, ECE, and MAE. Because these classes had very low support (only 1 or 2 files), the GP did not have enough verified data points in its memory to map their boundaries. When it encountered these files, it either guessed wrong or routed almost all of them to HITL.
* **Over-Reliance on Dominant Classes:** The 82.22% AI accuracy is currently being carried by the Medical (Support: 24) and CS (Support: 10) classes. The GP is highly confident in these areas because it has seen enough of them, but remains too conservative to automate the minority classes.

---

## 6. Actionable Steps for Optimization
To push the automation rate toward the 70-80% range while bringing those minority class AI-scores up from 0.00, implement the following:

1. **Leverage a Vector Database "Fast-Track":**
   * Use a vector database (like ChromaDB) to cache human-verified documents. 
   * When a new document arrives, check its cosine similarity against the verified embeddings. If similarity is very high (e.g., >0.90), inherit the verified label and bypass the GP/NLI/HITL sequence entirely.
2. **Implement Class-Specific HITL Thresholds:**
   * Lower the uncertainty threshold for highly stable classes (like Medical and CS) so the system auto-classifies them more aggressively.
   * Keep the threshold strict for volatile, low-support classes (like MAE and ECE) until the GP maps their embedding space.
3. **Optimize the Gaussian Process Kernel:**
   * Switch the GP's standard RBF kernel to a Matern kernel (which often performs better in high-dimensional text embedding spaces) or a composite kernel to smooth out uncertainty scores.
4. **Targeted Data Injection (Warm Start):**
   * Instead of feeding the system random batches, artificially front-load the pipeline with 10-15 human-verified examples of your weakest classes (ECE, MAE, Biochemistry) to force the GP to learn their boundaries early.