# Project-Exhibition-2

# NLI File Sorter: Intelligent Document Organizer & Digital Archivist 

An AI-powered desktop application designed to solve the problem of cluttered directories by automatically organizing PDF and TXT files into designated folders. It utilizes a sophisticated hybrid architecture that combines K-Nearest Neighbors (KNN) for memory-based learning and Natural Language Inference (NLI) for deep semantic understanding to sort files with high precision.

---

## 🚀 Key Features

* **Hybrid AI Classification**
  Combines the memory-based speed of KNN with the deep semantic zero-shot verification of a BART-based NLI model (`facebook/bart-large-mnli`).

* **Uncertainty-Aware Decision Making**
  Uses entropy, dynamic temperature scaling, and probability gaps to prevent overconfident misclassifications.

* **Human-in-the-Loop (HITL)**
  Automatically pauses and requests manual intervention when the AI's confidence is low, ensuring 100% sorting accuracy.

* **Continuous Active Learning**
  Every manual classification is fed back into the local KNN model, making the system smarter and more personalized with every use.

* **OCR Support**
  Integrated support for scanned PDFs and screenshots using Tesseract and Poppler.

* **Dynamic Dashboard**
  Real-time statistics on auto-sorted vs. manual files, memory size, and a live activity log presented in a clean, dark-themed Tkinter interface.

---

## 🛠️ System Architecture

The system operates on a five-stage pipeline:

1. **Extraction**
   Text is pulled from files. If a PDF is a scan, Poppler converts pages to images and Tesseract performs OCR.

2. **KNN Memory Check**
   Document embeddings (`all-MiniLM-L6-v2`) are compared against stored vectors in ChromaDB.

3. **NLI Verification**
   The model evaluates the hypothesis:

   > "This document is about *Folder Name*"

4. **Confidence Scoring**
   Uses entropy and top-2 probability gaps.

5. **Action**

   * High confidence → auto-sort
   * Low confidence → manual intervention popup

---

## 🔬 Experimentation & Testing Phase

### Experimental Configurations and Results

| Experiment | Model Description               | Accuracy | Key Observation                                 |
| ---------- | ------------------------------- | -------- | ----------------------------------------------- |
| 1          | NLI Only                        | 22.00%   | Poor performance due to lack of learning/memory |
| 2          | Flat-KNN + NLI                  | 76.00%   | Significant gain from memory-based learning     |
| 3          | Hierarchical                    | 78.47%   | Reduced inter-domain confusion                  |
| 4          | Hierarchical + DLTS             | 82.95%   | Improved confidence calibration                 |
| 5          | Final Model (Hier + DLTS + SSP) | 83.10%   | Best performance via optimized hybrid system    |

---

## 🧪 Uncertainty and Validation Tools

* **Entropy Measures**
  Calculates randomness in prediction probabilities.

* **Top-2 Probability Gap**
  Ensures a clear distinction between top predictions.

* **Dynamic Logit Temperature Scaling (DLTS)**
  Adjusts confidence based on input text length.

* **Attention-based Saliency (SSP)**
  Penalizes uncertain attention patterns.

---

## 📊 Performance Analysis

* **Confusion Matrix**
  Strong diagonal values with minor confusion between similar domains.

* **Accuracy Growth Curve**
  Accuracy improves and stabilizes (~83%) over time.

* **Source Distribution**
  Increasing reliance on KNN as memory grows, reducing NLI usage.

---

## 🧠 Technology Stack

| Component      | Technology                             |
| -------------- | -------------------------------------- |
| GUI Framework  | Python Tkinter (Custom Dark Theme)     |
| NLI Model      | facebook/bart-large-mnli               |
| Embeddings     | sentence-transformers/all-MiniLM-L6-v2 |
| Vector Search  | ChromaDB / Scikit-learn KNN            |
| OCR Engine     | Tesseract OCR                          |
| PDF Processing | pdf2image (Poppler) / PyPDF2           |

---

## 📋 Prerequisites

### 1. Python Dependencies

```bash
pip install torch transformers sentence-transformers scikit-learn chromadb pytesseract pdf2image pillow PyPDF2
```

### 2. External Tools

* **Tesseract OCR** → Install separately
* **Poppler** → Required for PDF-to-image conversion

> ⚠️ Default path:
> `C:\Program Files\poppler-25.12.0\Library\bin`
> Update `POPPLER_PATH` in `gui_integrated.py` if needed.

---

## 📖 How to Use

### 1. Setup Folders

* **Source Folder** → Place unsorted `.pdf` and `.txt` files
* **Target Folder** → Create category subfolders (e.g., Computer Science, Medical)

### 2. Run the Application

```bash
python gui_integrated.py
```

### 3. Start Sorting

* Go to **Automated Sorting tab**
* Select Source & Target folders
* Click **RUN SORTING**

### Human-in-the-Loop

If a warning popup appears:

* Review extracted text
* Select correct folder manually
* System learns instantly from your input

---

## ⚠️ Troubleshooting

* **Poppler Error**
  Ensure path matches `POPPLER_PATH` exactly.

* **Performance Issues**
  BART model is heavy → GPU (CUDA) recommended.

* **First Run Delay**
  Downloads ~1.5GB model weights (one-time).
  
--- 

dataset from: https://data.mendeley.com/datasets/9rw3vkcfy4/6
