## Kidney Stone Detection using CNN + Harris Hawks Optimization (HHO)

This project implements a deep learning system for automatic kidney stone detection from axial CT images. It compares a standard Convolutional Neural Network (CNN) with a hybrid CNN optimized using the Harris Hawks Optimization (HHO) algorithm.


Requirements

Python 3.8+
Recommended: GPU or Apple Silicon (MPS)

Installation

1. **Create a virtual environment**
```
python -m venv venv
venv\Scripts\activate
```

2. **Install dependencies**

**Option 1: Install individual packages**
```
pip install torch torchvision numpy pandas matplotlib seaborn scikit-learn
```

**Option 2: Using requirements.txt (recommended)**
To get the exact versions as specified, use:
```
pip install -r requirements.txt
```


Dataset Setup

**Dataset Source:**
Download the **Axial CT Imaging Dataset for AI-Powered Kidney Stone Detection** from:
https://www.kaggle.com/datasets/shuvokumarbasakbd/kidney-stone-axial-ct-imaging-colorized-dataset

**Expected Structure:**
After downloading, your dataset should have the following structure:
```
Dataset/
├── Non-Stone/
│   ├── image_001.jpg
│   ├── image_002.jpg
│   └── ...
└── Stone/
    ├── image_001.jpg
    ├── image_002.jpg
    └── ...
```

**Integration Steps:**

1. **Place the dataset** in your project directory or any location on your system

2. **Update the dataset path** in the following files:

   **For baseline_fixed.py** (Line 30):
   ```python
   data_dir: str = r"./path/to/your/dataset"
   ```

   **For fixed_hho.py** (Line 34):
   ```python
   data_dir: str = r"./path/to/your/dataset"
   ```

   **For fixed_hho.py** (when running with custom paths):
   ```
   python fixed_hho.py --data_dir "path/to/your/dataset"
   ```
   **For baseline_fixed.py** (Line 636):
   ```python
   data_dir: str = r"./path/to/your/dataset"
   ```
   **For baseline_fixed.py** (Line 636):
   ```python
   data_dir: str = r"./path/to/your/dataset"
   ```

3. **Verify structure**: Ensure the dataset has two folders named exactly:
   - `Stone/` (contains kidney stone images)
   - `Non-Stone/` (contains non-stone images)


How to Run

1) Train Baseline CNN
```
python baseline_cnn.py
```

• Trains for 15 epochs  
• Saves best_model.pt 
• Saves history.csv  

2) Train Hybrid CNN + HHO
```
python cnn_hho.py
```

Quick test:
```
python cnn_hho.py --quick
```

Skip optimization:
```
python cnn_hho.py --skip_hho
```

• Saves hho_cnn_presentation_best.pt  
• Saves training_results.csv

3) Generate Plots
Before running check if the directory for the dataset and the csv files are correct



# Evaluation Metrics

- Accuracy  
- Precision  
- Recall  
- F1‑Score  
- Confusion Matrix  
- Training / Validation Curves  
- Comparison charts between both models for the metrics


# What HHO Optimizes

- Learning rate  
- Weight decay  
- Dropout  
- Data augmentation strength  
- Batch size  
- Channel scaling  
- Optional network depth  


##Notes

Hybrid training may take several hours or days depending on hardware.
Use --quick for fast testing.
