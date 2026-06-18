# 🔐 Deep Learning Credit Card Fraud Detection System

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3+-green.svg)](https://scikit-learn.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen.svg)]()

> A production-grade deep learning project for credit card fraud detection — built systematically from first principles using PyTorch, with rigorous experimentation, professional documentation, and comprehensive learning materials.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage Guide](#usage-guide)
- [Project Structure](#project-structure)
- [Dataset Information](#dataset-information)
- [Key Results](#key-results)
- [Learning Roadmap](#learning-roadmap)
- [Documentation](#documentation)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

Credit card fraud costs the global financial industry **$32 billion+ annually**. This project builds a production-quality fraud detection system while systematically teaching deep learning fundamentals.

### 🎯 Dual Objective
1. **Production Quality:** Build a real fraud detection system with rigorous experimentation and validation
2. **Educational Depth:** Understand every concept — neurons, backpropagation, optimizers, regularization — not just use APIs

### ✨ Key Features
- 📚 **11 comprehensive learning notebooks** covering EDA through advanced modeling
- 🔬 **4 dedicated study notebooks**: Activation Functions, Optimizers, Weight Initialization, Regularization Techniques
- ⚖️ **4 class imbalance strategies** compared head-to-head (Baseline, Weighted BCE, SMOTE, Combined)
- 📊 **Complete experiment tracking** with metrics logged for every study
- 💼 **Business-aware modeling** with threshold optimization and cost-benefit analysis
- 🗂️ **Professional documentation** system (8 specialized documentation files)
- 🚀 **Production-ready code** with modular architecture in `src/`

---

## Quick Start

```bash
# Clone the repository
git clone https://github.com/ikartiksavaliya/credit-card-fraud-detection-system-using-deep-learning.git
cd credit-card-fraud-detection-system-using-deep-learning

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook

# Open notebook 01_eda.ipynb to get started!
```

---

## Prerequisites

- **Python 3.10 or higher**
- **pip** (Python package manager)
- **Git** for cloning the repository
- **4GB+ RAM** (8GB recommended for smooth operation)
- **2GB+ disk space** for dependencies and outputs

### System-Specific Notes

**Windows Users:**
- Use `venv\Scripts\activate` instead of `source venv/bin/activate`
- If you encounter PyTorch installation issues, visit [pytorch.org](https://pytorch.org) for OS-specific commands

**macOS Users:**
- If using Apple Silicon (M1/M2/M3), you may need to install PyTorch via conda:
  ```bash
  conda install pytorch::pytorch torchvision torchaudio -c pytorch
  ```

**Linux Users:**
- Ensure development headers are installed:
  ```bash
  sudo apt-get install python3-dev python3-venv build-essential
  ```

---

## Installation

### Option 1: Virtual Environment (Recommended)

```bash
# Clone repository
git clone https://github.com/ikartiksavaliya/credit-card-fraud-detection-system-using-deep-learning.git
cd credit-card-fraud-detection-system-using-deep-learning

# Create virtual environment
python -m venv venv

# Activate virtual environment
source venv/bin/activate  # Linux/macOS
# OR
venv\Scripts\activate  # Windows

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
```

### Option 2: Conda Environment

```bash
# Clone repository
git clone https://github.com/ikartiksavaliya/credit-card-fraud-detection-system-using-deep-learning.git
cd credit-card-fraud-detection-system-using-deep-learning

# Create conda environment
conda create -n fraud-detection python=3.10

# Activate environment
conda activate fraud-detection

# Install dependencies
pip install -r requirements.txt
```

### Option 3: Docker (Coming Soon)

A Dockerfile will be provided for containerized deployment.

---

## Usage Guide

### 1. **Running the Complete Pipeline**

```bash
# Start Jupyter Notebook
jupyter notebook

# Execute notebooks in order:
# 1. 01_eda.ipynb - Explore and understand the data
# 2. 02_preprocessing.ipynb - Prepare data for modeling
# 3. 03_baseline_mlp.ipynb - Build baseline model
# 4. 04_activation_study.ipynb through 10_threshold_optimization.ipynb - Advanced studies
# 5. 11_final_report.ipynb - Generate final report
```

### 2. **Quick Model Training**

```python
# In a Python script or notebook cell
import torch
from src.models import SimpleMLP
from src.training import train_model
from src.preprocessing import load_and_preprocess_data

# Load data
X_train, X_test, y_train, y_test = load_and_preprocess_data('data/credit_card_fraud_10k.csv')

# Create model
model = SimpleMLP(input_dim=8, hidden_dims=[64, 32], output_dim=1)

# Train
history = train_model(model, X_train, y_train, epochs=50, batch_size=32)

# Save model
torch.save(model.state_dict(), 'outputs/models/fraud_detector.pt')
```

### 3. **Making Predictions**

```python
import torch
from src.models import SimpleMLP

# Load trained model
model = SimpleMLP(input_dim=8, hidden_dims=[64, 32], output_dim=1)
model.load_state_dict(torch.load('outputs/models/fraud_detector.pt'))
model.eval()

# Make predictions
with torch.no_grad():
    predictions = model(X_test)
    fraud_probabilities = torch.sigmoid(predictions).numpy()
```

### 4. **Running Individual Studies**

Each study notebook is self-contained and can be run independently:

```bash
# Run specific notebook
jupyter notebook notebooks/04_activation_study.ipynb

# Or execute from command line
jupyter nbconvert --to notebook --execute notebooks/04_activation_study.ipynb
```

---

## Project Structure

```
credit-card-fraud-detection-system-using-deep-learning/
│
├── data/
│   └── credit_card_fraud_10k.csv          # Main dataset (10,000 transactions)
│
├── notebooks/
│   ├── 01_eda.ipynb                       # Exploratory Data Analysis
│   ├── 02_preprocessing.ipynb             # Feature Engineering & Preprocessing
│   ├── 03_baseline_mlp.ipynb              # NN Foundations + Baseline Model
│   ├── 04_activation_study.ipynb          # Activation Functions: Sigmoid vs Tanh vs ReLU vs GELU
│   ├── 05_optimizer_study.ipynb           # Optimizer Comparison: SGD vs Adam vs AdamW (7 variants)
│   ├── 06_initialization_study.ipynb      # Weight Init: Random vs Xavier vs He
│   ├── 07_regularization_study.ipynb      # Regularization: Dropout vs BatchNorm vs L1/L2
│   ├── 08_class_imbalance_study.ipynb     # Imbalance Handling: Baseline vs WeightedBCE vs SMOTE
│   ├── 09_advanced_model.ipynb            # Advanced Architecture & Techniques
│   ├── 10_threshold_optimization.ipynb    # Business-Optimal Threshold Selection
│   └── 11_final_report.ipynb              # Final Report & Portfolio Assembly
│
├── src/
│   ├── __init__.py                        # Package initialization
│   ├── preprocessing.py                   # Data cleaning, encoding, scaling, SMOTE
│   ├── models.py                          # MLP architectures (configurable)
│   ├── training.py                        # Training loop, schedulers, early stopping
│   ├── evaluation.py                      # Metrics, ROC, PR curves, confusion matrix
│   └── utils.py                           # Seeding, plotting, logging utilities
│
├── docs/
│   ├── PROJECT_MASTER.md                  # Master reference document
│   ├── LEARNING_CHECKLIST.md              # Topic completion tracking
│   ├── TOPIC_COVERAGE_MATRIX.md           # Which notebook covers which topic
│   ├── EXPERIMENT_TRACKER.md              # All experiment results
│   ├── DECISION_LOG.md                    # Technical decisions + rationale
│   ├── MODEL_EVOLUTION.md                 # Model version history
│   ├── INTERVIEW_NOTES.md                 # Deep learning interview Q&A
│   └── BUSINESS_IMPACT.md                 # Business context + cost analysis
│
├── outputs/
│   ├── figures/                           # All saved plots and visualizations
│   ├── models/                            # Saved model checkpoints (.pt files)
│   └── reports/                           # Generated HTML/PDF reports
│
├── requirements.txt                       # Python dependencies
├── README.md                              # This file
└── LICENSE                                # MIT License

```

---

## Dataset Information

| Property | Details |
|---|---|
| **Source** | Synthetic credit card transaction data (local CSV) |
| **Size** | 10,000 transactions |
| **Input Features** | 9 engineered features |
| **Target Variable** | Binary classification (0=Legitimate, 1=Fraud) |
| **Class Distribution** | ~3–10% fraud rate (imbalanced) |
| **Data Format** | CSV with headers |

### Features Description

| Feature | Description |
|---|---|
| `amount` | Transaction amount in currency units |
| `transaction_hour` | Hour of day (0-23) transaction occurred |
| `merchant_category` | Merchant category code |
| `foreign_transaction` | Boolean: 1 if overseas transaction |
| `location_mismatch` | Boolean: 1 if location unusual for cardholder |
| `device_trust_score` | Device trustworthiness score (0-1) |
| `velocity_last_24h` | Number of transactions in last 24 hours |
| `cardholder_age` | Age of cardholder in years |
| `target` | 1 if fraudulent, 0 if legitimate |

---

## Key Results

Model progression on holdout test set (default threshold T = 0.500):

| Model Version | Recall | Precision | F1-Score | PR-AUC | ROC-AUC |
|---|---|---|---|---|---|
| **Baseline MLP** | 73.9% | 81.0% | 77.3% | 0.8422 | 0.9964 |
| **+ Best Activation** (Leaky ReLU) | 73.9% | 70.8% | 72.3% | 0.8338 | 0.9962 |
| **+ Best Optimizer** (AdamW + Warmup Cosine) | 69.6% | 72.7% | 71.1% | 0.7936 | 0.9954 |
| **+ Best Initialization** (Random Normal) | 69.6% | 84.2% | 76.2% | 0.8655 | 0.9975 |
| **+ Regularization** (Early Stopping / Gradient Clipping) | 73.9% | 85.0% | 79.1% | 0.8370 | 0.9968 |
| **+ Class Imbalance** (WeightedRandomSampler) | 95.7% | 44.9% | 61.1% | 0.8580 | 0.9967 |
| **Final (Threshold Optimized)** | 95.7% | 38.6% | 55.0% | 0.8273 | 0.9960 |

**Insights:**
- High recall (95.7%) catches most fraud cases
- Precision-recall tradeoff depends on business requirements
- Threshold optimization is crucial for production deployment

---

## Learning Roadmap

| Phase | Topic | Notebook | Status |
|---|---|---|---|
| 1 | Planning & Dataset Inspection | — | ✅ Complete |
| 2 | Exploratory Data Analysis | `01_eda.ipynb` | ✅ Complete |
| 3 | Preprocessing & Feature Engineering | `02_preprocessing.ipynb` | ✅ Complete |
| 4 | NN Foundations & Baseline MLP | `03_baseline_mlp.ipynb` | ✅ Complete |
| 5 | Activation Function Study | `04_activation_study.ipynb` | ✅ Complete |
| 6 | Optimizer Study | `05_optimizer_study.ipynb` | ✅ Complete |
| 7 | Weight Initialization Study | `06_initialization_study.ipynb` | ✅ Complete |
| 8 | Regularization Study | `07_regularization_study.ipynb` | ✅ Complete |
| 9 | Class Imbalance Study | `08_class_imbalance_study.ipynb` | ✅ Complete |
| 10 | Advanced Architecture | `09_advanced_model.ipynb` | ✅ Complete |
| 11 | Threshold Optimization | `10_threshold_optimization.ipynb` | ✅ Complete |
| 12 | Final Report & Portfolio | `11_final_report.ipynb` | ✅ Complete |

---

## Documentation

All detailed documentation is located in the `docs/` folder:

- **[`PROJECT_MASTER.md`](docs/PROJECT_MASTER.md)** — Master reference and project overview
- **[`LEARNING_CHECKLIST.md`](docs/LEARNING_CHECKLIST.md)** — Deep learning topic completion tracker
- **[`TOPIC_COVERAGE_MATRIX.md`](docs/TOPIC_COVERAGE_MATRIX.md)** — Which notebook covers which topic
- **[`EXPERIMENT_TRACKER.md`](docs/EXPERIMENT_TRACKER.md)** — Complete experiment results and metrics
- **[`DECISION_LOG.md`](docs/DECISION_LOG.md)** — Technical decisions and rationale
- **[`MODEL_EVOLUTION.md`](docs/MODEL_EVOLUTION.md)** — Detailed model version history
- **[`INTERVIEW_NOTES.md`](docs/INTERVIEW_NOTES.md)** — Deep learning interview Q&A
- **[`BUSINESS_IMPACT.md`](docs/BUSINESS_IMPACT.md)** — Business context and cost analysis

---

## Troubleshooting

### Common Issues and Solutions

#### 1. **PyTorch Installation Fails**

**Error:** `ERROR: Could not find a version that satisfies the requirement torch`

**Solution:**
```bash
# Try installing CPU version first
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# For GPU (CUDA 11.8):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Visit https://pytorch.org for your specific configuration
```

#### 2. **Jupyter Notebook Won't Start**

**Error:** `Command 'jupyter' not found`

**Solution:**
```bash
# Reinstall jupyter
pip install --upgrade jupyter notebook ipykernel

# Register kernel
python -m ipykernel install --user

# Try again
jupyter notebook
```

#### 3. **Data File Not Found**

**Error:** `FileNotFoundError: data/credit_card_fraud_10k.csv not found`

**Solution:**
```bash
# Ensure you're in the correct directory
pwd  # or 'cd' on Windows

# Check if data exists
ls data/  # Linux/macOS
dir data  # Windows

# If missing, download or regenerate the dataset
```

#### 4. **Memory Issues During Training**

**Error:** `RuntimeError: CUDA out of memory` or `MemoryError`

**Solution:**
```python
# Reduce batch size in training
batch_size = 16  # Instead of 32

# Reduce model complexity
hidden_dims = [32, 16]  # Instead of [128, 64, 32]

# Clear cache between epochs
import torch
torch.cuda.empty_cache()
```

#### 5. **Slow Performance on Windows**

**Cause:** Antivirus or Windows Defender interference

**Solution:**
- Add Python and Jupyter to antivirus exclusions
- Disable real-time protection temporarily during training
- Use WSL (Windows Subsystem for Linux) for better performance

#### 6. **Import Errors in Notebooks**

**Error:** `ModuleNotFoundError: No module named 'src'`

**Solution:**
```python
# Add project root to path at the beginning of notebook
import sys
sys.path.insert(0, '/path/to/credit-card-fraud-detection-system-using-deep-learning')

# Or ensure you're running notebooks from the project root directory
```

---

### Performance Optimization Tips

1. **Use GPU if Available:**
   ```python
   import torch
   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   print(f"Using device: {device}")
   ```

2. **Parallelize Data Loading:**
   ```python
   DataLoader(..., num_workers=4, pin_memory=True)
   ```

3. **Monitor Memory:**
   ```python
   import psutil
   process = psutil.Process()
   print(f"Memory usage: {process.memory_info().rss / 1024**2:.2f} MB")
   ```

---

## Quick Reference

### Run All Notebooks in Sequence

```bash
for notebook in notebooks/{01..11}_*.ipynb; do
    jupyter nbconvert --to notebook --execute "$notebook"
done
```

### Generate Results Report

See `11_final_report.ipynb` for automated report generation.

### Access Model Outputs

```bash
ls outputs/figures/       # View generated plots
ls outputs/models/        # View saved model checkpoints
ls outputs/reports/       # View generated reports
```

---

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

## Contact & Support

- **Email:** ikartiksavaliya@gmail.com
- **GitHub:** [ikartiksavaliya](https://github.com/ikartiksavaliya)
- **Issues:** [Report a bug or feature request](https://github.com/ikartiksavaliya/credit-card-fraud-detection-system-using-deep-learning/issues)

---

*Built with ❤️ for mastering deep learning from first principles.*

**Last Updated:** June 2026
