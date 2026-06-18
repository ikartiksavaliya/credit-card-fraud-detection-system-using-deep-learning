# 🚀 Getting Started Guide

Welcome to the Credit Card Fraud Detection System! This guide will help you set up the project and run it successfully.

---

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Installation Steps](#installation-steps)
3. [First Run](#first-run)
4. [Understanding the Project](#understanding-the-project)
5. [Next Steps](#next-steps)

---

## System Requirements

### Minimum Requirements
- **OS:** Windows 10, macOS 10.14+, Linux (Ubuntu 18.04+)
- **Python:** 3.10, 3.11, or 3.12
- **RAM:** 4 GB (8 GB recommended)
- **Disk Space:** 2 GB for dependencies and outputs
- **Internet:** Required for initial setup

### Recommended Setup
- **OS:** Linux (Ubuntu 20.04+) or macOS
- **Python:** 3.11 or 3.12
- **RAM:** 16 GB or more
- **GPU:** NVIDIA GPU with CUDA support (optional but helpful)
- **Disk Space:** 5+ GB for future projects

### Check Your Python Version

```bash
python --version
# Expected output: Python 3.10.x, 3.11.x, or 3.12.x

# If you have multiple Python versions, specify explicitly:
python3 --version
python3.10 --version
```

---

## Installation Steps

### Step 1: Clone the Repository

```bash
# Via HTTPS (recommended for beginners)
git clone https://github.com/ikartiksavaliya/credit-card-fraud-detection-system-using-deep-learning.git
cd credit-card-fraud-detection-system-using-deep-learning

# OR via SSH (if you have SSH keys configured)
git clone git@github.com:ikartiksavaliya/credit-card-fraud-detection-system-using-deep-learning.git
cd credit-card-fraud-detection-system-using-deep-learning
```

### Step 2: Create a Virtual Environment

Virtual environments keep this project's dependencies isolated from your system Python.

#### On Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

#### On macOS/Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```

**Verify activation:** Your terminal prompt should show `(venv)` at the beginning.

### Step 3: Upgrade pip

```bash
pip install --upgrade pip
# On some systems, use pip3 instead:
pip3 install --upgrade pip
```

### Step 4: Install Dependencies

```bash
# Basic installation (recommended for most users)
pip install -r requirements.txt

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import sklearn; print(f'Scikit-learn: {sklearn.__version__}')"
```

### Step 5: (Optional) GPU Support

If you have an NVIDIA GPU and want to use CUDA:

```bash
# For CUDA 11.8 (most common)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Verify GPU is available
python -c "import torch; print(f'GPU Available: {torch.cuda.is_available()}')"
```

---

## First Run

### Launch Jupyter Notebook

```bash
jupyter notebook
```

This will open Jupyter in your default browser at `http://localhost:8888`

### Run Your First Notebook

1. In the Jupyter interface, navigate to `notebooks/` folder
2. Click on `01_eda.ipynb` (Exploratory Data Analysis)
3. Press `Shift + Enter` to run each cell sequentially
4. Wait for each cell to complete before moving to the next

### Expected Output

After running the EDA notebook, you should see:
- Data summary statistics
- Class distribution visualization
- Feature distributions
- Correlation heatmap
- Basic exploratory plots

If successful, you're ready to continue with other notebooks!

---

## Understanding the Project Structure

```
project/
├── notebooks/           # Run these in order (01 → 11)
├── src/                 # Python modules (import these in notebooks)
├── data/                # Your input data
├── outputs/             # Generated results (automatically created)
├── docs/                # Detailed documentation
└── requirements.txt     # Dependencies
```

### Notebook Order (Follow This Path)

1. **`01_eda.ipynb`** — Understand your data
2. **`02_preprocessing.ipynb`** — Clean and prepare data
3. **`03_baseline_mlp.ipynb`** — Build first model
4. **`04_activation_study.ipynb`** — Compare activation functions
5. **`05_optimizer_study.ipynb`** — Compare optimizers
6. **`06_initialization_study.ipynb`** — Compare weight initialization
7. **`07_regularization_study.ipynb`** — Compare regularization techniques
8. **`08_class_imbalance_study.ipynb`** — Handle imbalanced classes
9. **`09_advanced_model.ipynb`** — Build advanced model
10. **`10_threshold_optimization.ipynb`** — Optimize prediction threshold
11. **`11_final_report.ipynb`** — Generate final report

---

## Common Issues & Quick Fixes

### Issue: "Command 'python' not found"

**Solution:**
```bash
# Try python3 instead
python3 --version
python3 -m venv venv
```

### Issue: "No module named 'torch'"

**Solution:**
```bash
# Make sure virtual environment is activated
source venv/bin/activate  # macOS/Linux
# OR
venv\Scripts\activate     # Windows

# Then reinstall
pip install torch
```

### Issue: "Jupyter not found"

**Solution:**
```bash
# Reinstall jupyter
pip install jupyter notebook

# Then try launching again
jupyter notebook
```

### Issue: "data/credit_card_fraud_10k.csv not found"

**Solution:**
```bash
# Make sure you're in the project root directory
pwd  # Check current directory

# If you cloned but can't see the data file:
git pull  # Get latest files
```

### Issue: Notebooks run very slowly

**Possible causes:**
- Insufficient RAM
- Antivirus scanning Jupyter files
- Running on CPU instead of GPU

**Solutions:**
- Close unnecessary applications
- Add Python to antivirus exceptions
- Check if GPU is available: `python -c "import torch; print(torch.cuda.is_available())"`

---

## Next Steps

### Once You've Completed Installation:

1. **Run the EDA Notebook** (`01_eda.ipynb`)
   - Explore the data
   - Understand features and target distribution
   - Identify potential issues

2. **Read the Documentation**
   - Start with `docs/PROJECT_MASTER.md`
   - Review `docs/LEARNING_CHECKLIST.md` for topics covered

3. **Follow the Notebooks**
   - Execute each notebook sequentially
   - Read code comments and markdown explanations
   - Understand what each section does

4. **Experiment & Learn**
   - Modify hyperparameters
   - Try different architectures
   - Analyze results

5. **Create Your Own**
   - Try different datasets
   - Build custom models
   - Document your learnings

---

## Getting Help

### Documentation
- **README.md** — Project overview and quick reference
- **docs/PROJECT_MASTER.md** — Detailed project guide
- **docs/INTERVIEW_NOTES.md** — Deep learning concepts explained

### Code Comments
- Each notebook has explanatory markdown cells
- Python files in `src/` have detailed docstrings

### Troubleshooting
- Check `README.md#Troubleshooting` section
- Review error messages carefully
- Search GitHub issues for similar problems

### Support
- Email: ikartiksavaliya@gmail.com
- GitHub Issues: [Report a bug](https://github.com/ikartiksavaliya/credit-card-fraud-detection-system-using-deep-learning/issues)

---

## Quick Reference Commands

```bash
# Activate environment
source venv/bin/activate          # macOS/Linux
venv\Scripts\activate             # Windows

# Start Jupyter
jupyter notebook

# Stop Jupyter (in terminal)
Ctrl + C

# Deactivate environment
deactivate

# List installed packages
pip list

# Update a specific package
pip install --upgrade package_name

# Show package info
pip show package_name
```

---

## Recommended Learning Path

### Beginner (1-2 weeks)
- [ ] Complete setup successfully
- [ ] Run notebooks 01-03
- [ ] Understand data, preprocessing, and baseline model
- [ ] Review `LEARNING_CHECKLIST.md`

### Intermediate (2-4 weeks)
- [ ] Run notebooks 04-08
- [ ] Understand activation functions, optimizers, regularization
- [ ] Experiment with hyperparameters
- [ ] Try different architectures

### Advanced (4+ weeks)
- [ ] Run notebooks 09-11
- [ ] Build custom models
- [ ] Deploy models
- [ ] Work on real datasets

---

**You're all set! 🎉 Start with `01_eda.ipynb` and happy learning!**
