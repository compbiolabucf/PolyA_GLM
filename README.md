# PolyA-GLM: De Novo Polyadenylation Site Identification Using Genome Language Models
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the implementation of **PolyA-GLM**, a computational framework for predicting polyadenylation sites using genome language models. Our approach leverages state-of-the-art Genome Language Models (DNABERT2, Nucleotide Transformer, HyenaDNA) to identify poly(A) sites with high accuracy through both few-shot classification and fine-tuning approaches. **Notably, PolyA-GLM introduces the first comprehensive framework for de novo discovery of novel polyadenylation sites across the entire human genome, validated through systematic signal perturbation experiments and attention analysis to ensure biological relevance.**

## 🔬 Overview

Polyadenylation sites are critical regulatory elements in post-transcriptional gene regulation. Traditional methods rely on sequence motifs and experimental validation but often fail to generalize across diverse biological contexts. PolyA-GLM addresses this challenge through:

🌟 De Novo Detection

- First genome-wide AI approach: Systematic identification of novel poly(A) sites across all human chromosomes using genome language models
- Biological filtering: Stringent criteria based on canonical cleavage patterns (C/G-A dinucleotides) and 18 polyadenylation signal variants
- Complete genome coverage: Identification of functional sites missed by current annotations

🤖 Multi-Model Architecture
- Dual learning strategies: Few-shot classification (up to 0.75 AUC) and fine-tuning (up to 0.80 AUC)
- Multi-scale analysis: Both sequence-level and token-level classification for comprehensive site identification
- Long-range modeling: Capture genomic dependencies spanning thousands of nucleotides

🧪 Comprehensive Experimental Validation

- Signal perturbation analysis: Systematic validation showing progressive performance degradation as canonical motifs are disrupted
- Attention visualization: Models focus on AATAAA motifs and A-rich/T-rich 3' UTR regions without explicit supervision
- Token-level precision: High-resolution identification achieving 0.87 AUC across 50M+ genomic positions
- Cross-validation framework: Robust 5-fold evaluation ensuring reproducible results


# 🛠️ Installation

## Prerequisites

- **Operating System**: Linux, macOS, or Windows with WSL2
- **Python**: 3.8 or higher
- **Memory**: At least 16GB RAM (32GB recommended for genome-wide analysis)
- **GPU**: CUDA-compatible GPU with 8GB+ VRAM (recommended for model training)
- **Storage**: At least 100GB free space for datasets and results

## Method 1: Conda Environment (Recommended)

### Step 1: Install Conda/Miniconda
```bash
# Download and install Miniconda (if not already installed)
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
# Follow the installation prompts and restart your terminal
```

### Step 2: Clone the Repository
```bash
git clone https://github.com/compbiolabucf/PolyA_GLM.git
cd PolyA-GLM
```

### Step 3: Create and Activate Environment
```bash
# Create environment from the provided YAML file
conda env create -f environment.yml

# Activate the environment
conda activate polya-glm

# Verify installation
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import transformers; print(f'Transformers version: {transformers.__version__}')"
```

## Method 2: Virtual Environment with pip

### Step 1: Clone Repository
```bash
git clone https://github.com/compbiolabucf/PolyA_GLM.git
cd PolyA-GLM
```

### Step 2: Create Virtual Environment
```bash
# Create virtual environment
python -m venv polya_env

# Activate environment
source polya_env/bin/activate  # Linux/macOS
# or
polya_env\Scripts\activate  # Windows
```

### Step 3: Install Dependencies
```bash
# Upgrade pip
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

