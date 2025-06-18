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

# 🏃 How to Run PolyA-GLM Training

## Step-by-Step Execution Guide

### Step 1: Prepare Your Environment
```bash
# Activate your conda environment
conda activate polya-glm

# Navigate to the project directory
cd PolyA-GLM/

# Verify your data is ready
ls data/your_dataset/merged.csv  # Should exist with sequences in column 4, labels in column 6
```

### Step 2: Create a bash file

```bash
# Create a new bash script
nano train_polya.sh
```

### Step 3: Copy the Script Content
```bash
#!/bin/bash
# PolyA-GLM Training Script

# Configuration - MODIFY THESE PATHS
data_path="/home/sourav/Poly_A_Site/data/Human/intergene"  # Path to directory containing merged.cs
lr=1e-6
MAX_LENGTH=512
seed=42


echo "Starting PolyA-GLM training with data_path: $data_path"

python train.py \
    --model_name_or_path zhihan1996/DNABERT-2-117M \ # Pre-trained model from HuggingFace
    --data_path ${data_path} \
    --kmer -1 \ # For NMT use kmer number
    --run_name DNABERT2_polya_seed${seed} \ # Experiment name for tracking 
    --model_max_length ${MAX_LENGTH} \ # Maximum input sequence length
    --per_device_train_batch_size 4 \ # Training batch size per GPU
    --per_device_eval_batch_size 4 \ # Evaluation batch size per GPU
    --gradient_accumulation_steps 1 \ # Number of steps to accumulate gradients. Increase if you need larger effective batch size. Effective batch = batch_size * accumulation_steps
    --learning_rate ${lr} \ # Base learning rate for optimizer
    --num_train_epochs 3 \ # Number of training epochs
    --fp16 \
    --save_steps 200 \ # Save model checkpoint every N steps 
    --output_dir results/dnabert2_polya_training/ \ # Directory to save results and models
    --evaluation_strategy steps \ # When to run evaluation. 'steps': Evaluate every eval_steps, 'epoch': Evaluate at end of each epoch
    --eval_steps 200 \ # Evaluate every N training steps
    --warmup_steps 50 \ # Number of warmup steps for learning rate
    --logging_steps 200 \ # Log training metrics every N steps
    --overwrite_output_dir True \ # Overwrite existing output directory. Set to False to prevent accidental overwrites
    --log_level info \
    --seed ${seed} # Random seed for reproducible results. Ensures same train/val/test splits across runs

echo "Training completed! Check results in results/dnabert2_polya_training/"
```

### Step 4: Make Script Executable and Run
```bash
# Make the script executable
chmod +x train_polya.sh

# Run the training
./train_polya.sh
```

## Optional
### Run in Background with Logging
```bash
# Run training in background with output logging
nohup ./train_polya.sh > training_output.log 2>&1 &

# Check the process
ps aux | grep train.py

# Monitor progress
tail -f training_output.log
```

### Token Classification Training

#### Overview
Token classification provides **nucleotide-level polyadenylation site prediction**, identifying individual nucleotides that correspond to polyadenylation sites across long genomic sequences.

#### Step 1: Create Token Classification Script
```bash
# Create the token classification training script
nano train_token_classification.sh
```

#### Step 2: Script Content
```bash
#!/bin/bash
# Token Classification Training Script for PolyA-GLM

# Configuration - MODIFY THESE PATHS
BASE_MODEL_PATH="LongSafari/hyenadna-small-32k-seqlen-hf"     # HyenaDNA model path
MERGED_CSV="/path/to/your/token_classification_v2.csv"        # Your token classification data
OUTPUT_DIR="results/token_classification_output/"              # Output directory
NUM_LABELS=2                                                   # Binary classification
EPOCHS=3                                                       # Training epochs
BATCH_SIZE=1                                                   # Batch size (keep small for memory)
SEED=42                                                        # Random seed

echo "Starting Token Classification Training..."
echo "Model: $BASE_MODEL_PATH"
echo "Data: $MERGED_CSV"
echo "Output: $OUTPUT_DIR"

# Run token classification training
python model.py \
    --base_model_path "$BASE_MODEL_PATH" \
    --merged_csv "$MERGED_CSV" \
    --output_dir "$OUTPUT_DIR" \
    --num_labels $NUM_LABELS \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --seed $SEED

echo "Token classification training completed!"
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Check outputs:"
echo "- Training/validation/test splits: $OUTPUT_DIR/*.csv"
echo "- Token-level predictions: $OUTPUT_DIR/test_token_predictions.csv"
echo "- Final model: $OUTPUT_DIR/final_model/"
```

#### Step 3: Execute the Script
```bash
# Make script executable
chmod +x train_token_classification.sh

# Run token classification training
./train_token_classification.sh
```

#### Optional: Background Training (Recommended for Long Sequences)
```bash
# Run in background with logging
nohup ./train_token_classification.sh > token_training.log 2>&1 &

# Monitor progress
tail -f token_training.log

# Check if training is running
ps aux | grep "python model.py"

# Check GPU usage
watch -n 1 nvidia-smi
```

#### Data Format Requirements
Your CSV file should contain:
- **sequence**: DNA sequences (up to 32,000 nucleotides)
- **labels**: Space-separated binary labels (0 or 1) for each nucleotide

```csv
sequence,labels
ATCGATCGATCG...,0 0 0 1 0 0 0 0 0 1 0 0...
GCTAGCTAGCTA...,0 0 0 0 0 1 0 0 0 0 0 0...
```

#### Expected Outputs
After training, your output directory will contain:
- `train_split.csv`, `val_split.csv`, `test_split.csv` - Data splits
- `test_token_predictions.csv` - Nucleotide-level predictions
- `final_model/` - Trained model for inference

## Real-Time Monitoring During Training

### Monitor GPU Usage
```bash
# In a separate terminal, monitor GPU
watch -n 1 nvidia-smi
```

### Monitor Training Progress
```bash
# Watch training logs
tail -f results/your_training_dir/fold_*/train.log

# Monitor CPU/Memory usage
htop
```

### Check Training Status
```bash
# Check if training is still running
ps aux | grep "python train.py"

# Check current fold progress
ls -la results/your_training_dir/fold_*/

# View latest metrics
cat results/your_training_dir/fold_1/val_results.json
```
## Execution Checklist

Before running training, ensure:

- [ ] **Environment activated**: `conda activate polya-glm`
- [ ] **Data prepared**: `merged.csv` exists in your data directory
- [ ] **GPU available**: `nvidia-smi` shows available memory
- [ ] **Output directory**: Ensure sufficient disk space
- [ ] **Permissions**: Scripts are executable (`chmod +x script.sh`)

