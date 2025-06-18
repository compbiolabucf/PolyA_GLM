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

### Method 3: Few-Shot Classification

#### Overview
Few-shot classification enables **polyadenylation site prediction with minimal training data** by using prototype-based similarity matching. This approach is ideal when you have limited labeled data or want to quickly evaluate model capabilities without full fine-tuning.

#### Features
- **Minimal data requirements**: Uses only 2-10 prototype sequences per class
- **No parameter updates**: Leverages pre-trained model embeddings directly
- **Fast evaluation**: Quick assessment of model performance
- **Prototype extraction**: Automatically selects representative sequences from training data
- **Configurable thresholds**: Adjustable classification confidence levels
- **Multiple model support**: Works with DNABERT2, Nucleotide Transformer, and HyenaDNA

#### When to Use Few-Shot Classification
- **Quick model evaluation**: Rapid assessment before full training
- **Limited data scenarios**: When training data is scarce
- **Baseline establishment**: Compare against fine-tuned models
- **Cross-domain testing**: Evaluate model generalization
- **Prototype analysis**: Understand what the model considers representative

#### Step 1: Create Few-Shot Classification Script
```bash
# Create the few-shot classification script
nano train_few_shot.sh
```

#### Step 2: Script Content
```bash
#!/bin/bash
# Few-Shot Classification Training Script for PolyA-GLM

# Configuration - MODIFY THESE PATHS
MODEL_NAME="InstaDeepAI/nucleotide-transformer-500m-human-ref"    # Model to use for few-shot learning
TRAIN_CSV="/path/to/your/fold_1/train.csv"                       # Training data for prototype extraction
TEST_CSV="/path/to/your/fold_1/test.csv"                         # Test data for evaluation
OUTPUT_DIR="results/few_shot_classification/"                     # Output directory
RUN_NAME="few_shot_nt_fold1"                                     # Experiment name
N_PROTOTYPES=2                                                    # Number of prototypes per class
THRESHOLD=0.5                                                     # Classification threshold
SEED=42                                                           # Random seed

echo "Starting Few-Shot Classification..."
echo "Model: $MODEL_NAME"
echo "Training data: $TRAIN_CSV"
echo "Test data: $TEST_CSV"
echo "Output directory: $OUTPUT_DIR"
echo "Prototypes per class: $N_PROTOTYPES"
echo "Classification threshold: $THRESHOLD"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run few-shot classification
python fewshot_model.py \
    --model_name "$MODEL_NAME" \
    --train_csv "$TRAIN_CSV" \
    --test_csv "$TEST_CSV" \
    --output_dir "$OUTPUT_DIR" \
    --run_name "$RUN_NAME" \
    --n_prototypes $N_PROTOTYPES \
    --threshold $THRESHOLD \
    --seed $SEED \
    --device auto

echo ""
echo "Few-shot classification completed!"
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Check outputs:"
echo "- Predictions: $OUTPUT_DIR/${RUN_NAME}_predictions.csv"
echo "- Metrics: $OUTPUT_DIR/${RUN_NAME}_metrics.json"
echo "- Report: $OUTPUT_DIR/${RUN_NAME}_report.txt"
echo "- Prototypes: $OUTPUT_DIR/prototypes.json"

# Optional: Display quick summary
if [ -f "$OUTPUT_DIR/${RUN_NAME}_metrics.json" ]; then
    echo ""
    echo "Quick Results Summary:"
    echo "====================="
    python3 -c "
import json
with open('$OUTPUT_DIR/${RUN_NAME}_metrics.json', 'r') as f:
    metrics = json.load(f)
print(f'Accuracy: {metrics[\"accuracy\"]:.4f}')
print(f'Precision: {metrics[\"precision\"]:.4f}')
print(f'Recall: {metrics[\"recall\"]:.4f}')
print(f'F1 Score: {metrics[\"f1\"]:.4f}')
print(f'AUC: {metrics[\"auc\"]:.4f}')
"
fi
```

#### Step 3: Execute the Script
```bash
# Make script executable
chmod +x train_few_shot.sh

# Run few-shot classification
./train_few_shot.sh
```

#### Optional: Background Execution
```bash
# Run in background with logging
nohup ./train_few_shot.sh > fewshot_training.log 2>&1 &

# Monitor progress
tail -f fewshot_training.log

# Check if running
ps aux | grep "python fewshot_model.py"
```

#### Model-Specific Configuration

##### DNABERT2 Few-Shot
```bash
# DNABERT2 configuration
MODEL_NAME="zhihan1996/DNABERT-2-117M"
RUN_NAME="few_shot_dnabert2_fold1"
N_PROTOTYPES=2                              # Standard for DNABERT2
THRESHOLD=0.5                               # Default threshold
```

##### Nucleotide Transformer Few-Shot
```bash
# Nucleotide Transformer configuration
MODEL_NAME="InstaDeepAI/nucleotide-transformer-500m-human-ref"
RUN_NAME="few_shot_nt_fold1"
N_PROTOTYPES=3                              # Slightly more prototypes for larger model
THRESHOLD=0.4                               # Lower threshold for higher sensitivity
```

##### HyenaDNA Few-Shot
```bash
# HyenaDNA configuration
MODEL_NAME="LongSafari/hyenadna-medium-450k-seqlen-hf"
RUN_NAME="few_shot_hyena_fold1"
N_PROTOTYPES=2                              # Standard prototypes
THRESHOLD=0.6                               # Higher threshold for precision
```

#### Advanced Configuration Options

##### Multiple Prototype Testing
```bash
# Test different numbers of prototypes
for n_proto in 1 2 3 5; do
    python fewshot_model.py \
        --model_name "InstaDeepAI/nucleotide-transformer-500m-human-ref" \
        --train_csv "data/fold_1/train.csv" \
        --test_csv "data/fold_1/test.csv" \
        --output_dir "results/few_shot_prototypes/" \
        --run_name "nt_${n_proto}proto" \
        --n_prototypes $n_proto \
        --seed 42
done
```

##### Threshold Optimization
```bash
# Test different thresholds
for thresh in 0.3 0.4 0.5 0.6 0.7; do
    python fewshot_model.py \
        --model_name "InstaDeepAI/nucleotide-transformer-500m-human-ref" \
        --train_csv "data/fold_1/train.csv" \
        --test_csv "data/fold_1/test.csv" \
        --output_dir "results/few_shot_thresholds/" \
        --run_name "nt_thresh${thresh}" \
        --threshold $thresh \
        --seed 42
done
```

#### Data Format Requirements
Your CSV files should contain:
- **Raw Sequence**: DNA sequences (101bp for standard PolyA-GLM)
- **Label**: Binary labels (0 or 1)

```csv
Raw Sequence,Label
ATCGATCGATCG...,1
GCTAGCTAGCTA...,0
```

#### Expected Outputs

##### Directory Structure
```
results/few_shot_classification/
├── few_shot_nt_fold1_predictions.csv      # Sequence-level predictions
├── few_shot_nt_fold1_metrics.json         # Performance metrics
├── few_shot_nt_fold1_report.txt           # Detailed report
└── prototypes.json                        # Selected prototype sequences
```

##### Metrics Output
The few-shot classification provides:
- **Accuracy**: Overall classification accuracy
- **Precision**: Positive class precision
- **Recall**: Positive class recall  
- **F1 Score**: Harmonic mean of precision and recall
- **AUC**: Area under the ROC curve
- **Confusion Matrix**: TP, FP, FN, TN counts

#### Cross-Validation with Few-Shot

##### Run Few-Shot Across All Folds
```bash
#!/bin/bash
# Few-shot across all 5 folds

MODEL_NAME="InstaDeepAI/nucleotide-transformer-500m-human-ref"
BASE_DIR="results/cross_validation"
OUTPUT_DIR="results/few_shot_cv"

mkdir -p "$OUTPUT_DIR"

for fold in {1..5}; do
    echo "Running few-shot for fold $fold..."
    
    python fewshot_model.py \
        --model_name "$MODEL_NAME" \
        --train_csv "${BASE_DIR}/fold_${fold}/train.csv" \
        --test_csv "${BASE_DIR}/fold_${fold}/test.csv" \
        --output_dir "$OUTPUT_DIR" \
        --run_name "nt_fold${fold}" \
        --n_prototypes 2 \
        --threshold 0.5 \
        --seed 42
done

echo "Few-shot cross-validation completed!"
```

#### Integration with Pipeline

##### Use Few-Shot for Initial Screening
```bash
# Quick model evaluation before full training
./train_few_shot.sh

# If results are promising, proceed with fine-tuning
./train_polya_model.sh

# Finally, run token classification for high-resolution analysis
./train_token_classification.sh
```
## Execution Checklist

Before running training, ensure:

- [ ] **Environment activated**: `conda activate polya-glm`
- [ ] **Data prepared**: `merged.csv` exists in your data directory
- [ ] **GPU available**: `nvidia-smi` shows available memory
- [ ] **Output directory**: Ensure sufficient disk space
- [ ] **Permissions**: Scripts are executable (`chmod +x script.sh`)

