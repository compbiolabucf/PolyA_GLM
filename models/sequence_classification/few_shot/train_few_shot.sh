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