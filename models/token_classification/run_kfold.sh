#!/usr/bin/env bash

# BASE_FOLDER="/home/sourav/Poly_A_Site/data/Human/"
# CSV_FILE="token_classification_v2.csv"
# MODEL_NAME="LongSafari/hyenadna-medium-160k-seqlen-hf"
# OUTPUT_DIR="/home/sourav/Poly_A_Site/token_classification/model_code/output"
# SPLITS=5
# NUM_EPOCHS=3
# BATCH_SIZE=8
# MAX_LENGTH=160000

# python model.py \
#   --base_folder $BASE_FOLDER \
#   --csv_file $CSV_FILE \
#   --model_name $MODEL_NAME \
#   --output_dir $OUTPUT_DIR \
#   --n_splits $SPLITS \
#   --num_epochs $NUM_EPOCHS \
#   --batch_size $BATCH_SIZE \
#   --max_length $MAX_LENGTH

#!/bin/bash

# BASE_MODEL_PATH="LongSafari/hyenadna-medium-160k-seqlen-hf"
# MERGED_CSV="/home/sourav/Poly_A_Site/data/Human/token_classification_v2.csv"  # Single file containing all examples
# OUTPUT_DIR="hyena_tokencls_output"

# python my_training_script.py \
#   --base_model_path "$BASE_MODEL_PATH" \
#   --merged_csv "$MERGED_CSV" \
#   --output_dir "$OUTPUT_DIR"




# Run the Python script with these arguments
python model.py \
  --base_model_path "LongSafari/hyenadna-small-32k-seqlen-hf" \
  --merged_csv "/home/sourav/sourav/Poly_A_Site/data/Human/token_classification_v2.csv" \
  --output_dir "/home/sourav/sourav/Poly_A_Site/token_classification/model_code/output_refined/" \
  --num_labels 2 \
  --epochs 3 \
  --batch_size 1 \
  --seed 42
