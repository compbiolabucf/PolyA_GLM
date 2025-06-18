#!/usr/bin/env python3
"""
Few-shot polyadenylation site classification using genome language models.
Enhanced version with command-line arguments and configurable parameters.
"""

import argparse
import os
import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from scipy.spatial.distance import cosine
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
import json
from datetime import datetime

def get_embedding(dna_sequence: str, model, tokenizer) -> np.ndarray:
    """
    Convert a DNA sequence to a numeric embedding using the model's
    last hidden states, mean-pooled across the sequence dimension.
    """
    inputs = tokenizer(dna_sequence, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    # outputs.last_hidden_state has shape [batch_size, seq_len, hidden_dim]
    # We'll take the mean over seq_len => [batch_size, hidden_dim]
    embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
    return embedding

def get_prototypes_from_data(train_csv_path: str, model, tokenizer, n_prototypes: int = 2):
    """
    Extract prototypes from training data by randomly sampling sequences from each class.
    """
    train_df = pd.read_csv(train_csv_path)
    train_df["Label"] = train_df["Label"].astype(int)
    
    # Sample prototypes for each class
    prototypes = {}
    
    # Poly-A site sequences (label = 1)
    poly_a_sequences = train_df[train_df["Label"] == 1]["Raw Sequence"].tolist()
    if len(poly_a_sequences) >= n_prototypes:
        prototypes["poly_a_site"] = np.random.choice(poly_a_sequences, n_prototypes, replace=False).tolist()
    else:
        prototypes["poly_a_site"] = poly_a_sequences
    
    # Non-poly-A site sequences (label = 0)
    non_poly_a_sequences = train_df[train_df["Label"] == 0]["Raw Sequence"].tolist()
    if len(non_poly_a_sequences) >= n_prototypes:
        prototypes["not_poly_a_site"] = np.random.choice(non_poly_a_sequences, n_prototypes, replace=False).tolist()
    else:
        prototypes["not_poly_a_site"] = non_poly_a_sequences
    
    print(f"Selected {len(prototypes['poly_a_site'])} poly-A prototypes and {len(prototypes['not_poly_a_site'])} non-poly-A prototypes")
    
    return prototypes

def classify_sequence_and_score(dna_sequence: str, label_embeddings: dict, model, tokenizer):
    """
    Returns:
      pred_label: 0 or 1 (integer)
      prob_poly: continuous score [0,1] for "poly_a_site"
    """
    query_emb = get_embedding(dna_sequence, model, tokenizer)
    
    # Compute cosine similarity to each prototype
    sim_poly = 1 - cosine(query_emb, label_embeddings["poly_a_site"])
    sim_not_poly = 1 - cosine(query_emb, label_embeddings["not_poly_a_site"])
    
    # Convert similarities to a "probability-like" score via softmax
    exp_poly = np.exp(sim_poly)
    exp_not = np.exp(sim_not_poly)
    sum_exp = exp_poly + exp_not
    prob_poly = exp_poly / sum_exp  # Probability of "poly_a_site"
    
    # Convert to binary prediction
    pred_label = 1 if prob_poly >= 0.5 else 0
    
    return pred_label, prob_poly

def main():
    parser = argparse.ArgumentParser(description="Few-shot polyadenylation site classification")
    
    # Model arguments
    parser.add_argument("--model_name", type=str, required=True,
                        help="HuggingFace model name (e.g., InstaDeepAI/nucleotide-transformer-500m-human-ref)")
    
    # Data arguments
    parser.add_argument("--train_csv", type=str, required=True,
                        help="Path to training CSV file for prototype extraction")
    parser.add_argument("--test_csv", type=str, required=True,
                        help="Path to test CSV file")
    
    # Few-shot arguments
    parser.add_argument("--n_prototypes", type=int, default=2,
                        help="Number of prototype sequences per class")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Classification threshold for positive class")
    
    # Output arguments
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save results")
    parser.add_argument("--run_name", type=str, default="few_shot_experiment",
                        help="Name for this experimental run")
    
    # Other arguments
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device to use (auto, cpu, cuda)")
    
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    print(f"Using device: {device}")
    print(f"Model: {args.model_name}")
    print(f"Training data: {args.train_csv}")
    print(f"Test data: {args.test_csv}")
    print(f"Output directory: {args.output_dir}")
    
    # --------------------
    # 1. Load Model and Tokenizer
    # --------------------
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name)
    model.to(device)
    model.eval()  # Put model in evaluation mode
    
    print("Model loaded successfully. Few-shot classification will now begin.")
    
    # --------------------
    # 2. Extract Prototypes from Training Data
    # --------------------
    print("Extracting prototypes from training data...")
    prototypes = get_prototypes_from_data(args.train_csv, model, tokenizer, args.n_prototypes)
    
    # Save prototypes for reference
    prototypes_path = os.path.join(args.output_dir, "prototypes.json")
    with open(prototypes_path, 'w') as f:
        json.dump(prototypes, f, indent=2)
    print(f"Prototypes saved to {prototypes_path}")
    
    # --------------------
    # 3. Compute Prototype Embeddings
    # --------------------
    print("Computing prototype embeddings...")
    label_embeddings = {}
    for label, seq_list in prototypes.items():
        emb_list = [get_embedding(seq, model, tokenizer) for seq in seq_list]
        label_embeddings[label] = np.mean(emb_list, axis=0)
    
    # --------------------
    # 4. Load Test Data
    # --------------------
    print("Loading test data...")
    df = pd.read_csv(args.test_csv)
    df["Label"] = df["Label"].astype(int)
    
    print(f"Test data loaded: {len(df)} sequences")
    print(f"Class distribution: {df['Label'].value_counts().to_dict()}")
    
    # --------------------
    # 5. Generate Predictions & Probabilities
    # --------------------
    print("Generating predictions...")
    pred_labels = []
    prob_scores = []
    
    for i, seq in enumerate(df["Raw Sequence"]):
        if i % 100 == 0:
            print(f"Processing sequence {i+1}/{len(df)}")
        
        label_int, prob_val = classify_sequence_and_score(seq, label_embeddings, model, tokenizer)
        
        # Apply custom threshold
        label_int = 1 if prob_val >= args.threshold else 0
        
        pred_labels.append(label_int)
        prob_scores.append(prob_val)
    
    df["pred_label"] = pred_labels
    df["prob_poly_a"] = prob_scores
    
    # --------------------
    # 6. Evaluate Performance
    # --------------------
    print("Evaluating performance...")
    y_true = df["Label"].tolist()
    y_pred = df["pred_label"].tolist()
    y_score = df["prob_poly_a"].tolist()
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    recall = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    
    # Compute AUC using the continuous probability-like score
    try:
        auc = roc_auc_score(y_true, y_score)
    except ValueError:
        auc = 0.0  # Handle case where all predictions are same class
    
    # Calculate additional metrics
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    
    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "total_sequences": len(df),
        "threshold": args.threshold,
        "n_prototypes": args.n_prototypes,
        "model_name": args.model_name,
        "timestamp": datetime.now().isoformat()
    }
    
    print("\n=== Few-Shot Classification Metrics ===")
    print(f"Model: {args.model_name}")
    print(f"Prototypes per class: {args.n_prototypes}")
    print(f"Classification threshold: {args.threshold}")
    print(f"Total sequences: {len(df)}")
    print("---")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"AUC:       {auc:.4f}")
    print("---")
    print(f"TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}")
    
    # --------------------
    # 7. Save Results
    # --------------------
    # Save predictions
    predictions_path = os.path.join(args.output_dir, f"{args.run_name}_predictions.csv")
    df[["Raw Sequence", "Label", "pred_label", "prob_poly_a"]].to_csv(predictions_path, index=False)
    print(f"\nPredictions saved to {predictions_path}")
    
    # Save metrics
    metrics_path = os.path.join(args.output_dir, f"{args.run_name}_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_path}")
    
    # Save summary report
    report_path = os.path.join(args.output_dir, f"{args.run_name}_report.txt")
    with open(report_path, 'w') as f:
        f.write("Few-Shot Polyadenylation Site Classification Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Experiment: {args.run_name}\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model: {args.model_name}\n")
        f.write(f"Training data: {args.train_csv}\n")
        f.write(f"Test data: {args.test_csv}\n")
        f.write(f"Prototypes per class: {args.n_prototypes}\n")
        f.write(f"Classification threshold: {args.threshold}\n")
        f.write(f"Random seed: {args.seed}\n\n")
        
        f.write("Results:\n")
        f.write("-" * 20 + "\n")
        for key, value in metrics.items():
            if key not in ["timestamp", "model_name"]:
                f.write(f"{key}: {value}\n")
    
    print(f"Summary report saved to {report_path}")
    print("\nFew-shot classification completed successfully!")

if __name__ == "__main__":
    main()