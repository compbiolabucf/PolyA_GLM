import os
import csv
import copy
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Optional, Dict, Sequence, Tuple, List, Union

import torch
import transformers
import sklearn
import numpy as np
from torch.utils.data import Dataset
from peft import (
    LoraConfig,
    get_peft_model,
)
from transformers import get_scheduler
from torch.optim.lr_scheduler import CyclicLR
from sklearn.model_selection import StratifiedKFold, train_test_split



from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, precision_score, recall_score, roc_auc_score
from transformers import EvalPrediction
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
print(f"Using GPU: {torch.cuda.get_device_name(0)}")

class CustomSchedulerCallback(transformers.TrainerCallback):
    def __init__(self, scheduler):
        super().__init__()
        self.scheduler = scheduler

    def on_step_end(self, args, state, control, **kwargs):
        # This is called at the end of each training step
        self.scheduler.step()

class WarmUpThenCyclicScheduler:
    def __init__(self, optimizer, num_warmup_steps, base_lr, max_lr, step_size_up, step_size_down):
        self.optimizer = optimizer
        self.warmup_scheduler = get_scheduler(
            "linear",
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=step_size_up + step_size_down
        )
        self.cyclic_scheduler = CyclicLR(
            optimizer,
            base_lr=base_lr,
            max_lr=max_lr,
            step_size_up=step_size_up,
            step_size_down=step_size_down,
            mode="triangular"
        )
        self.step_num = 0
        self.num_warmup_steps = num_warmup_steps

    def step(self):
        if self.step_num < self.num_warmup_steps:
            self.warmup_scheduler.step()
        else:
            self.cyclic_scheduler.step()
        self.step_num += 1

    def state_dict(self):
        return {
            "step_num": self.step_num,
            "warmup_scheduler": self.warmup_scheduler.state_dict(),
            "cyclic_scheduler": self.cyclic_scheduler.state_dict()
        }

    def load_state_dict(self, state_dict):
        self.step_num = state_dict["step_num"]
        self.warmup_scheduler.load_state_dict(state_dict["warmup_scheduler"])
        self.cyclic_scheduler.load_state_dict(state_dict["cyclic_scheduler"])
        
def load_merged_dataset(file_path):
    """
    Load the merged dataset from CSV, reading the raw DNA sequence from column index 4
    and the label from column index 6 (0-based indexing).
    """
    sequences = []
    labels = []
    with open(file_path, "r") as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        for row in reader:
            sequences.append(row[4])  # Raw DNA sequence
            labels.append(int(row[6]))  # Class label

    dataset = [{"raw_sequence": seq, "labels": label} for seq, label in zip(sequences, labels)]
    return dataset
        
        
def save_raw_data_to_csv(sequences, labels, filename):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Raw Sequence", "Label"])

        for seq, label in zip(sequences, labels):
            writer.writerow([seq, label])

    print(f"Saved {filename} successfully.")


def tokenize_sequences(sequences, labels, tokenizer, data_path, kmer):
    """
    Tokenizes the given sequences after splitting, applying k-mer transformation if required.
    """
    
    if kmer != -1:
        sequences = load_or_generate_kmer(data_path, sequences, kmer)

    tokenized = tokenizer(
        sequences,
        return_tensors="pt",
        padding="longest",
        max_length=tokenizer.model_max_length,
        truncation=True,
    )

    dataset = []
    for i in range(len(labels)):
        dataset.append({
            "input_ids": tokenized["input_ids"][i],
            "attention_mask": tokenized["attention_mask"][i],
            "labels": torch.tensor(labels[i]),
            "raw_sequence": sequences[i]    # <-- store the original sequence
        })
    return dataset



# ================= Model, Data, and Training Arguments =====================
@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    use_lora: bool = field(default=False, metadata={"help": "whether to use LoRA"})
    lora_r: int = field(default=8, metadata={"help": "hidden dimension for LoRA"})
    lora_alpha: int = field(default=32, metadata={"help": "alpha for LoRA"})
    lora_dropout: float = field(default=0.05, metadata={"help": "dropout rate for LoRA"})
    lora_target_modules: str = field(default="query,value", metadata={"help": "where to perform LoRA"})


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    kmer: int = field(default=-1, metadata={"help": "k-mer for input sequence. -1 means not using k-mer."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    run_name: str = field(default="run")
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(default=512, metadata={"help": "Maximum sequence length."})
    per_device_train_batch_size: int = field(default=1)
    per_device_eval_batch_size: int = field(default=1)
    num_train_epochs: int = field(default=1)
    output_dir: str = field(default="output")
    seed: int = field(default=42)
    save_total_limit: int = field(default=3, metadata={"help": "Limit the total checkpoints to keep. Deletes older ones."})
    load_best_model_at_end: bool = field(default=True, metadata={"help": "Load the best model found during training at the end."})
    max_learning_rate: float = field(default=5e-5, metadata={"help": "Maximum learning rate for cyclic schedule."})  # Added a max learning rate 

# =================== Helper Functions =====================
def generate_kmer_str(sequence: str, k: int) -> str:
    """Generate k-mer string from DNA sequence."""
    return " ".join([sequence[i:i+k] for i in range(len(sequence) - k + 1)])


def load_or_generate_kmer(data_path: str, texts: List[str], k: int) -> List[str]:
    """Load or generate k-mer string for each DNA sequence."""
    data_path = data_path + 'merged.csv'
    kmer_path = data_path.replace(".csv", f"_{k}mer.json")
    if os.path.exists(kmer_path):
        logging.warning(f"Loading k-mer from {kmer_path}...")
        with open(kmer_path, "r") as f:
            kmer = json.load(f)
    else:        
        logging.warning(f"Generating k-mer...")
        kmer = [generate_kmer_str(text, k) for text in texts]
        with open(kmer_path, "w") as f:
            logging.warning(f"Saving k-mer to {kmer_path}...")
            json.dump(kmer, f)
    return kmer




@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.Tensor(labels).long()
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )




# def calculate_metric_with_sklearn(predictions: np.ndarray, labels: np.ndarray):
#     # Ensure predictions are 1D (i.e., class predictions)
#     if predictions.ndim > 1:  # In case of logits (multi-dimensional array)
#         predictions = np.argmax(predictions, axis=-1)  # Get the class with the highest score
    
#     valid_mask = labels != -100  # Mask for valid labels (if any)
#     valid_predictions = predictions[valid_mask]  # Use the valid mask to filter predictions
#     valid_labels = labels[valid_mask]  # Use the same valid mask for labels
    
#     # Now calculate the metrics
#     return {
#         "accuracy": sklearn.metrics.accuracy_score(valid_labels, valid_predictions),
#         "f1": sklearn.metrics.f1_score(valid_labels, valid_predictions, average="macro"),
#         "matthews_correlation": sklearn.metrics.matthews_corrcoef(valid_labels, valid_predictions),
#         "precision": sklearn.metrics.precision_score(valid_labels, valid_predictions, average="macro"),
#         "recall": sklearn.metrics.recall_score(valid_labels, valid_predictions, average="macro"),
#     }

def calculate_metric_with_sklearn(eval_pred: EvalPrediction):
    predictions, labels = eval_pred  # Extract predictions and labels
    
    if predictions.ndim > 1:  # If predictions are logits (multidimensional)
        predictions = np.argmax(predictions, axis=-1)  # Convert logits to predicted class indices
    
    # Mask to handle invalid labels (e.g., -100 padding labels)
    valid_mask = labels != -100
    valid_predictions = predictions[valid_mask]  # Apply the mask to predictions
    valid_labels = labels[valid_mask]  # Apply the mask to labels
    
    return {
        "accuracy": accuracy_score(valid_labels, valid_predictions),
        "f1": f1_score(valid_labels, valid_predictions, average="macro"),
        "matthews_correlation": matthews_corrcoef(valid_labels, valid_predictions),
        "precision": precision_score(valid_labels, valid_predictions, average="macro"),
        "recall": recall_score(valid_labels, valid_predictions, average="macro"),
        "auc": roc_auc_score(valid_labels, valid_predictions, average="macro"),
        
    }
    
#from: https://discuss.huggingface.co/t/cuda-out-of-memory-when-using-trainer-with-compute-metrics/2941/13
def preprocess_logits_for_metrics(logits:Union[torch.Tensor, Tuple[torch.Tensor, Any]], _):
    if isinstance(logits, tuple):  # Unpack logits if it's a tuple
        logits = logits[0]

    if logits.ndim == 3:
        # Reshape logits to 2D if needed
        logits = logits.reshape(-1, logits.shape[-1])

    return torch.argmax(logits, dim=-1)


"""
Compute metrics used for huggingface trainer.
""" 
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    return calculate_metric_with_sklearn(predictions, labels)

def save_results(trainer,model_output_dir,val_subset, test_dataset):
    # Evaluate on validation set
        print("Evaluating on validation set...")
        val_metrics = trainer.evaluate(eval_dataset=val_subset)
        with open(os.path.join(model_output_dir, "val_results.json"), "w") as f:
            json.dump(val_metrics, f, indent=4)
        
        # Save predictions for validation set to JSON
        val_predictions = trainer.predict(val_subset)
        test_embeddings = val_predictions.predictions  # Extract prediction embeddings
        val_json_path = os.path.join(model_output_dir, "val_predictions.json")
        with open(val_json_path, "w") as f:
            json.dump({
                "predictions": val_predictions.predictions.tolist(),
                "labels": val_predictions.label_ids.tolist(),
                # "input_sequences": [val_subset[i]["input_ids"].tolist() for i in range(len(test_dataset))],
                # "sequences": [val_subset[i]["raw_sequence"] for i in range(len(test_dataset))],
                # "embeddings": test_embeddings.tolist()  # Save embeddings
            }, f, indent=4)

        # Convert validation predictions JSON to CSV
        val_csv_path = os.path.join(model_output_dir, "val_predictions.csv")
        with open(val_json_path, "r") as json_file:
            val_data = json.load(json_file)
            with open(val_csv_path, mode="w", newline="") as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(["Prediction", "Label"])
                for pred, label in zip(val_data["predictions"], val_data["labels"]):
                    writer.writerow([pred, label])
        
        # Evaluate on test set
        print("Evaluating on test set...")
        test_metrics = trainer.evaluate(eval_dataset=test_dataset)
        with open(os.path.join(model_output_dir, "test_results.json"), "w") as f:
            json.dump(test_metrics, f, indent=4)
        
        # Save predictions and embeddings for test set to JSON
        test_predictions = trainer.predict(test_dataset)
        test_embeddings = test_predictions.predictions  # Extract prediction embeddings
        test_json_path = os.path.join(model_output_dir, "test_predictions.json")
        with open(test_json_path, "w") as f:
            json.dump({
                "predictions": test_predictions.predictions.tolist(),
                "labels": test_predictions.label_ids.tolist(),
                # "input_sequences": [test_dataset[i]["input_ids"].tolist() for i in range(len(test_dataset))],
                "sequences": [test_dataset[i]["raw_sequence"] for i in range(len(test_dataset))],
                "embeddings": test_embeddings.tolist()  # Save embeddings
            }, f, indent=4)

        # Convert test predictions JSON to CSV
        test_csv_path = os.path.join(model_output_dir, "test_predictions.csv")
        with open(test_json_path, "r") as json_file:
            test_data = json.load(json_file)
            with open(test_csv_path, mode="w", newline="") as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(["Prediction", "Label","Raw Sequence", "Embedding"])
                
                for pred, label, seq, embedding in zip(test_data["predictions"], test_data["labels"], test_data["sequences"], test_data["embeddings"]):
                    print(f"embedding: {embedding}, type: {type(embedding)}")  # Debugging
                    embedding_str = " ".join(map(str, embedding)) if isinstance(embedding, (list, np.ndarray, torch.Tensor)) else str(embedding)
                    writer.writerow([pred, label, seq, embedding_str])
        
        

# ===================== Training Logic ======================
def k_fold_cross_validation(merged_dataset,tokenizer, model_args, data_args, training_args, k=5):
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=training_args.seed)
    # Extract labels for stratification
    # Extract raw sequences and labels separately from merged dataset
    # Extract raw sequences and labels based on the specified column indices (4 and 6)
    raw_sequences = [item["raw_sequence"] for item in merged_dataset]  # Column index 4 for sequences
    labels = np.array([int(item["labels"]) for item in merged_dataset])  # Column index 6 for labels
    
    def calculate_warmup_steps(total_examples, batch_size, num_epochs, warmup_ratio=0.1):
        total_steps = (total_examples * num_epochs) // batch_size
        warmup_steps = int(total_steps * warmup_ratio)
        return warmup_steps
    
    fold = 1  # Initialize fold counter
    
    for train_val_idx, test_idx in skf.split(raw_sequences, labels):
        print(f"Starting fold {fold}...")
        
        # === Changed Part: Define the checkpoint directory and check for existing checkpoints ===
        fold_checkpoint_dir = os.path.join(training_args.output_dir, f"fold_{fold}")
        last_checkpoint = None

        # Check if a checkpoint exists
        if os.path.exists(fold_checkpoint_dir):
            checkpoints = [os.path.join(fold_checkpoint_dir, ckpt) for ckpt in os.listdir(fold_checkpoint_dir)]
            checkpoints = [ckpt for ckpt in checkpoints if os.path.isdir(ckpt)]
            if checkpoints:
                last_checkpoint = sorted(checkpoints)[-1]  # Get the latest checkpoint
                print(f"Resuming from checkpoint: {last_checkpoint}")
        # === End of Changed Part ===
        
        train_idx, val_idx = train_test_split(
        train_val_idx, 
        test_size=0.25,  # 25% of train_val (which is 80% of total) = 20% of total
        stratify=labels[train_val_idx],  # Maintain class balance in train/val
        random_state=training_args.seed
        )

        # Extract the respective splits using the indices
        train_sequences = [raw_sequences[i] for i in train_idx]
        train_labels = [labels[i] for i in train_idx]

        val_sequences = [raw_sequences[i] for i in val_idx]
        val_labels = [labels[i] for i in val_idx]

        test_sequences = [raw_sequences[i] for i in test_idx]
        test_labels = [labels[i] for i in test_idx]

        fold_output_dir = os.path.join(training_args.output_dir, f"fold_{fold}")
        os.makedirs(fold_output_dir, exist_ok=True)

        save_raw_data_to_csv(train_sequences, train_labels, os.path.join(fold_output_dir, "train.csv"))
        save_raw_data_to_csv(val_sequences, val_labels, os.path.join(fold_output_dir, "val.csv"))
        save_raw_data_to_csv(test_sequences, test_labels, os.path.join(fold_output_dir, "test.csv"))


        print(f"Fold {fold}: Train={len(train_sequences)}, Val={len(val_sequences)}, Test={len(test_sequences)}")

        # Perform tokenization after splitting
        train_dataset = tokenize_sequences(train_sequences, train_labels, tokenizer, data_args.data_path, data_args.kmer)
        val_dataset = tokenize_sequences(val_sequences, val_labels, tokenizer, data_args.data_path, data_args.kmer)
        test_dataset = tokenize_sequences(test_sequences, test_labels, tokenizer, data_args.data_path, data_args.kmer)

        # print(train_subset[2])
        # print("Got here!!!")
        # load model
        model = transformers.AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        num_labels=2,
        trust_remote_code=True,
    )

        if model_args.use_lora:
            lora_config = LoraConfig(
                r=model_args.lora_r, lora_alpha=model_args.lora_alpha, 
                target_modules=list(model_args.lora_target_modules.split(",")), lora_dropout=model_args.lora_dropout,
                task_type="SEQ_CLS",
            )
            model = get_peft_model(model, lora_config)
            
        # Define optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=training_args.learning_rate)

        # === Changed Part: Scheduler Integration ===
        total_examples = len(train_sequences)
        num_epochs = training_args.num_train_epochs
        batch_size = training_args.per_device_train_batch_size

        # Calculate warm-up steps
        warmup_steps = calculate_warmup_steps(total_examples, batch_size, num_epochs, warmup_ratio=0.1)

        # Define scheduler
        scheduler = WarmUpThenCyclicScheduler(
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            base_lr=training_args.learning_rate,
            max_lr=training_args.max_learning_rate,
            step_size_up=500,  # Cyclic step size (can be tuned)
            step_size_down=500,
        )
        # === End of Changed Part ===
        
        # 3. Create the custom scheduler callback
        scheduler_callback = CustomSchedulerCallback(scheduler)
        
        early_stopping_callback = transformers.EarlyStoppingCallback(early_stopping_patience=6)
        trainer = transformers.Trainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=calculate_metric_with_sklearn,
            data_collator=DataCollatorForSupervisedDataset(tokenizer=tokenizer),
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            callbacks=[early_stopping_callback, scheduler_callback]
        )
        os.makedirs(fold_checkpoint_dir, exist_ok=True)
        # === Changed Part: Resume training if checkpoint exists ===
        trainer.train(resume_from_checkpoint=last_checkpoint)
        # === End of Changed Part ===
        # Save model and results
        trainer.save_model(fold_checkpoint_dir)
        save_results(trainer, fold_checkpoint_dir, val_dataset, test_dataset)
        print(f"Fold {fold} completed. Results saved to {fold_checkpoint_dir}")
        fold += 1

def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # load tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
        trust_remote_code=True,
    )
    
    # Load the merged dataset (raw sequences, not tokenized)
    merged_dataset = load_merged_dataset(os.path.join(data_args.data_path, "merged.csv"))

    # print(f"Sample data: {train_dataset[0]}")  # Print the first data item
    # exit()
    k_fold_cross_validation(merged_dataset, tokenizer, model_args, data_args, training_args)

if __name__ == "__main__":
    train()
