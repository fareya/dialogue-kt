# Required Imports
from huggingface_hub import login
from collections import defaultdict
import torch
from torch.utils.data import Dataset, DataLoader
import json
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification, BitsAndBytesConfig
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from tqdm import tqdm

# Constants and Global Variables
MATHDIAL_DIALOGUE_DESC = "the student is attempting to solve a math problem."
DATA_PATH = '/work/pi_juanzhai_umass_edu/fareya_workspace/dialogue-kt/teacher_moves/processed_data/train.jsonl'
# MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
ACCESS_TOKEN = "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"
SYSTEM_PROMPT_TEMPLATE = (
    "You are an experienced math teacher. You are given a dialogue between a student and teacher where {desc} "
    "Your job is to predict the next teacher move. There are four teacher moves: generic, focus, telling, probing. "
    "The dialogue is as follows:"
)

# Initialize Tokenizer
def initialize_tokenizer(model_name, access_token):
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=access_token, )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

# Read JSONL File
def read_jsonl(data_path):
    with open(data_path) as f:
        return [json.loads(line) for line in f]

# Extract Sequences and Labels
def extract_sequences_labels(data_path, tokenizer, system_prompt):
    data = read_jsonl(data_path)
    grouped_data = defaultdict(list)
    
    for entry in data:
        grouped_data[entry["id"]].append(entry)
    
    sequences, labels = [], []
    for qid, turns in grouped_data.items():
        turns = sorted(turns, key=lambda x: x["turn"])
        for i, current_turn in enumerate(turns):
            for context_size in range(0, 5):
                if i >= context_size:
                    context_turns = turns[i - context_size:i]
                    context = " ".join(
                        f"[TEACHER] {turn['teacher_move'].strip()} [STUDENT] {turn['student_move'].strip()}"
                        for turn in context_turns
                    )
                    current = f"[TEACHER] {current_turn['teacher_move'].strip()} [STUDENT] {current_turn['student_move'].strip()}"
                    sequence = f"{context} {current}"
                    label = current_turn["future_teacher_move_type"].strip()
                    prompt = tokenizer.apply_chat_template([
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": sequence},
                    ], tokenize=False)
#ValueError: Cannot use chat template functions because tokenizer.chat_template is not set and no template argument was passed! For information about writing templates and setting the tokenizer.chat_template attribute, please see the documentation at https://huggingface.co/docs/transformers/main/en/chat_templating

                    sequences.append(sequence)
                    labels.append(label)
    return sequences, labels

# Dataset Class
class DialogueDataset(Dataset):
    def __init__(self, sequences, labels, tokenizer, max_length=512):
        self.sequences = sequences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            sequence,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": label
        }


# Initialize model for multi-class classification
def initialize_model(model_ckpt, num_labels, config):
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=4,
        problem_type="single_label_classification",  # Multi-class classification
        config=config,
    )
    return model
# Training Loop
def train(model, dataloader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0
    
    for batch in tqdm(dataloader, desc="Training"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)  # Multi-class uses integer labels
        
        optimizer.zero_grad()
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        logits = outputs.logits  # Shape: (batch_size, num_labels)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss

# Validation Loop
def validate(model, dataloader, loss_fn, device):
    model.eval()
    total_loss = 0
    correct_predictions = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            logits = outputs.logits
            loss = loss_fn(logits, labels)
            total_loss += loss.item()
            
            predictions = torch.argmax(logits, dim=-1)  # Get class predictions
            correct_predictions += (predictions == labels).sum().item()
            total_samples += labels.size(0)
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct_predictions / total_samples
    return avg_loss, accuracy


# Main Function
# def main():
#     tokenizer = initialize_tokenizer(MODEL_NAME, ACCESS_TOKEN)
#     system_prompt = SYSTEM_PROMPT_TEMPLATE.format(desc=MATHDIAL_DIALOGUE_DESC)
#     sequences, labels = extract_sequences_labels(DATA_PATH, tokenizer, system_prompt)
#     print(len(sequences), len(labels))
#     print(sequences[:4], labels[:4])
#     dataset = DialogueDataset(sequences, labels, tokenizer, max_length=512)
#     dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
#     for batch in dataloader:
#         print(batch)
#         break

# # Entry Point
# if __name__ == "__main__":
#     main()


def main():
    # Initialize tokenizer
    login(token=ACCESS_TOKEN)
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    print(quantization_config)
    tokenizer = initialize_tokenizer(MODEL_NAME, ACCESS_TOKEN)
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(desc=MATHDIAL_DIALOGUE_DESC)
    sequences, labels = extract_sequences_labels(DATA_PATH, tokenizer, system_prompt)

    # Map labels to integer indices for multi-class classification
    unique_labels = list(set(labels))
    num_labels = len(unique_labels)
    label_map = {label: i for i, label in enumerate(unique_labels)}
    labels = [label_map[label] for label in labels]  # Convert to integer indices

    # Split data into training and validation sets
    split_idx = int(0.8 * len(sequences))
    train_sequences, val_sequences = sequences[:split_idx], sequences[split_idx:]
    train_labels, val_labels = labels[:split_idx], labels[split_idx:]
    
    train_dataset = DialogueDataset(train_sequences, train_labels, tokenizer, max_length=512)
    val_dataset = DialogueDataset(val_sequences, val_labels, tokenizer, max_length=512)
    
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    # print checks 
    print(len(train_sequences), len(val_sequences))
    print(len(train_labels), len(val_labels))
    print(len(train_dataset), len(val_dataset))
    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model = initialize_model(MODEL_NAME, num_labels, quantization_config).to(device)
    
    # Define optimizer and loss function
    optimizer = AdamW(model.parameters(), lr=5e-5)
    loss_fn = CrossEntropyLoss()
    
    # Training and validation loop
    # epochs = 10
    # for epoch in range(epochs):
    #     print(f"Epoch {epoch + 1}/{epochs}")
        
    #     train_loss = train(model, train_dataloader, optimizer, loss_fn, device)
    #     print(f"Training Loss: {train_loss:.4f}")
        
    #     val_loss, val_accuracy = validate(model, val_dataloader, loss_fn, device)
    #     print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

if __name__ == "__main__":
     main()
print("done")