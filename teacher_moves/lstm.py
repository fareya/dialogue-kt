# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import Dataset, DataLoader
# from sklearn.model_selection import train_test_split
# import json
# from collections import defaultdict
# from itertools import product
# import random

# RANDOM_SEED = 42
# torch.manual_seed(RANDOM_SEED)
# random.seed(RANDOM_SEED)

# ### ======= Data Processing ======= ###
# def read_jsonl(data_path):
#     with open(data_path) as f:
#         return [json.loads(line) for line in f]

# def group_data_by_id(data):
#     grouped_data = defaultdict(list)
#     for entry in data:
#         grouped_data[entry["id"]].append(entry)
#     return grouped_data

# def split_train_val(grouped_data, test_size=0.2):
#     train_data, val_data = train_test_split(list(grouped_data.values()), test_size=test_size, random_state=RANDOM_SEED)
#     return train_data, val_data

# def get_test_formatted(grouped_data):
#     test_data = list(grouped_data.values())
#     return test_data

# def format_teacher_moves_only(data, window_size, pred_label_name="teacher_move_type"):
#     sequences, labels = [], []
#     for conversation in data:
#         teacher_move_types = [turn[pred_label_name] for turn in conversation if turn.get(pred_label_name)]
#         for i in range(len(teacher_move_types) - window_size):
#             input_seq = teacher_move_types[i:i+window_size]
#             label = teacher_move_types[i+window_size]
#             sequences.append(input_seq)
#             labels.append(label)
#     return sequences, labels

# def encode_teacher_moves(sequences, labels, move2idx=None):
#     if move2idx is None:
#         all_moves = sorted(set([move for seq in sequences for move in seq] + labels))
#         move2idx = {move: idx for idx, move in enumerate(all_moves)}
    
#     encoded_X = torch.tensor([[move2idx[move] for move in seq] for seq in sequences])
#     encoded_y = torch.tensor([move2idx[label] for label in labels])
    
#     return encoded_X, encoded_y, move2idx

# ### ======= Dataset and Dataloader ======= ###
# class TeacherMoveDataset(Dataset):
#     def __init__(self, X, y):
#         self.X = X
#         self.y = y

#     def __len__(self):
#         return len(self.X)

#     def __getitem__(self, idx):
#         return self.X[idx], self.y[idx]

# def create_dataloader(X, y, batch_size=32, shuffle=True):
#     dataset = TeacherMoveDataset(X, y)
#     return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

# ### ======= LSTM Model ======= ###
# class LSTMModel(nn.Module):
#     def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, num_layers, dropout):
#         super(LSTMModel, self).__init__()
#         self.embedding = nn.Embedding(vocab_size, embedding_dim)
#         self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout)
#         self.fc = nn.Linear(hidden_dim, output_dim)
#         self.dropout = nn.Dropout(dropout)

#     def forward(self, x):
#         embedded = self.embedding(x)
#         lstm_out, _ = self.lstm(embedded)
#         out = self.fc(lstm_out[:, -1, :])
#         return out

# ### ======= Training & Evaluation ======= ###
# def train_model(model, train_loader, val_loader, criterion, optimizer, epochs, device):
#     for epoch in range(epochs):
#         model.train()
#         total_loss = 0
#         for X_batch, y_batch in train_loader:
#             X_batch, y_batch = X_batch.to(device), y_batch.to(device)
#             optimizer.zero_grad()
#             output = model(X_batch)
#             loss = criterion(output, y_batch)
#             loss.backward()
#             optimizer.step()
#             total_loss += loss.item()
        
#         model.eval()
#         correct, total = 0, 0
#         with torch.no_grad():
#             for X_batch, y_batch in val_loader:
#                 X_batch, y_batch = X_batch.to(device), y_batch.to(device)
#                 output = model(X_batch)
#                 predictions = torch.argmax(output, dim=1)
#                 correct += (predictions == y_batch).sum().item()
#                 total += y_batch.size(0)
        
#         print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}, Val Acc: {correct/total:.4f}")

# def evaluate_model(model, test_loader, device):
#     model.eval()
#     correct, total = 0, 0
#     with torch.no_grad():
#         for X_batch, y_batch in test_loader:
#             X_batch, y_batch = X_batch.to(device), y_batch.to(device)
#             output = model(X_batch)
#             predictions = torch.argmax(output, dim=1)
#             correct += (predictions == y_batch).sum().item()
#             total += y_batch.size(0)
#     print(f"Test Accuracy: {correct/total:.4f}")

# ### ======= Main ======= ###
# if __name__ == "__main__":
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Using device: {device}")

#     # Load data
#     train_path = "/work/pi_andrewlan_umass_edu/fikram_umass-edu/dialogue-kt/teacher_moves/processed_data/train_check4.jsonl"
#     test_path = "/work/pi_andrewlan_umass_edu/fikram_umass-edu/dialogue-kt/teacher_moves/processed_data/test_check4.jsonl"

#     train_data_raw = read_jsonl(train_path)
#     test_data_raw = read_jsonl(test_path)
    
#     grouped_train = group_data_by_id(train_data_raw)
#     grouped_test = get_test_formatted(group_data_by_id(test_data_raw))

#     train_data, val_data = split_train_val(grouped_train)
#     window_size = 3

#     # Train/Val
#     sequences_train, labels_train = format_teacher_moves_only(train_data, window_size)
#     X_train, y_train, move2idx = encode_teacher_moves(sequences_train, labels_train)

#     sequences_val, labels_val = format_teacher_moves_only(val_data, window_size)
#     X_val, y_val, _ = encode_teacher_moves(sequences_val, labels_val, move2idx)

#     train_loader = create_dataloader(X_train, y_train, 32)
#     val_loader = create_dataloader(X_val, y_val, 32)

#     # Sanity check
#     for i in range(5):
#         print(f"Train Input: {sequences_train[i]} → Label: {labels_train[i]}")

#     # Model
#     model = LSTMModel(
#         vocab_size=len(move2idx),
#         embedding_dim=64,
#         hidden_dim=128,
#         output_dim=len(move2idx),
#         num_layers=1,
#         dropout=0.3
#     ).to(device)

#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(model.parameters(), lr=0.001)

#     train_model(model, train_loader, val_loader, criterion, optimizer, epochs=5, device=device)

#     # Test
#     sequences_test, labels_test = format_teacher_moves_only(grouped_test, window_size)
#     X_test, y_test, _ = encode_teacher_moves(sequences_test, labels_test, move2idx)
#     test_loader = create_dataloader(X_test, y_test, 32, shuffle=False)

#     print("\nEvaluating on Test Set:")
#     evaluate_model(model, test_loader, device)



# use this 
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import Dataset, DataLoader
# from sklearn.model_selection import train_test_split
# import json
# from collections import defaultdict
# import random

# RANDOM_SEED = 42
# torch.manual_seed(RANDOM_SEED)
# random.seed(RANDOM_SEED)

# ### ======= Configuration ======= ###
# window_size = 3
# INPUT_KEY = "teacher_move_type"
# LABEL_KEY = "correctness_annotation"  # options: "teacher_move_type", "correctness_annotation"
# LABEL_MODE = "current"  # options: "future", "current"

# print("Input Key:", INPUT_KEY)
# print("Label Key:", LABEL_KEY)
# print("Label Key:", LABEL_MODE)

# ### ======= Data Processing ======= ###
# def read_jsonl(data_path):
#     with open(data_path) as f:
#         return [json.loads(line) for line in f]

# def group_data_by_id(data):
#     grouped_data = defaultdict(list)
#     for entry in data:
#         grouped_data[entry["id"]].append(entry)
#     return grouped_data

# def split_train_val(grouped_data, test_size=0.2):
#     train_data, val_data = train_test_split(list(grouped_data.values()), test_size=test_size, random_state=RANDOM_SEED)
#     return train_data, val_data

# def get_test_formatted(grouped_data):
#     return list(grouped_data.values())

# def format_sequences_with_labels(data, window_size, input_key, label_key, label_mode):
#     assert label_mode in ["current", "future"], "label_mode must be 'current' or 'future'"
#     sequences, labels = [], []
#     for conversation in data:
#         input_seq = [turn[input_key] for turn in conversation if input_key in turn and label_key in turn]
#         label_seq = [turn[label_key] for turn in conversation if input_key in turn and label_key in turn]

#         for i in range(len(input_seq) - window_size):
#             window = input_seq[i:i+window_size]
#             if label_mode == "future":
#                 label_idx = i + window_size
#             else:
#                 label_idx = i + window_size - 1

#             if label_idx < len(label_seq):
#                 label = label_seq[label_idx]
#                 if label is not None:
#                     sequences.append(window)
#                     labels.append(label)
#     return sequences, labels

# def encode_teacher_moves(sequences, labels, move2idx=None):
#     if move2idx is None:
#         all_labels = sorted(set([move for seq in sequences for move in seq] + labels))
#         move2idx = {move: idx for idx, move in enumerate(all_labels)}
    
#     encoded_X = torch.tensor([[move2idx[move] for move in seq] for seq in sequences])
#     encoded_y = torch.tensor([move2idx[label] for label in labels])
    
#     return encoded_X, encoded_y, move2idx

# ### ======= Dataset and Dataloader ======= ###
# class TeacherMoveDataset(Dataset):
#     def __init__(self, X, y):
#         self.X = X
#         self.y = y

#     def __len__(self):
#         return len(self.X)

#     def __getitem__(self, idx):
#         return self.X[idx], self.y[idx]

# def create_dataloader(X, y, batch_size=32, shuffle=True):
#     dataset = TeacherMoveDataset(X, y)
#     return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

# ### ======= LSTM Model ======= ###
# class LSTMModel(nn.Module):
#     def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, num_layers, dropout):
#         super(LSTMModel, self).__init__()
#         self.embedding = nn.Embedding(vocab_size, embedding_dim)
#         self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout)
#         self.fc = nn.Linear(hidden_dim, output_dim)
#         self.dropout = nn.Dropout(dropout)

#     def forward(self, x):
#         embedded = self.embedding(x)
#         lstm_out, _ = self.lstm(embedded)
#         out = self.fc(lstm_out[:, -1, :])
#         return out

# ### ======= Training & Evaluation ======= ###
# def train_model(model, train_loader, val_loader, criterion, optimizer, epochs, device):
#     for epoch in range(epochs):
#         model.train()
#         total_loss = 0
#         for X_batch, y_batch in train_loader:
#             X_batch, y_batch = X_batch.to(device), y_batch.to(device)
#             optimizer.zero_grad()
#             output = model(X_batch)
#             loss = criterion(output, y_batch)
#             loss.backward()
#             optimizer.step()
#             total_loss += loss.item()
        
#         model.eval()
#         correct, total = 0, 0
#         with torch.no_grad():
#             for X_batch, y_batch in val_loader:
#                 X_batch, y_batch = X_batch.to(device), y_batch.to(device)
#                 output = model(X_batch)
#                 predictions = torch.argmax(output, dim=1)
#                 correct += (predictions == y_batch).sum().item()
#                 total += y_batch.size(0)
        
#         print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}, Val Acc: {correct/total:.4f}")

# def evaluate_model(model, test_loader, device):
#     model.eval()
#     correct, total = 0, 0
#     with torch.no_grad():
#         for X_batch, y_batch in test_loader:
#             X_batch, y_batch = X_batch.to(device), y_batch.to(device)
#             output = model(X_batch)
#             predictions = torch.argmax(output, dim=1)
#             correct += (predictions == y_batch).sum().item()
#             total += y_batch.size(0)
#     print(f"Test Accuracy: {correct/total:.4f}")

# ### ======= Main ======= ###
# if __name__ == "__main__":
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Using device: {device}")

#     train_path = "/work/pi_andrewlan_umass_edu/fikram_umass-edu/dialogue-kt/teacher_moves/processed_data/train_check4.jsonl"
#     test_path = "/work/pi_andrewlan_umass_edu/fikram_umass-edu/dialogue-kt/teacher_moves/processed_data/test_check4.jsonl"

#     train_data_raw = read_jsonl(train_path)
#     test_data_raw = read_jsonl(test_path)

#     grouped_train = group_data_by_id(train_data_raw)
#     grouped_test = get_test_formatted(group_data_by_id(test_data_raw))

#     train_data, val_data = split_train_val(grouped_train)

#     # Train/Val
#     sequences_train, labels_train = format_sequences_with_labels(
#         train_data, window_size, INPUT_KEY, LABEL_KEY, LABEL_MODE
#     )
#     X_train, y_train, label2idx = encode_teacher_moves(sequences_train, labels_train)

#     sequences_val, labels_val = format_sequences_with_labels(
#         val_data, window_size, INPUT_KEY, LABEL_KEY, LABEL_MODE
#     )
#     X_val, y_val, _ = encode_teacher_moves(sequences_val, labels_val, label2idx)

#     train_loader = create_dataloader(X_train, y_train)
#     val_loader = create_dataloader(X_val, y_val)

#     print(f"\nSanity check (predicting {LABEL_MODE} {LABEL_KEY}):")
#     for i in range(5):
#         print(f"Input: {sequences_train[i]} → Label: {labels_train[i]}")

#     model = LSTMModel(
#         vocab_size=len(label2idx),
#         embedding_dim=64,
#         hidden_dim=128,
#         output_dim=len(label2idx),
#         num_layers=1,
#         dropout=0.3
#     ).to(device)

#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(model.parameters(), lr=0.001)

#     train_model(model, train_loader, val_loader, criterion, optimizer, epochs=5, device=device)

#     # Test
#     sequences_test, labels_test = format_sequences_with_labels(
#         grouped_test, window_size, INPUT_KEY, LABEL_KEY, LABEL_MODE
#     )
#     X_test, y_test, _ = encode_teacher_moves(sequences_test, labels_test, label2idx)
#     test_loader = create_dataloader(X_test, y_test, shuffle=False)

#     print("\nEvaluating on Test Set:")
#     evaluate_model(model, test_loader, device)



import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
import json
from collections import defaultdict
import random
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

MOVE_LIST = ['focus', 'telling', 'probing', 'generic']
MOVE_TO_INDEX = {label: i for i, label in enumerate(MOVE_LIST)}
CORR_LABEL_LIST = ['false', 'true']
CORR_LABEL_TO_INDEX = {label: i for i, label in enumerate(CORR_LABEL_LIST)}

### ======= Configuration ======= ###
INPUT_KEY = "teacher_move_type"
# LABEL_KEY = "final_correctness"  # options: "teacher_move_type", "future_teacher_move_type", "correctness_annotation", "final_correctness"
# LABEL_KEY = "teacher_move_type"
# LABEL_KEY = "future_teacher_move_type" 
LABEL_KEY = "correctness_annotation" 

### ======= Helper Functions ======= ###

def true_positive_true_negative_2(y_true, y_pred, test_data):
    from collections import Counter

    class_names = ['generic', 'focus', 'telling', 'probing']
    tp = {cls: 0 for cls in class_names}
    fn = {cls: 0 for cls in class_names}
    fp = {cls: 0 for cls in class_names}
    
    # Store misclassified samples (limited to 5 per class)
    misclassified_examples = {cls: [] for cls in class_names}

    for i in range(len(y_true)):
        try:
            assert y_pred[i] in class_names
            assert y_true[i] in class_names
        except:
            print(f"y_pred[i]: {y_pred[i]}")
            print(f"y_true[i]: {y_true[i]}")
            continue
        if y_true[i] == y_pred[i]:
            tp[y_true[i]] += 1
        else:
            fn[y_true[i]] += 1
            fp[y_pred[i]] += 1
            
            # Store misclassified examples (limit to 5 per class)
            if len(misclassified_examples[y_true[i]]) < 5:
                misclassified_examples[y_true[i]].append({
                    "true_label": y_true[i],
                    "predicted_label": y_pred[i],
                })

    # Analyze distributions of y_true and y_pred
    y_true_distribution = Counter(y_true)
    y_pred_distribution = Counter(y_pred)

    print("\nDistribution of y_true:")
    for cls, count in y_true_distribution.items():
        print(f"{cls}: {count}")

    print("\nDistribution of y_pred:")
    for cls, count in y_pred_distribution.items():
        print(f"{cls}: {count}")

    # Identify most common misclassifications
    misclassification_counts = Counter(
        (true_label, pred_label)
        for true_label, pred_label in zip(y_true, y_pred)
        if true_label != pred_label
    )
    print("\nMost Common Misclassifications:")
    for (true_label, pred_label), count in misclassification_counts.most_common(5):
        print(f"True: {true_label}, Predicted: {pred_label}, Count: {count}")

    # Print results
    print("\nTrue Positives: ", tp)
    print("False Negatives: ", fn)
    print("False Positives: ", fp)

def save_results_to_jsonl(results, output_file):
    """
    Save results to a JSONL file. Appends to the file if it already exists.

    Args:
        results (dict): The results to save.
        output_file (str): The file path to save the results.
    """
    with open(output_file, 'a') as f:
        f.write(json.dumps(results) + '\n')
    print(f"Results appended to {output_file}")

### ======= Data Processing ======= ###
def read_jsonl(data_path):
    with open(data_path) as f:
        return [json.loads(line) for line in f]

def group_data_by_id(data):
    grouped_data = defaultdict(list)
    for entry in data:
        grouped_data[entry["id"]].append(entry)
    return list(grouped_data.values())

def split_train_val(data, test_size=0.2):
    train_data, val_data = train_test_split(data, test_size=test_size, random_state=RANDOM_SEED)
    return train_data, val_data

def format_sequences_with_labels(data, input_key, label_key):
    sequences, labels = [], []
    for conversation in data:
        input_seq, label_seq = [], []
        for turn in conversation:
            if input_key in turn and label_key in turn:
                print(turn[label_key])
                input_seq.append(turn[input_key])
                label_seq.append(turn[label_key])
        sequences.append(input_seq)
        labels.append(label_seq)
    return sequences, labels

def encode_teacher_moves(sequences, labels, label_map):
    encoded_X = [
        torch.stack([F.one_hot(torch.tensor(MOVE_TO_INDEX[move]), len(MOVE_TO_INDEX)).type(torch.float32) for move in seq])
        for seq in sequences
    ]
    # if this is all zeros we can make this a vector 
    encoded_y = [
        torch.tensor([label_map.get(label, -100) for label in seq_labels])
        for seq_labels in labels
    ]
    return encoded_X, encoded_y

### ======= Dataset and Dataloader ======= ###
class TeacherMoveDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class LSTMCollator:
    def __call__(self, batch):
        X_batch = pad_sequence([X for X, _ in batch], batch_first=True, padding_value=0)
        y_batch = pad_sequence([y for _, y in batch], batch_first=True, padding_value=-100)
        return X_batch, y_batch

def create_dataloader(X, y, batch_size, shuffle):
    dataset = TeacherMoveDataset(X, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=LSTMCollator())

### ======= LSTM Model ======= ###
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = self.dropout(lstm_out)
        out = self.fc(lstm_out)
        return out

### ======= Training & Evaluation ======= ###
def train_model(model: nn.Module, train_loader, val_loader, binary, lr, epochs, device):
    w_p = torch.FloatTensor([3]).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    if binary:
        criterion = nn.BCEWithLogitsLoss(pos_weight = w_p)
    else:
        criterion = nn.CrossEntropyLoss()
    best_val_loss = None
    best_model_sd = None
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            output = model(X_batch)
            if binary:
                mask = y_batch != -100
                loss = criterion(output[mask].view(-1), y_batch[mask].view(-1))
            else:
                loss = criterion(output.view(-1, output.shape[-1]), y_batch.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                output = model(X_batch)
                if binary:
                    mask = y_batch != -100
                    loss = criterion(output[mask].view(-1), y_batch[mask].view(-1))
                else:
                    loss = criterion(output.view(-1, output.shape[-1]), y_batch.view(-1))
                val_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}")

        if not best_val_loss or val_loss < best_val_loss:
            print("Best model")
            best_val_loss = val_loss
            best_model_sd = model.state_dict()

    return best_model_sd

def evaluate_model(model, test_loader, device, binary, multi_label):
    model.eval()
    y_pred, y_true = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            output = model(X_batch)
            mask = y_batch != -100
            print(y_batch)
            print("mask")
            print(mask)
            if multi_label:
                y_pred.extend((output[mask] > 0).tolist())
                y_true.extend(y_batch[mask].tolist())
            else:
                y_pred.extend(output[mask].argmax(dim=-1).tolist())
                y_true.extend(y_batch[mask].tolist())
    
    # Calculate metrics
    average = 'binary' if binary else 'macro'
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average=average, zero_division=0
    )
    accuracy = accuracy_score(y_true, y_pred)  # Calculate accuracy
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }, y_true, y_pred

# def evaluate_model(model, test_loader, device, binary, multi_label):
#     model.eval()
#     y_pred, y_true = [], []
#     with torch.no_grad():
#         for X_batch, y_batch in test_loader:
#             X_batch, y_batch = X_batch.to(device), y_batch.to(device)
#             output = model(X_batch)
#             mask = y_batch != -100
#             if multi_label:
#                 y_pred.extend((output[mask] > 0).tolist())
#                 y_true.extend(y_batch[mask].tolist())
#             else:
#                 y_pred.extend(output[mask].argmax(dim=-1).tolist())
#                 y_true.extend(y_batch[mask].tolist())
    
#     # Calculate metrics
#     average = 'binary' if binary else 'macro'
#     precision, recall, f1, _ = precision_recall_fscore_support(
#         y_true, y_pred, average=average, zero_division=0
#     )
#     accuracy = accuracy_score(y_true, y_pred)  # Calculate accuracy

#     print(f"Test Accuracy: {accuracy:.4f}")
#     print(f"Test Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
#     return y_true, y_pred, accuracy

# def evaluate_model(model, test_loader, device, binary, multi_label):
#     model.eval()
#     y_pred, y_true = [], []
#     with torch.no_grad():
#         for X_batch, y_batch in test_loader:
#             X_batch, y_batch = X_batch.to(device), y_batch.to(device)
#             output = model(X_batch)
#             mask = y_batch != -100
#             if multi_label:
#                 y_pred.extend((output[mask] > 0).tolist())
#                 y_true.extend(y_batch[mask].tolist())
#             else:
#                 y_pred.extend(output[mask].argmax(dim=-1).tolist())
#                 y_true.extend(y_batch[mask].tolist())
#     average = 'binary' if binary else 'macro'
#     precision, recall, f1, _ = precision_recall_fscore_support(
#         y_true, y_pred, average=average, zero_division=0
#     )
#     print(f"Test Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
#     return y_true, y_pred

### ======= Main ======= ###
if __name__ == "__main__":

    print(INPUT_KEY)
    print(LABEL_KEY)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_path = "/work/pi_andrewlan_umass_edu/fikram_umass-edu/dialogue-kt/teacher_moves/processed_data/train_check4.jsonl"
    test_path = "/work/pi_andrewlan_umass_edu/fikram_umass-edu/dialogue-kt/teacher_moves/processed_data/test_check4.jsonl"
  
    train_data_raw = read_jsonl(train_path)
    test_data_raw = read_jsonl(test_path)

    all_train_data = group_data_by_id(train_data_raw)
    test_data = group_data_by_id(test_data_raw)

    train_data, val_data = split_train_val(all_train_data)

    # Train/Val
    sequences_train, labels_train = format_sequences_with_labels(
        train_data, INPUT_KEY, LABEL_KEY
    )

    move_task = LABEL_KEY in ("teacher_move_type", "future_teacher_move_type")
    label_map = MOVE_TO_INDEX if move_task else CORR_LABEL_TO_INDEX

    X_train, y_train = encode_teacher_moves(sequences_train, labels_train, label_map)

    for i in range(10):
        print(sequences_train[i])
        print(X_train[i])
        print(labels_train[i])
        print(y_train[i])

    sequences_val, labels_val = format_sequences_with_labels(
        val_data, INPUT_KEY, LABEL_KEY
    )
    X_val, y_val = encode_teacher_moves(sequences_val, labels_val, label_map)

    train_loader = create_dataloader(X_train, y_train, batch_size=256, shuffle=True)
    val_loader = create_dataloader(X_val, y_val, batch_size=256, shuffle=False)

    print(f"\nSanity check (predicting {LABEL_KEY}):")
    for i in range(5):
        print(f"Input: {sequences_train[i]} → Label: {labels_train[i]}")

    # Create model, output dim depends on task
    BEST_PARAM = {'hidden_dim': 128, 'num_layers': 2, 'dropout': 0.3, 'lr': 0.001}
    model = LSTMModel(
        input_dim=len(MOVE_LIST),
        hidden_dim=128,
        output_dim=len(label_map),
        num_layers=1,
        dropout=0.3
    ).to(device)

    # Train model and load best model at end
    best_model_sd = train_model(model, train_loader, val_loader, False, 0.001, epochs=10, device=device)
    model.load_state_dict(best_model_sd)

    # Test
    sequences_test, labels_test = format_sequences_with_labels(
        test_data, INPUT_KEY, LABEL_KEY
    )
    X_test, y_test = encode_teacher_moves(sequences_test, labels_test, label_map)
    test_loader = create_dataloader(X_test, y_test, batch_size=256, shuffle=False)

    print("\nEvaluating on Test Set:")
    test_results, _, _ = evaluate_model(model, test_loader, device, not move_task, False)

    results = {
        "input_key": INPUT_KEY,
        "label_key": LABEL_KEY,
        "test_results": test_results
    }
    output_file = "/work/pi_andrewlan_umass_edu/fikram_umass-edu/dialogue-kt/results/lstm_results.jsonl"
    save_results_to_jsonl(results, output_file)

# 


# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# from torch.nn.utils.rnn import pad_sequence
# from torch.utils.data import Dataset, DataLoader
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import precision_recall_fscore_support
# import json
# from collections import defaultdict
# import random
# import itertools

# # Globals
# RANDOM_SEED = 42
# torch.manual_seed(RANDOM_SEED)
# random.seed(RANDOM_SEED)

# MOVE_LIST = ['focus', 'telling', 'probing', 'generic']
# MOVE_TO_INDEX = {label: i for i, label in enumerate(MOVE_LIST)}
# CORR_LABEL_LIST = ['false', 'true']
# CORR_LABEL_TO_INDEX = {label: i for i, label in enumerate(CORR_LABEL_LIST)}

# # Data Processing
# def read_jsonl(data_path):
#     with open(data_path) as f:
#         return [json.loads(line) for line in f]

# def group_data_by_id(data):
#     grouped_data = defaultdict(list)
#     for entry in data:
#         grouped_data[entry["id"]].append(entry)
#     return list(grouped_data.values())

# def split_train_val(data, test_size=0.2):
#     train_data, val_data = train_test_split(data, test_size=test_size, random_state=RANDOM_SEED)
#     return train_data, val_data

# def format_sequences_with_labels(data, input_key, label_key):
#     sequences, labels = [], []
#     for conversation in data:
#         input_seq, label_seq = [], []
#         for turn in conversation:
#             if input_key in turn and label_key in turn:
#                 input_seq.append(turn[input_key])
#                 label_seq.append(turn[label_key])
#         sequences.append(input_seq)
#         labels.append(label_seq)
#     return sequences, labels

# def encode_teacher_moves(sequences, labels, label_map):
#     encoded_X = [
#         torch.stack([F.one_hot(torch.tensor(MOVE_TO_INDEX[move]), len(MOVE_TO_INDEX)).type(torch.float32) for move in seq])
#         for seq in sequences
#     ]
#     encoded_y = [
#         torch.tensor([label_map.get(label, -100) for label in seq_labels])
#         for seq_labels in labels
#     ]
#     return encoded_X, encoded_y

# # Dataset and Dataloader
# class TeacherMoveDataset(Dataset):
#     def __init__(self, X, y):
#         self.X = X
#         self.y = y

#     def __len__(self):
#         return len(self.X)

#     def __getitem__(self, idx):
#         return self.X[idx], self.y[idx]

# class LSTMCollator:
#     def __call__(self, batch):
#         X_batch = pad_sequence([X for X, _ in batch], batch_first=True, padding_value=0)
#         y_batch = pad_sequence([y for _, y in batch], batch_first=True, padding_value=-100)
#         return X_batch, y_batch

# def create_dataloader(X, y, batch_size, shuffle):
#     dataset = TeacherMoveDataset(X, y)
#     return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=LSTMCollator())

# # LSTM Model
# class LSTMModel(nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout):
#         super(LSTMModel, self).__init__()
#         self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout)
#         self.fc = nn.Linear(hidden_dim, output_dim)
#         self.dropout = nn.Dropout(dropout)

#     def forward(self, x):
#         lstm_out, _ = self.lstm(x)
#         lstm_out = self.dropout(lstm_out)
#         out = self.fc(lstm_out)
#         return out

# # Train and Evaluate

# def train_model(model, train_loader, val_loader, binary, lr, epochs, device):
#     optimizer = optim.AdamW(model.parameters(), lr=lr)
#     criterion = nn.BCEWithLogitsLoss() if binary else nn.CrossEntropyLoss()
#     best_val_loss, best_model_sd = None, None

#     for epoch in range(epochs):
#         model.train()
#         total_loss = 0
#         for X_batch, y_batch in train_loader:
#             X_batch, y_batch = X_batch.to(device), y_batch.to(device)
#             optimizer.zero_grad()
#             output = model(X_batch)
#             mask = y_batch != -100
#             if binary:
#                 loss = criterion(output[mask].view(-1), y_batch[mask].view(-1))
#             else:
#                 loss = criterion(output.view(-1, output.shape[-1]), y_batch.view(-1))
#             loss.backward()
#             optimizer.step()
#             total_loss += loss.item()

#         model.eval()
#         val_loss = 0
#         with torch.no_grad():
#             for X_batch, y_batch in val_loader:
#                 X_batch, y_batch = X_batch.to(device), y_batch.to(device)
#                 output = model(X_batch)
#                 mask = y_batch != -100
#                 if binary:
#                     loss = criterion(output[mask].view(-1), y_batch[mask].view(-1))
#                 else:
#                     loss = criterion(output.view(-1, output.shape[-1]), y_batch.view(-1))
#                 val_loss += loss.item()

#         print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}")

#         if best_val_loss is None or val_loss < best_val_loss:
#             best_val_loss = val_loss
#             best_model_sd = model.state_dict()

#     return best_model_sd, best_val_loss

# # Main with hyperparameter tuning
# if __name__ == "__main__":
#     INPUT_KEY = "teacher_move_type"
#     LABEL_KEY = "future_teacher_move_type"
    
#     train_path = "/work/pi_andrewlan_umass_edu/fikram_umass-edu/dialogue-kt/teacher_moves/processed_data/train_check4.jsonl"
#     test_path = "/work/pi_andrewlan_umass_edu/fikram_umass-edu/dialogue-kt/teacher_moves/processed_data/test_check4.jsonl"
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     train_data_raw = read_jsonl(train_path)
#     test_data_raw = read_jsonl(test_path)
#     all_train_data = group_data_by_id(train_data_raw)
#     train_data, val_data = split_train_val(all_train_data)

#     sequences_train, labels_train = format_sequences_with_labels(train_data, INPUT_KEY, LABEL_KEY)
#     move_task = LABEL_KEY in ("teacher_move_type", "future_teacher_move_type")
#     label_map = MOVE_TO_INDEX if move_task else CORR_LABEL_TO_INDEX
#     X_train, y_train = encode_teacher_moves(sequences_train, labels_train, label_map)

#     sequences_val, labels_val = format_sequences_with_labels(val_data, INPUT_KEY, LABEL_KEY)
#     X_val, y_val = encode_teacher_moves(sequences_val, labels_val, label_map)

#     batch_size = 256
#     train_loader = create_dataloader(X_train, y_train, batch_size=batch_size, shuffle=True)
#     val_loader = create_dataloader(X_val, y_val, batch_size=batch_size, shuffle=False)

#     # Hyperparameter tuning
#     best_val_loss = float('inf')
#     best_config = None
#     best_state_dict = None

#     #{'hidden_dim': 128, 'num_layers': 2, 'dropout': 0.3, 'lr': 0.001}
#     for hidden_dim, num_layers, dropout, lr in itertools.product([64, 128], [1, 2], [0.2, 0.3], [1e-3, 5e-4]):
#         print(f"\nTrying config: hidden_dim={hidden_dim}, num_layers={num_layers}, dropout={dropout}, lr={lr}")
#         model = LSTMModel(
#             input_dim=len(MOVE_LIST),
#             hidden_dim=hidden_dim,
#             output_dim=len(label_map),
#             num_layers=num_layers,
#             dropout=dropout
#         ).to(device)

#         state_dict, val_loss = train_model(model, train_loader, val_loader, not move_task, lr, 10, device)

#         if val_loss < best_val_loss:
#             best_val_loss = val_loss
#             best_state_dict = state_dict
#             best_config = {
#                 'hidden_dim': hidden_dim,
#                 'num_layers': num_layers,
#                 'dropout': dropout,
#                 'lr': lr
#             }

#     print(f"\nBest config: {best_config} with val loss {best_val_loss:.4f}")
