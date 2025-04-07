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
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import json
from collections import defaultdict
import random

RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

### ======= Configuration ======= ###
window_size = 3
INPUT_KEY = "teacher_move_type"
LABEL_KEY = "correctness_annotation"  # options: "teacher_move_type", "correctness_annotation"
LABEL_MODE = "future"  # options: "future", "current"

### ======= Data Processing ======= ###
def read_jsonl(data_path):
    with open(data_path) as f:
        return [json.loads(line) for line in f]

def group_data_by_id(data):
    grouped_data = defaultdict(list)
    for entry in data:
        grouped_data[entry["id"]].append(entry)
    return grouped_data

def split_train_val(grouped_data, test_size=0.2):
    train_data, val_data = train_test_split(list(grouped_data.values()), test_size=test_size, random_state=RANDOM_SEED)
    return train_data, val_data

def get_test_formatted(grouped_data):
    return list(grouped_data.values())

def format_sequences_with_labels(data, window_size, input_key, label_key, label_mode):
    assert label_mode in ["current", "future"], "label_mode must be 'current' or 'future'"
    sequences, labels = [], []
    for conversation in data:
        input_seq = [turn[input_key] for turn in conversation if input_key in turn and label_key in turn]
        label_seq = [turn[label_key] for turn in conversation if input_key in turn and label_key in turn]

        for i in range(len(input_seq) - window_size):
            window = input_seq[i:i+window_size]
            if label_mode == "future":
                label_idx = i + window_size
            else:
                label_idx = i + window_size - 1

            if label_idx < len(label_seq):
                label = label_seq[label_idx]
                if label is not None:
                    sequences.append(window)
                    labels.append(label)
    return sequences, labels

def encode_teacher_moves(sequences, labels, move2idx=None):
    if move2idx is None:
        all_labels = sorted(set([move for seq in sequences for move in seq] + labels))
        move2idx = {move: idx for idx, move in enumerate(all_labels)}
    
    encoded_X = torch.tensor([[move2idx[move] for move in seq] for seq in sequences])
    encoded_y = torch.tensor([move2idx[label] for label in labels])
    
    return encoded_X, encoded_y, move2idx

### ======= Dataset and Dataloader ======= ###
class TeacherMoveDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def create_dataloader(X, y, batch_size=32, shuffle=True):
    dataset = TeacherMoveDataset(X, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

### ======= LSTM Model ======= ###
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, num_layers, dropout):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        out = self.fc(lstm_out[:, -1, :])
        return out

### ======= Training & Evaluation ======= ###
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs, device):
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                output = model(X_batch)
                predictions = torch.argmax(output, dim=1)
                correct += (predictions == y_batch).sum().item()
                total += y_batch.size(0)
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}, Val Acc: {correct/total:.4f}")

def evaluate_model(model, test_loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            output = model(X_batch)
            predictions = torch.argmax(output, dim=1)
            correct += (predictions == y_batch).sum().item()
            total += y_batch.size(0)
    print(f"Test Accuracy: {correct/total:.4f}")

### ======= Main ======= ###
if __name__ == "__main__":

    print(INPUT_KEY)
    print(LABEL_KEY)
    print(LABEL_MODE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_path = "/work/pi_andrewlan_umass_edu/fikram_umass-edu/dialogue-kt/teacher_moves/processed_data/train_check4.jsonl"
    test_path = "/work/pi_andrewlan_umass_edu/fikram_umass-edu/dialogue-kt/teacher_moves/processed_data/test_check4.jsonl"

    train_data_raw = read_jsonl(train_path)
    test_data_raw = read_jsonl(test_path)

    grouped_train = group_data_by_id(train_data_raw)
    grouped_test = get_test_formatted(group_data_by_id(test_data_raw))

    train_data, val_data = split_train_val(grouped_train)

    # Train/Val
    sequences_train, labels_train = format_sequences_with_labels(
        train_data, window_size, INPUT_KEY, LABEL_KEY, LABEL_MODE
    )

    X_train, y_train, label2idx = encode_teacher_moves(sequences_train, labels_train)

    for i in range(10):
        print(sequences_train[i])
        print(X_train[i])
        print(labels_train[i])
        print(y_train[i])

    print(label2idx)
    
    sequences_val, labels_val = format_sequences_with_labels(
        val_data, window_size, INPUT_KEY, LABEL_KEY, LABEL_MODE
    )
    X_val, y_val, _ = encode_teacher_moves(sequences_val, labels_val, label2idx)

    train_loader = create_dataloader(X_train, y_train)
    val_loader = create_dataloader(X_val, y_val)

    print(f"\nSanity check (predicting {LABEL_MODE} {LABEL_KEY}):")
    for i in range(5):
        print(f"Input: {sequences_train[i]} → Label: {labels_train[i]}")

    print("vocab_size", len(label2idx))
    model = LSTMModel(
        vocab_size=len(label2idx),
        embedding_dim=64,
        hidden_dim=128,
        output_dim=len(label2idx),
        num_layers=1,
        dropout=0.3
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_model(model, train_loader, val_loader, criterion, optimizer, epochs=5, device=device)

    # Test
    sequences_test, labels_test = format_sequences_with_labels(
        grouped_test, window_size, INPUT_KEY, LABEL_KEY, LABEL_MODE
    )
    X_test, y_test, _ = encode_teacher_moves(sequences_test, labels_test, label2idx)
    test_loader = create_dataloader(X_test, y_test, shuffle=False)

    print("\nEvaluating on Test Set:")
    evaluate_model(model, test_loader, device)
