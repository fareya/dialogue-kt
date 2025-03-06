
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
import pandas as pd
from itertools import product
import json
from collections import defaultdict
from helpers import get_test
RANDOM_SEED = 42

# Set device
def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device

def load_tokenizer(name="bert-base-uncased"):
    return AutoTokenizer.from_pretrained(name)

def load_data(filepath):
    return pd.read_json(filepath, lines=True)

def read_jsonl(data_path):
    with open(data_path) as f:
        return [json.loads(line) for line in f]

def group_data_by_id(data):
    # add all ids to a list
    grouped_data = defaultdict(list)
    for entry in data:
        grouped_data[entry["id"]].append(entry)
    
    
    # remove duplicates 
    # for key in grouped_data.keys():
    #     grouped_data[key] = pd.DataFrame(grouped_data[key]).drop_duplicates(subset=['teacher_move', 'student_move']).to_dict('records')
    return grouped_data

def split_train_val(grouped_data, test_size=0.2):
    train_data, val_data = train_test_split(list(grouped_data.values()), test_size=test_size, random_state=RANDOM_SEED)
    return train_data, val_data

def get_test_formatted(grouped_data):
    test_data = list(grouped_data.values())
    return test_data

# def format_dialogues_with_window(df, window_size, label_column="future_teacher_move_type", include_teacher_move_type=True):
#     formatted_texts, formatted_labels = [], []
#     grouped = df.groupby("id")
    
#     for _, conversation in grouped:
#         conversation = conversation.sort_values("turn")
#         dialogue_window = []
        
#         for _, row in conversation.iterrows():
#             dialogue_window.append(f"Teacher: {row['teacher_move']} Student: {row['student_move']}")
            
#             if len(dialogue_window) > window_size:
#                 dialogue_window.pop(0)
            
#             if len(dialogue_window) >= window_size and pd.notna(row[label_column]):
#                 if include_teacher_move_type and 'teacher_move_type' in row:
#                     formatted_labels.append(f"{row[label_column]}_{row['teacher_move_type']}")
#                 else:
#                     formatted_texts.append(" ".join(dialogue_window))
#                     formatted_labels.append(row[label_column])
    
#     return formatted_texts, formatted_labels

def format_dialogue_with_window(data, pred_label_name, window_size):
    # change from dataframe to list of dicts
    # data = df.to_dict(orient='records')

    formatted_outputs = []
    labels = [] 
    ids = [] 
    print("Length of data: ", len(data))
    for conversation in data: 
        # print("Length of conversation: ", len(conversation))
        dialogue_text = []
        # print(conversation)
        # print(enumerate(conversation))
        # print(len(conversation))
        for i, turn in enumerate(conversation):
            if pred_label_name == "future_teacher_move_type":
                # print("Turn: ", turn)
                if turn['future_teacher_move_type']:
                    dialogue_text.append(f"\nTeacher Turn {turn['turn']}: {turn['teacher_move']} \nStudent Turn  {turn['turn']}: {turn['student_move']}")
                    user_dialogue = "".join(dialogue_text) # replaced 
                    user_prompt = "[BEGIN DIALOGUE]" + user_dialogue + "\n[END DIALOGUE]"       
                    formatted_outputs.append(user_prompt)
                    labels.append(turn[pred_label_name])
                    ids.append(turn["id"])
            
            elif pred_label_name == "teacher_move_type":
                # print("Turn: ", turn)
                # print (turn)
                if turn['teacher_move_type'] and i < len(conversation)-1:
                    curr_teacher_move_model= conversation[i+1]['teacher_move']
                    curr_teacher_turn = conversation[i+1]['turn']
                    curr_teacher_type = conversation[i+1]['teacher_move_type']
                    
                    additional_text = f"\nTeacher Turn {curr_teacher_turn}: {curr_teacher_move_model}"
                    dialogue_text.append(f"\nTeacher Turn {turn['turn']}: {turn['teacher_move']} \nStudent Turn  {turn['turn']}: {turn['student_move']}")
                    user_dialogue = "".join(dialogue_text) # replaced 
                    user_prompt = "[BEGIN DIALOGUE]" + user_dialogue + additional_text + "\n[END DIALOGUE]"       
                    formatted_outputs.append(user_prompt)
                    labels.append(curr_teacher_type)
                    ids.append(turn["id"])
            if len(dialogue_text) > window_size:
                dialogue_text.pop(0)

    print("Length of formatted outputs: ", len(formatted_outputs))

    # print a few examples
    for i in range(5):
        print("Formatted Output: ", formatted_outputs[i])
        print("Label: ", labels[i])
        print("ID: ", ids[i])
        print("-----")
    assert len(formatted_outputs) == len(labels)
    return formatted_outputs, labels

def preprocess_data(data, tokenizer, window_size, device):
    print("Preprocessing data...")
   
    # pred_label_name = "future_teacher_move_type"
    pred_label_name = "correctness_annotation"
    print("Label column: ", pred_label_name)
    formatted_texts, formatted_labels =format_dialogue_with_window(data, pred_label_name, window_size)
    encoding = tokenizer(formatted_texts, padding=True, truncation=True, return_tensors="pt")
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
    
    label_mapping = {label: idx for idx, label in enumerate(set(formatted_labels))}
    labels_encoded = torch.tensor([label_mapping[label] for label in formatted_labels], dtype=torch.long, device=device)
    
    return input_ids, attention_mask, labels_encoded, label_mapping

def split_data(input_ids, labels_encoded, test_size=0.2):
    return train_test_split(input_ids, labels_encoded, test_size=test_size, random_state=42)

class TeacherMoveDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def create_dataloaders(X_train, X_val, y_train, y_val, batch_size):
    train_dataset = TeacherMoveDataset(X_train, y_train)
    val_dataset = TeacherMoveDataset(X_val, y_val)
    return DataLoader(train_dataset, batch_size=batch_size, shuffle=True), DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

def create_dataloader(X, y, batch_size):
    dataset = TeacherMoveDataset(X, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

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

def hyperparameter_tuning(vocab_size, label_mapping, train_loader, val_loader, test_loader, device, window_sizes=[3, 5]):
    best_acc = 0
    best_model = None
    
    param_grid = {
        "embedding_dim": [64, 128],
        "hidden_dim": [128, 256],
        "num_layers": [1, 2],
        "dropout": [0.3, 0.5],
        "lr": [0.001, 0.0005]
    }
    
    for window_size in window_sizes:
        print(f"Testing with window size: {window_size}")
        for params in product(*param_grid.values()):
            param_dict = dict(zip(param_grid.keys(), params))
            model = LSTMModel(vocab_size, param_dict["embedding_dim"], param_dict["hidden_dim"], len(label_mapping), param_dict["num_layers"], param_dict["dropout"]).to(device)
            criterion = nn.CrossEntropyLoss().to(device)
            optimizer = optim.Adam(model.parameters(), lr=param_dict["lr"])
            
            print(f"Testing params: {param_dict}")
            train_model(model, train_loader, val_loader, criterion, optimizer, 5, device)
            evaluate_model(model, test_loader, device)
            
if __name__ == "__main__":
    device = get_device()
    tokenizer = load_tokenizer()
    # df = load_data("/work/pi_andrewlan_umass_edu/fikram_umass-edu/dialogue-kt/teacher_moves/processed_data/train_check4.jsonl")    
    data  = read_jsonl("/work/pi_andrewlan_umass_edu/fikram_umass-edu/dialogue-kt/teacher_moves/processed_data/train_check4.jsonl")
    test_data = read_jsonl("/work/pi_andrewlan_umass_edu/fikram_umass-edu/dialogue-kt/teacher_moves/processed_data/test_check4.jsonl")
    for window_size in [3, 5]:  # Testing different window sizes
        grouped_data = group_data_by_id(data)
        train_data, val_data = split_train_val(grouped_data)
        input_ids_train, attention_mask_train, labels_encoded_train, label_mapping_train = preprocess_data(train_data, tokenizer, window_size, device)
        input_ids_val, attention_mask_val, labels_encoded_val, label_mapping_val = preprocess_data(val_data, tokenizer, window_size, device)

        train_loader = create_dataloader(input_ids_train, labels_encoded_train,  32)
        val_loader = create_dataloader(input_ids_val, labels_encoded_val, 32)

        test_data = get_test()
        input_ids_test, attention_mask_test, labels_encoded_test, label_mapping_test = preprocess_data(test_data, tokenizer, window_size, device)
        test_loader = create_dataloader(input_ids_test, labels_encoded_test, 32)

        hyperparameter_tuning(tokenizer.vocab_size, label_mapping_train, train_loader, val_loader, test_loader, device, [window_size])
