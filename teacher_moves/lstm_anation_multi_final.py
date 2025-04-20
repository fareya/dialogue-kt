import json
from ast import literal_eval
import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
from collections import defaultdict
import random
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, f1_score

POS_WEIGHT = None
RANDOM_SEED = 42

def train_model(model, train_loader, val_loader, binary, lr, epochs, device, pos_weight=None, num_labels=None):
    model.to(device)

    if pos_weight is not None and pos_weight.any():
        pos_weight_tensor = pos_weight.reshape(1, num_labels).to(device)
    else:
        pos_weight_tensor = None

    if binary:
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    else:
        raise NotImplementedError("This trainer only supports binary multi-label classification.")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    best_val_loss = float("inf")
    best_model_state_dict = None

    for epoch in range(epochs):
        # Training loop
        model.train()
        total_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()

            output = model(X_batch)
            mask = (y_batch != -100).any(dim=-1)
            output_masked = output[mask]
            y_masked = y_batch[mask].float()

            loss = criterion(output_masked, y_masked)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}")

        # Validation loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                output = model(X_batch)
                mask = (y_batch != -100).any(dim=-1)
                output_masked = output[mask]
                y_masked = y_batch[mask].float()

                if y_masked.numel() == 0:  # Edge case: no valid samples
                    continue

                loss = criterion(output_masked, y_masked)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1}/{epochs} - Val Loss: {avg_val_loss:.4f}")

        # Save the best model based on validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state_dict = model.state_dict()

    return best_model_state_dict, best_val_loss


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
                y_pred.extend((output[mask] > 0.45).tolist())
                y_true.extend(y_batch[mask].tolist())
            else:
                y_pred.extend(output[mask].argmax(dim=-1).tolist())
                y_true.extend(y_batch[mask].tolist())
    
    # Calculate metrics
    # average = 'binary' if binary else 'macro'
    # precision, recall, f1, _ = precision_recall_fscore_support(
    #     y_true, y_pred, average=average, zero_division=0
    # )
    # accuracy = accuracy_score(y_true, y_pred)  # Calculate accuracy


    return y_true, y_pred

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
# ---------- Parameters ---------- #
DEBUG = False
PREDICT_CURRENT = False   # Predict the current turn instead of next
PREDICT_CORRECTNESS = False  # Predict correctness instead of labels
VALIDATION_SPLIT = 0.2
LABEL_LIST = [
    'questioning', 'giving_explanation', 'giving_instruction', 'confirmatory_feedback',
    'negative_feedback', 'asking_for_elaboration', 'praising_and_encouraging',
    'providing_further_references', 'managing_discussions', 'conceptual_knowledge', 'computational_skill', 
    'irrelevant_statement', 'acknowledging_tutor_issue','encouraging_peer_tutoring','giving_answers', 'managing_frustration',
    'guiding_peer_tutoring', 'correcting', 'other'
]
LABEL_TO_INDEX = {label: i for i, label in enumerate(LABEL_LIST)}

def chunk_list(lst, chunk_size):
    lst = [int(i) for i in lst]  # Ensure all elements are integers
    print(lst)
    clist = [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]  # Chunk the list
    decoded_list =  []
    for sublist in clist: 
        new_list = [] 
        for i in range(len(sublist)):
            if sublist[i] == 1: 
                new_list.append(LABEL_LIST[i])
        decoded_list.append(new_list)
    print(decoded_list)
    return decoded_list


def evaluate_multi_label(y_true, y_pred):
    # This code IS for the multi-label
    import ast
    from collections import Counter
    from sklearn.preprocessing import MultiLabelBinarizer
    from sklearn.metrics import f1_score, accuracy_score, hamming_loss

    mlb = MultiLabelBinarizer()
    y_true_bin = mlb.fit_transform(y_true)
    y_pred_bin = mlb.transform(y_pred)
    f1_micro = f1_score(y_true_bin, y_pred_bin, average='micro')  # or 'macro', 'samples'
    f1_macro = f1_score(y_true_bin, y_pred_bin, average='macro') 
    f1_samples = f1_score(y_true_bin, y_pred_bin, average='samples') 
    f1_weighted = f1_score(y_true_bin, y_pred_bin, average='weighted') 
    print("F1 Score(weighted):", f1_weighted)
    print("F1 Score(micro):", f1_micro)
    print("F1 Score(macro):", f1_macro)
    print("F1 Score(samples):", f1_samples)
    accuracy = accuracy_score(y_true_bin, y_pred_bin)

    
    print("Accuracy")
    print(accuracy)

    return {"accuracy": accuracy, "f1_macro": f1_macro, "f1_micro": f1_micro, "f1_samples":f1_samples,"f1_weighted":f1_weighted, "total_samples":len(y_true)}



  
# ---------- Data Processing ---------- #
def convert_labels_to_multihot(label_str):
    try:
        labels = literal_eval(label_str) if label_str != '[]' else []
        assert isinstance(labels, list), f"Expected list, got {type(labels)}"
    except Exception as e:
        print(f"Error parsing label string: {label_str}")
        raise e

    vector = torch.zeros(len(LABEL_LIST), dtype=torch.float32)
    for label in labels:
        assert label in LABEL_TO_INDEX, f"Unknown label: {label}"
        vector[LABEL_TO_INDEX[label]] = 1.0
    return vector

def load_and_group_conversations(file_path):
    convo_dict = defaultdict(list)
    with open(file_path, "r") as f:
        for line in f:
            turn = json.loads(line)
            convo_dict[turn["id"]].append(turn)
    grouped = list(convo_dict.values())
    for convo in grouped:
        convo.sort(key=lambda x: x["id2"])
    return grouped

def extract_label_sequences(data):
    sequences, labels = [], []
    for convo in data:
        if DEBUG:
            print("\n=== NEW CONVERSATION ===")
            for turn in convo:
                if turn["is_tutor"]:
                    print(f"id2: {turn['id2']}, Labels: {turn['list_of_labels']}, Success: {turn['Success']}")

        tutor_turns = [turn for turn in convo if turn["is_tutor"]]
        if not tutor_turns:
            continue
        move_vecs = torch.stack([convert_labels_to_multihot(turn["list_of_labels"]) for turn in tutor_turns])
        
        correctness_vecs = torch.Tensor([turn["Success"] for turn in tutor_turns])

        for i in range(move_vecs.size(0)):  # Iterate over rows
            if move_vecs[i].sum() == 0:  # Check if the sum of the row is 0
                move_vecs[i] = torch.full_like(move_vecs[i], -100) 
        

        # ALT 
        # mask = move_vecs.sum(dim=1) == 0
        # move_vecs[mask] = -100
        if PREDICT_CORRECTNESS:
            sequences.append(move_vecs)
            labels.append(correctness_vecs)
        else:
            if PREDICT_CURRENT:
                sequences.append(move_vecs)
                labels.append(move_vecs)
            else:
                sequences.append(move_vecs[:-1])
                labels.append(move_vecs[1:])

        if DEBUG:
            for idx, (vec, correct) in enumerate(zip(move_vecs, correctness_vecs)):
                seq_labels = [label for label, i in LABEL_TO_INDEX.items() if vec[i] == 1.0]
                print(f"Step {idx}: One-hot => {vec.tolist()}, Decoded => {seq_labels}, Correctness => {correct[0]}")
    # print("sequences")
    # print(len(sequences))
    # print(sequences)
    # print("labels")
    # print(labels)
    result_cat = torch.cat(sequences, dim = 0)
    print("result_cat")
    print(result_cat.size())

    # Calculate the number of zeros and ones in each row (ignoring -100)
    valid_mask = result_cat != -100
    num_zeros = (result_cat == 0).sum(dim=0) * valid_mask.sum(dim=0)
    num_ones = (result_cat == 1).sum(dim=0) * valid_mask.sum(dim=0)

    # Compute the ratio of zeros to ones for each row
    ratios = num_zeros / (num_ones + 1e-5)  # Avoid division by zero
    print("ratios")
    print(ratios)
    print(ratios.size())
    print("num_labels")
    num_labels = ratios.size(0)
    print(num_labels)
    return sequences, labels, ratios, num_labels

# ---------- HyperParameter Tuning ---------- #
from itertools import product

def run_hyperparameter_search(
    train_loader, val_loader, num_labels, device, pos_weight, 
    hidden_dims=[64, 128, 256, 512],
    num_layers_list=[1, 2, 3],
    dropouts=[0.1, 0.3, 0.5, 0.7],
    lrs=[1e-2, 5e-3, 1e-3, 5e-4, 1e-4],
    batch_sizes=[16, 32, 64],  # optional if you allow dynamic batch sizes
    epochs=10
):
    best_score = -1
    best_config = None
    results_log = []

    for hidden_dim, num_layers, dropout, lr in product(hidden_dims, num_layers_list, dropouts, lrs):
        print(f"Testing config: hidden_dim={hidden_dim}, num_layers={num_layers}, dropout={dropout}, lr={lr}")
        
        model = LSTMModel(
            input_dim=num_labels,
            hidden_dim=hidden_dim,
            output_dim=num_labels,
            num_layers=num_layers,
            dropout=dropout
        ).to(device)

        best_model_sd, _ = train_model(
            model, train_loader, val_loader,
            binary=True, lr=lr, epochs=epochs,
            device=device, pos_weight=pos_weight,
            num_labels=num_labels
        )

        model.load_state_dict(best_model_sd)

        # Eval on val set
        yt, ypred = evaluate_model(model, val_loader, device, PREDICT_CORRECTNESS, True)
        yt_proc = chunk_list(yt, num_labels)
        ypred_proc = chunk_list(ypred, num_labels)
        metrics = evaluate_multi_label(yt_proc, ypred_proc)

        current_score = metrics["f1_weighted"]
        config = {
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "dropout": dropout,
            "learning_rate": lr,
            "f1_weighted": current_score
        }
        results_log.append(config)

        if current_score > best_score:
            best_score = current_score
            best_config = config

    print("Best Config:", best_config)
    return best_config, results_log


# ---------- Threshold ---------------- # 
def find_best_threshold(y_true, logits, thresholds=np.arange(0.1, 0.9, 0.05)):
    best_thresh = 0.5
    best_score = -1
    all_results = []

    for t in thresholds:
        y_pred = (logits > t).astype(int)
        f1_micro = f1_score(y_true, y_pred, average='micro')
        all_results.append((t, f1_micro))
        if f1_micro > best_score:
            best_score = f1_micro
            best_thresh = t

    print("Threshold Search Results:")
    for t, score in all_results:
        print(f"Threshold: {t:.2f}, Micro F1: {score:.4f}")
    
    print(f"Best Threshold: {best_thresh} (Micro F1: {best_score:.4f})")
    return best_thresh

def collect_logits_and_labels(model, data_loader, device):
    model.eval()
    raw_logits = []
    y_true_eval = []
    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            output = model(X_batch)
            mask = y_batch != -100
            raw_logits.extend(output[mask].cpu().numpy())
            y_true_eval.extend(y_batch[mask].cpu().numpy())
    return np.array(raw_logits), y_true_eval

# ---------- Run Everything ---------- #
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    file_path = "/work/pi_andrewlan_umass_edu/fikram_umass-edu/dialogue-kt/teacher_moves/processed_data/anation_train_data_final.jsonl"
    test_path = "/work/pi_andrewlan_umass_edu/fikram_umass-edu/dialogue-kt/teacher_moves/processed_data/anation_val_data_final.jsonl"

    # Load and prepare training data
    grouped_convos = load_and_group_conversations(file_path)
    train_data, val_data = split_train_val(grouped_convos, VALIDATION_SPLIT)
    sequences_train, labels_train, ratios, num_labels = extract_label_sequences(train_data)
    sequences_val, labels_val, _, _ = extract_label_sequences(val_data)
    train_loader = create_dataloader(sequences_train, labels_train, batch_size=256, shuffle=True)
    val_loader = create_dataloader(sequences_val, labels_val, batch_size=256, shuffle=False)

    # Best Hyperparameters: {'hidden_dim': 256, 'num_layers': 2, 'dropout': 0.5, 'learning_rate': 0.001}

    
    # Create model, output dim depends on task
    best_config, all_results = run_hyperparameter_search(
        train_loader, val_loader, num_labels, device, pos_weight=ratios
    )
    print(best_config)
    # Train final model with best config
    model = LSTMModel(
        input_dim=num_labels,
        hidden_dim=best_config["hidden_dim"],
        output_dim=num_labels,
        num_layers=best_config["num_layers"],
        dropout=best_config["dropout"]
    ).to(device)

    best_model_sd, _ = train_model(
        model, train_loader, val_loader,
        binary=True, lr=best_config["learning_rate"], epochs=10,
        device=device, pos_weight=ratios, num_labels=num_labels
    )
    model.load_state_dict(best_model_sd)

    # Train model and load best model at end
    best_model_sd, _= train_model(model, train_loader, val_loader, True, 0.001, epochs=10, device=device, pos_weight = ratios, num_labels = num_labels)
    model.load_state_dict(best_model_sd)

    # Load and prepare test data
    print("Evaluating on test set...")

    test_convos = load_and_group_conversations(test_path)
    sequences_test, labels_test, _ , _ = extract_label_sequences(test_convos)
    test_loader = create_dataloader(sequences_test, labels_test, batch_size=256, shuffle=False)


    yt, ypred = evaluate_model(model, test_loader, device, PREDICT_CORRECTNESS, True)

    if PREDICT_CORRECTNESS:
        name = "dialogue_correcntess"
    elif PREDICT_CURRENT:
        name = "teacher_move_type"
    else: 
        name = "future_teacher_move_type"
        yt_proc = chunk_list(yt, len(LABEL_LIST))
        ypred_proc = chunk_list(ypred, len(LABEL_LIST))
        results = evaluate_multi_label(yt_proc, ypred_proc)
    output_file = "/work/pi_andrewlan_umass_edu/fikram_umass-edu/dialogue-kt/results/lstm_results.jsonl"
    final_res = {'name':name, "res":results, "data": "anation"}
    
    total = 0
    for x in range(len(yt_proc)):
        if yt_proc[x] ==  ypred_proc[x]:
            total = total +1
    print("REF")
    print("Total")
    print(total)
    print("Exact Match")
    print(total/len(yt_proc))
    save_results_to_jsonl(final_res, output_file)

    # raw_logits, y_true_eval = collect_logits_and_labels(model, test_loader, device)
    # y_true_eval_chunked = chunk_list(y_true_eval, len(LABEL_LIST))
    # best_thresh = find_best_threshold(y_true_eval_chunked, raw_logits)
    # y_pred_final = (raw_logits > best_thresh).astype(int)
    # y_pred_labels = chunk_list(y_pred_final, len(LABEL_LIST))
    # results = evaluate_multi_label(y_true_eval_chunked, y_pred_labels)
    # print(results)
