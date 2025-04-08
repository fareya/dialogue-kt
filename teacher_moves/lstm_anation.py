import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from collections import defaultdict
from sklearn.metrics import precision_recall_fscore_support
import pandas as pd

# ---------- Parameters ---------- #
INPUT_WINDOW = 3
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

# ---------- Data Processing ---------- #
def convert_labels_to_multihot(label_str):
    try:
        labels = eval(label_str) if label_str != '[]' else []
        assert isinstance(labels, list), f"Expected list, got {type(labels)}"
    except Exception as e:
        print(f"Error parsing label string: {label_str}")
        raise e

    vector = np.zeros(len(LABEL_LIST), dtype=np.float32)
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
        convo.sort(key=lambda x: x["order"])
    return grouped

def extract_label_sequences(data, input_window=3):
    sequences = []
    for convo in data:
        if DEBUG:
            print("\n=== NEW CONVERSATION ===")
            for turn in convo:
                if turn["is_tutor"]:
                    print(f"Order: {turn['order']}, Labels: {turn['list_of_labels']}, Success: {turn['Success']}")

        tutor_turns = [turn for turn in convo if turn["is_tutor"]]
        label_vecs = [convert_labels_to_multihot(turn["list_of_labels"]) for turn in tutor_turns]
        correctness_vecs = [np.array([turn["Success"]], dtype=np.float32) for turn in tutor_turns]

        if DEBUG:
            for idx, (vec, correct) in enumerate(zip(label_vecs, correctness_vecs)):
                labels = [label for label, i in LABEL_TO_INDEX.items() if vec[i] == 1.0]
                print(f"Step {idx}: One-hot => {vec.tolist()}, Decoded => {labels}, Correctness => {correct[0]}")

        for i in range(len(label_vecs) - input_window):
            x_seq = label_vecs[i:i + input_window]
            if PREDICT_CURRENT:
                y = label_vecs[i + input_window - 1] if not PREDICT_CORRECTNESS else correctness_vecs[i + input_window - 1]
            else:
                y = label_vecs[i + input_window] if not PREDICT_CORRECTNESS else correctness_vecs[i + input_window]
            sequences.append((np.stack(x_seq), y))
    return sequences

# ---------- Dataset Class ---------- #
class LabelSequenceDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        x, y = self.sequences[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# ---------- Model Definition ---------- #
class LabelPredictorLSTM(nn.Module):
    def __init__(self, input_size=11, hidden_size=64, num_layers=1, output_size=11):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # Last timestep
        out = self.fc(out)
        return self.sigmoid(out)

# ---------- Training and Validation ---------- #
def train_model(model, train_loader, val_loader, epochs=10, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            output = model(x_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for x_val, y_val in val_loader:
                x_val, y_val = x_val.to(device), y_val.to(device)
                val_output = model(x_val)
                val_loss = criterion(val_output, y_val)
                total_val_loss += val_loss.item()

        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {total_train_loss:.4f} - Val Loss: {total_val_loss:.4f}")

# ---------- Run Everything ---------- #
if __name__ == "__main__":
    file_path = "/work/pi_andrewlan_umass_edu/fikram_umass-edu/dialogue-kt/teacher_moves/processed_data/anation_train_data.jsonl"
    test_path = "/work/pi_andrewlan_umass_edu/fikram_umass-edu/dialogue-kt/teacher_moves/processed_data/anation_val_data.jsonl"

    # Load and prepare training data
    grouped_convos = load_and_group_conversations(file_path)
    sequences = extract_label_sequences(grouped_convos, input_window=INPUT_WINDOW)
    dataset = LabelSequenceDataset(sequences)
    val_size = int(len(dataset) * VALIDATION_SPLIT)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])


    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=2)

    output_dim = 1 if PREDICT_CORRECTNESS else len(LABEL_LIST)
    model = LabelPredictorLSTM(input_size=len(LABEL_LIST), output_size=output_dim)
    train_model(model, train_loader, val_loader, epochs=60)

    # Load and prepare test data
    print("Evaluating on test set...")

    test_convos = load_and_group_conversations(test_path)
    test_sequences = extract_label_sequences(test_convos, input_window=INPUT_WINDOW)
    test_dataset = LabelSequenceDataset(test_sequences)
    test_loader = DataLoader(test_dataset, batch_size=2)


    y_true = []
    y_pred = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    criterion = nn.BCELoss()
    total_test_loss = 0
    with torch.no_grad():
        for x_test, y_test in test_loader:
            x_test, y_test = x_test.to(device), y_test.to(device)
            test_output = model(x_test)
            test_loss = criterion(test_output, y_test)
            total_test_loss += test_loss.item()

            preds = (test_output > 0.2).float().cpu().numpy()
            labels = y_test.cpu().numpy()
            y_pred.extend(preds)
            y_true.extend(labels)

    print(f"Test Loss: {total_test_loss:.4f}")

    average = 'binary' if PREDICT_CORRECTNESS else 'macro'
    precision, recall, f1, _ = precision_recall_fscore_support(
        np.array(y_true), np.array(y_pred), average=average, zero_division=0
    )
    print(f"Test Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    # Compute accuracy
    correct = 0
    total = 0
    for pred, true in zip(y_pred, y_true):
        if np.array_equal(pred, true):
            correct += 1
        total += 1
    accuracy = correct / total if total > 0 else 0
    print(f"Test Accuracy: {accuracy:.4f}")

    # Save predictions vs actual
    output_df = pd.DataFrame({
        'predicted': [list(p) for p in y_pred],
        'actual': [list(t) for t in y_true]
    })
    output_df.to_csv("predictions_vs_actuals.csv", index=False)
    print("Predictions saved to predictions_vs_actuals.csv")
