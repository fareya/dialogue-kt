import json
from ast import literal_eval
import torch
from collections import defaultdict

from teacher_moves.lstm import LSTMModel, create_dataloader, evaluate_model, train_model, split_train_val

# ---------- Parameters ---------- #
DEBUG = False
PREDICT_CURRENT = False   # Predict the current turn instead of next
PREDICT_CORRECTNESS = True  # Predict correctness instead of labels
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
        convo.sort(key=lambda x: x["order"])
    return grouped

def extract_label_sequences(data):
    sequences, labels = [], []
    for convo in data:
        if DEBUG:
            print("\n=== NEW CONVERSATION ===")
            for turn in convo:
                if turn["is_tutor"]:
                    print(f"Order: {turn['order']}, Labels: {turn['list_of_labels']}, Success: {turn['Success']}")

        tutor_turns = [turn for turn in convo if turn["is_tutor"]]
        if not tutor_turns:
            continue
        move_vecs = torch.stack([convert_labels_to_multihot(turn["list_of_labels"]) for turn in tutor_turns])
        correctness_vecs = torch.Tensor([turn["Success"] for turn in tutor_turns])

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

    return sequences, labels

# ---------- Run Everything ---------- #
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    file_path = "teacher_moves/processed_data/anation_train_data.jsonl"
    test_path = "teacher_moves/processed_data/anation_val_data.jsonl"

    # Load and prepare training data
    grouped_convos = load_and_group_conversations(file_path)
    train_data, val_data = split_train_val(grouped_convos, VALIDATION_SPLIT)
    sequences_train, labels_train = extract_label_sequences(train_data)
    sequences_val, labels_val = extract_label_sequences(val_data)
    train_loader = create_dataloader(sequences_train, labels_train, batch_size=256, shuffle=True)
    val_loader = create_dataloader(sequences_val, labels_val, batch_size=256, shuffle=False)

    # Create model, output dim depends on task
    output_dim = 1 if PREDICT_CORRECTNESS else len(LABEL_LIST)
    model = LSTMModel(
        input_dim=len(LABEL_LIST),
        hidden_dim=128,
        output_dim=output_dim,
        num_layers=1,
        dropout=0.3
    ).to(device)

    # Train model and load best model at end
    best_model_sd = train_model(model, train_loader, val_loader, True, 0.001, epochs=10, device=device)
    model.load_state_dict(best_model_sd)

    # Load and prepare test data
    print("Evaluating on test set...")

    test_convos = load_and_group_conversations(test_path)
    sequences_test, labels_test = extract_label_sequences(test_convos)
    test_loader = create_dataloader(sequences_test, labels_test, batch_size=256, shuffle=False)

    y_true, y_pred = evaluate_model(model, test_loader, device, PREDICT_CORRECTNESS, True)

    # Save predictions vs actual
    # output_df = pd.DataFrame({
    #     'predicted': [list(p) for p in y_pred],
    #     'actual': [list(t) for t in y_true]
    # })
    # output_df.to_csv("predictions_vs_actuals.csv", index=False)
    # print("Predictions saved to predictions_vs_actuals.csv")
