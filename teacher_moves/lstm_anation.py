import json
from ast import literal_eval
import torch
from collections import defaultdict

from lstm import LSTMModel, create_dataloader, evaluate_model, train_model, split_train_val, save_results_to_jsonl

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


def evaluate_multi_label_safe_2(y_true_raw, y_pred_raw):
    import ast
    from collections import Counter
    from sklearn.preprocessing import MultiLabelBinarizer
    from sklearn.metrics import f1_score, accuracy_score, hamming_loss

    all_labels = [
        'confirmatory feedback', 'negative feedback', 'correcting',
        'giving instruction', 'giving explanation',
        'providing further references', 'questioning', 'asking for elaboration',
        'praising and encouraging', 'managing frustration',
        'managing discussions', 'giving answers', 'encouraging peer tutoring',
        'guiding peer tutoring', 'acknowledging tutor issue', 'other',
        'irrelevant statement', 'computational skill', 'linguistic knowledge',
        'conceptual knowledge', 'strategic knowledge', 'affective control', 'none'
    ]
    # Parse and clean
    y_true = [ast.literal_eval(s) if isinstance(s, str) else s for s in y_true_raw]
    y_pred = [ast.literal_eval(s) if isinstance(s, str) else s for s in y_pred_raw]

    print(y_true)
    print(y_pred)
    # Strip spaces
    y_true = [[label.strip() for label in ex] for ex in y_true]
    y_pred = [[label.strip() for label in ex] for ex in y_pred]

    # Initialize binarizer with fixed class order
    mlb = MultiLabelBinarizer(classes=all_labels)
    mlb.fit(y_true + y_pred)

    y_true_bin = mlb.transform(y_true)
    y_pred_bin = mlb.transform(y_pred)

    # Sanity check
    if y_pred_bin.shape != y_true_bin.shape:
        raise ValueError("Shape mismatch between predictions and ground truth after binarization.")

    # Compute metrics
    f1 = f1_score(y_true_bin, y_pred_bin, average='samples')
    exact_match_acc = accuracy_score(y_true_bin, y_pred_bin)
    hamming_acc = 1 - hamming_loss(y_true_bin, y_pred_bin)

    print(f"F1 Score (samples): {f1:.4f}")
    print(f"Exact Match Accuracy: {exact_match_acc:.4f}")
    print(f"Hamming Accuracy: {hamming_acc:.4f}")

    # Analyze distributions of y_true and y_pred
    y_true_flat = [label for sublist in y_true for label in sublist]
    y_pred_flat = [label for sublist in y_pred for label in sublist]

    y_true_distribution = Counter(y_true_flat)
    y_pred_distribution = Counter(y_pred_flat)

    print("\nDistribution of y_true:")
    for cls, count in y_true_distribution.items():
        print(f"{cls}: {count}")

    print("\nDistribution of y_pred:")
    for cls, count in y_pred_distribution.items():
        print(f"{cls}: {count}")

    # Identify most common misclassifications
    misclassification_counts = Counter(
        (true_label, pred_label)
        for true_labels, pred_labels in zip(y_true, y_pred)
        for true_label in true_labels
        for pred_label in pred_labels
        if true_label != pred_label
    )
    print("\nMost Common Misclassifications:")
    for (true_label, pred_label), count in misclassification_counts.most_common(5):
        print(f"True: {true_label}, Predicted: {pred_label}, Count: {count}")

    return f1, exact_match_acc, hamming_acc

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
    # print(sequences)
    # print("labels")
    # print(labels)
    return sequences, labels

# ---------- Run Everything ---------- #
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    file_path = "/work/pi_andrewlan_umass_edu/fikram_umass-edu/dialogue-kt/teacher_moves/processed_data/anation_train_data_final.jsonl"
    test_path = "/work/pi_andrewlan_umass_edu/fikram_umass-edu/dialogue-kt/teacher_moves/processed_data/anation_val_data_final.jsonl"

    # Load and prepare training data
    grouped_convos = load_and_group_conversations(file_path)
    train_data, val_data = split_train_val(grouped_convos, VALIDATION_SPLIT)
    sequences_train, labels_train = extract_label_sequences(train_data)
    sequences_val, labels_val = extract_label_sequences(val_data)
    train_loader = create_dataloader(sequences_train, labels_train, batch_size=256, shuffle=True)
    val_loader = create_dataloader(sequences_val, labels_val, batch_size=256, shuffle=False)

    # Best Hyperparameters: {'hidden_dim': 256, 'num_layers': 2, 'dropout': 0.5, 'learning_rate': 0.001}

    # Create model, output dim depends on task
    output_dim = 1 if PREDICT_CORRECTNESS else len(LABEL_LIST)
    model = LSTMModel(
        input_dim=len(LABEL_LIST),
        hidden_dim=256,
        output_dim=output_dim,
        num_layers=2,
        dropout=0.5
    ).to(device)

    # Train model and load best model at end
    best_model_sd = train_model(model, train_loader, val_loader, True, 0.001, epochs=10, device=device)
    model.load_state_dict(best_model_sd)

    # Load and prepare test data
    print("Evaluating on test set...")

    test_convos = load_and_group_conversations(test_path)
    sequences_test, labels_test = extract_label_sequences(test_convos)
    test_loader = create_dataloader(sequences_test, labels_test, batch_size=256, shuffle=False)


    results, yt, ypred = evaluate_model(model, test_loader, device, PREDICT_CORRECTNESS, True)
    yt_proc = chunk_list(yt, len(LABEL_LIST))
    ypred_proc = chunk_list(ypred, len(LABEL_LIST))
    evaluate_multi_label_safe_2(yt_proc, ypred_proc)
    total_corr = 0 
    total = len(ypred)

    if not PREDICT_CORRECTNESS:
        for i in range(len(yt_proc)):
            if yt_proc[i] == ypred_proc[i]:
                total_corr= total_corr+1 
        accuracy = total_corr/total
        print(accuracy) # <-- exact match, is this right? 

    if PREDICT_CORRECTNESS:
        name = "dialogue_correcntess"
    elif PREDICT_CURRENT:
        name = "teacher_move_type"
    else: 
         name = "future_teacher_move_type"
    output_file = "/work/pi_andrewlan_umass_edu/fikram_umass-edu/dialogue-kt/results/lstm_results.jsonl"
    final_res = {'name':name, "res":results, "data": "anation"}
    save_results_to_jsonl(final_res, output_file)
    # Save predictions vs actual
    # output_df = pd.DataFrame({
    #     'predicted': [list(p) for p in y_pred],
    #     'actual': [list(t) for t in y_true]
    # })
    # output_df.to_csv("predictions_vs_actuals.csv", index=False)
    # print("Predictions saved to predictions_vs_actuals.csv")






# import json
# from ast import literal_eval
# import torch
# from collections import defaultdict
# import itertools
# from lstm import LSTMModel, create_dataloader, evaluate_model, train_model, split_train_val

# # ---------- Parameters ---------- #
# DEBUG = False
# VALIDATION_SPLIT = 0.2
# LABEL_LIST = [
#     'questioning', 'giving_explanation', 'giving_instruction', 'confirmatory_feedback',
#     'negative_feedback', 'asking_for_elaboration', 'praising_and_encouraging',
#     'providing_further_references', 'managing_discussions', 'conceptual_knowledge', 'computational_skill', 
#     'irrelevant_statement', 'acknowledging_tutor_issue','encouraging_peer_tutoring','giving_answers', 'managing_frustration',
#     'guiding_peer_tutoring', 'correcting', 'other'
# ]
# LABEL_TO_INDEX = {label: i for i, label in enumerate(LABEL_LIST)}

# PREDICT_CURRENT = False
# PREDICT_CORRECTNESS = True

# # ---------- Data Processing ---------- #
# def convert_labels_to_multihot(label_str):
#     try:
#         labels = literal_eval(label_str) if label_str != '[]' else []
#         assert isinstance(labels, list), f"Expected list, got {type(labels)}"
#     except Exception as e:
#         print(f"Error parsing label string: {label_str}")
#         raise e

#     vector = torch.zeros(len(LABEL_LIST), dtype=torch.float32)
#     for label in labels:
#         assert label in LABEL_TO_INDEX, f"Unknown label: {label}"
#         vector[LABEL_TO_INDEX[label]] = 1.0
#     return vector

# def load_and_group_conversations(file_path):
#     convo_dict = defaultdict(list)
#     with open(file_path, "r") as f:
#         for line in f:
#             turn = json.loads(line)
#             convo_dict[turn["id"]].append(turn)
#     grouped = list(convo_dict.values())
#     for convo in grouped:
#         convo.sort(key=lambda x: x["id2"])
#     return grouped

# def extract_label_sequences(data):
#     sequences, labels = [], []
#     for convo in data:
#         tutor_turns = [turn for turn in convo if turn["is_tutor"]]
#         if not tutor_turns:
#             continue
#         move_vecs = torch.stack([convert_labels_to_multihot(turn["list_of_labels"]) for turn in tutor_turns])
#         correctness_vecs = torch.Tensor([turn["Success"] for turn in tutor_turns])

#         if PREDICT_CORRECTNESS:
#             sequences.append(move_vecs)
#             labels.append(correctness_vecs)
#         else:
#             if PREDICT_CURRENT:
#                 sequences.append(move_vecs)
#                 labels.append(move_vecs)
#             else:
#                 sequences.append(move_vecs[:-1])
#                 labels.append(move_vecs[1:])
#     return sequences, labels

# # ---------- Run Everything ---------- #
# if __name__ == "__main__":
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     file_path = "/work/pi_andrewlan_umass_edu/fikram_umass-edu/dialogue-kt/teacher_moves/processed_data/anation_train_data_final.jsonl"
#     test_path = "/work/pi_andrewlan_umass_edu/fikram_umass-edu/dialogue-kt/teacher_moves/processed_data/anation_val_data_final.jsonl"

#     # Load and prepare training data
#     grouped_convos = load_and_group_conversations(file_path)
#     train_data, val_data = split_train_val(grouped_convos, VALIDATION_SPLIT)
#     sequences_train, labels_train = extract_label_sequences(train_data)
#     sequences_val, labels_val = extract_label_sequences(val_data)
#     train_loader = create_dataloader(sequences_train, labels_train, batch_size=256, shuffle=True)
#     val_loader = create_dataloader(sequences_val, labels_val, batch_size=256, shuffle=False)

#     # Best Hyperparameters: {'hidden_dim': 256, 'num_layers': 2, 'dropout': 0.5, 'learning_rate': 0.001}
#     # Hyperparameter grid
#     hidden_dims = [256]
#     num_layers_list = [2]
#     dropouts = [ 0.5]
#     learning_rates = [1e-3]
#     epochs = 10
#     batch_size = 256

#     best_val_loss = float("inf")
#     best_model_state = None
#     best_hparams = {}

#     for hidden_dim, num_layers, dropout, lr in itertools.product(hidden_dims, num_layers_list, dropouts, learning_rates):
#         print(f"\nTesting configuration: hidden_dim={hidden_dim}, num_layers={num_layers}, dropout={dropout}, lr={lr}")

#         model = LSTMModel(
#             input_dim=len(LABEL_LIST),
#             hidden_dim=hidden_dim,
#             output_dim=1 if PREDICT_CORRECTNESS else len(LABEL_LIST),
#             num_layers=num_layers,
#             dropout=dropout
#         ).to(device)

#         best_sd, val_loss = train_model(
#             model,
#             train_loader,
#             val_loader,
#             True,
#             lr=lr,
#             epochs=epochs,
#             device=device
#         )

#         if val_loss < best_val_loss:
#             best_val_loss = val_loss
#             best_model_state = best_sd
#             best_hparams = {
#                 "hidden_dim": hidden_dim,
#                 "num_layers": num_layers,
#                 "dropout": dropout,
#                 "learning_rate": lr
#             }

#     print(f"\nBest Hyperparameters: {best_hparams}")
#     model.load_state_dict(best_model_state)

#     # Evaluate
#     print("Evaluating on test set...")
#     test_convos = load_and_group_conversations(test_path)
#     sequences_test, labels_test = extract_label_sequences(test_convos)
#     test_loader = create_dataloader(sequences_test, labels_test, batch_size=batch_size, shuffle=False)
#     results = evaluate_model(model, test_loader, device, PREDICT_CORRECTNESS, True)

#     if PREDICT_CORRECTNESS:
#         name = "dialogue_correcntess"
#     elif PREDICT_CURRENT:
#         name = "teacher_move_type"
#     else: 
#          name = "future_teacher_move_type"
#     output_file = "/work/pi_andrewlan_umass_edu/fikram_umass-edu/dialogue-kt/results/lstm_results.jsonl"
#     final_res = {'name':name, "res":results, "data": "anation"}
#     save_results_to_jsonl(results, output_file)
