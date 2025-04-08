import torch
from tqdm import tqdm
from ultimate_data_loader import read_jsonl, group_data_by_id, get_test_formatted, split_train_val, DialogueDatasetUnpacked, DialogueCollatorUnpacked
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import DataLoader
from transformers import AdamW
from transformers import get_scheduler
from sklearn.metrics import f1_score
import numpy as np
import argparse

import ast
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score, accuracy_score, hamming_loss

MATHDIAL_DIALOGUE_DESC = "the student is attempting to solve a math problem."
# DATA_PATH = '/work/pi_andrewlan_umass_edu/fikram_umass-edu/dialogue-kt/teacher_moves/processed_data/test_check4.jsonl'
# DATA_PATH = "/work/pi_andrewlan_umass_edu/fikram_umass-edu/dialogue-kt/teacher_moves/processed_data/anation_val_data.jsonl"
DATA_PATH = "/work/pi_andrewlan_umass_edu/fikram_umass-edu/dialogue-kt/teacher_moves/processed_data/test_with_gpt_labels.jsonl"
MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
MATHDIAL = "mathdial"
ANATION = "anation"

def get_base_model(base_model_name: str, tokenizer: AutoTokenizer, quantize: bool):
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        pad_token_id=tokenizer.pad_token_id
    )
    base_model.config.use_cache = False
    base_model.config.pretraining_tp = 1 # What is this?
    return base_model

#copied over and modified from lm.py
def get_model(base_model_name: str, test: bool,
              model_name: str = None, pt_model_name: str = None,
              r: int = None, lora_alpha: int = None,
              quantize: bool = True, use_gradient_checkpointing: bool = True):
    print("Initializing tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, padding_side="right")
    tokenizer.pad_token = tokenizer.bos_token # Have to pick some token, and eos triggers a warning
    # retreive the base model
    print("Initializing model")
    model = get_base_model(base_model_name, tokenizer, quantize)
    return model, tokenizer

def compute_metrics(preds, labels):
    # compute the accuracy
    count = 0
    for i in range(len(preds)):
        if preds[i] == labels[i]:
            count += 1
    accuracy = count / len(preds)
    f1 = f1_score(labels, preds, average="macro")
    return {"accuracy": accuracy, "f1": f1}

def compute_metrics(preds, labels):
    # compute the accuracy
    count = 0
    for i in range(len(preds)):
        if preds[i] == labels[i]:
            count += 1
    accuracy = count / len(preds)
    f1 = f1_score(labels, preds, average="macro")
    return {"accuracy": accuracy, "f1": f1}

def evaluate_multi_label(y_true_raw, y_pred_raw):
    """
    y_true_raw: list of strings (e.g., "['label1', 'label2']")
    y_pred_raw: list of lists (e.g., [['label1'], ['label2', 'label3']])
    all_labels: list of all possible labels
    """


    all_labels = [
        'confirmatory feedback', 'negative feedback', 'correcting',
        'giving instruction', 'giving explanation',
        'providing further references', 'questioning', 'asking for elaboration',
        'praising and encouraging', 'managing frustration',
        'managing discussions', 'giving answers', 'encouraging peer tutoring',
        'guiding peer tutoring', 'acknowledging tutor issue', 'other',
        'irrelevant statement', 'computational skill', 'linguistic knowledge',
        'conceptual knowledge', 'strategic knowledge', 'affective control'
    ]
    # Convert stringified lists into actual Python lists
    y_true = [ast.literal_eval(item) if isinstance(item, str) else item for item in y_true_raw]

    # Initialize binarizer with fixed label set for consistent vectorization
    mlb = MultiLabelBinarizer(classes=all_labels)
    mlb.fit(y_true + y_pred_raw)  # Fit once

    y_true_bin = mlb.transform(y_true)
    y_pred_bin = mlb.transform(y_pred_raw)

    f1 = f1_score(y_true_bin, y_pred_bin, average='samples')
    exact_match_acc = accuracy_score(y_true_bin, y_pred_bin)
    hamming_acc = 1 - hamming_loss(y_true_bin, y_pred_bin)

    print(f"F1 Score (samples): {f1:.4f}")
    print(f"Exact Match Accuracy: {exact_match_acc:.4f}")
    print(f"Hamming Accuracy: {hamming_acc:.4f}")
    
    return f1, exact_match_acc, hamming_acc

def evaluate_multi_label_safe(y_true_raw, y_pred_raw):
    import ast
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

    total = 0 
    for i in range(len(y_true)): 
        # if y_true[i] == []: 
        #     y_true[i] == ['none']
        # if y_pred[i] == []:
        #     y_pred[i] = ['none']
        if y_pred[i] == y_true[i]:
            print(i)
            print(y_pred[i])
            total = total + 1 
    print(y_true)
    print(y_pred)

    print(total/ len(y_true))
    # Initialize binarizer with fixed class order
    mlb = MultiLabelBinarizer()
    mlb.fit(y_true + y_pred)
    # mlb.fit([])

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

    return f1, exact_match_acc, hamming_acc


def true_positive_true_negative(y_true, y_pred, test_data):
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
                    "input_data": test_data[i]  # Store input for reference
                })

    # Print results
    print("True Positives: ", tp)
    print("False Negatives: ", fn)
    print("False Positives: ", fp)
    
    # Print misclassified examples
    print("\nMisclassified Examples (Up to 5 per class):")
    for cls, examples in misclassified_examples.items():
        print(f"\nClass: {cls} (Misclassified Examples)")
        for ex in examples:
            print(f"- True: {ex['true_label']}, Predicted: {ex['predicted_label']}")
            print(f"  Input Data: {ex['input_data']}\n")  # Print input for debugging

def test_lora_model(dataset_name, pred_label_name, include_prev):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    DATASET_NAME = dataset_name
    PRED_LABEL_NAME = pred_label_name # options: future_teacher_move_type, teacher_move_type, correctness_annotation, future_correctness_annotation, final_correctness
    MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
    INCLUDE_PREV = include_prev

    if DATASET_NAME == MATHDIAL:
        DATA_PATH = "/work/pi_andrewlan_umass_edu/fikram_umass-edu/dialogue-kt/teacher_moves/processed_data/test_check4.jsonl"
    if DATASET_NAME == ANATION:
        DATA_PATH = "/work/pi_andrewlan_umass_edu/fikram_umass-edu/dialogue-kt/teacher_moves/processed_data/anation_val_data_final.jsonl"

    print(f"Using dataset: {DATASET_NAME}")
    print(f"Using data: {DATA_PATH}")
    print(f"Using model: {MODEL_NAME}")
    print(f"Using prediction label: {PRED_LABEL_NAME}")
    print(f"Using include_labels: {INCLUDE_PREV}")    
    print(f"Using device: {device}")

    lora_model_path = "/work/pi_andrewlan_umass_edu/fikram_umass-edu/dialogue-kt/teacher_moves/46_runs/"+MODEL_NAME+"_"+DATASET_NAME+"_"+PRED_LABEL_NAME+"_include_labels_"+str(INCLUDE_PREV)
    print(f"Trained Model:{lora_model_path}")
    data = read_jsonl(DATA_PATH)
    grouped_data = group_data_by_id(data)
    test_data = get_test_formatted(grouped_data)

    print(f"Using label: {PRED_LABEL_NAME}")
    print(f"Model: {MODEL_NAME}")   
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    test_dataset = DialogueDatasetUnpacked(test_data, tokenizer, output_key = PRED_LABEL_NAME, include_previous_turn_types=INCLUDE_PREV, dataset_name = DATASET_NAME)
    collator = DialogueCollatorUnpacked(tokenizer, device, is_test=True)
    test_loader = DataLoader(test_dataset, batch_size=1, collate_fn=collator)

    model_config = {
        "model_name": MODEL_NAME,
        "pt_model_name": "trainable_lora_model",
        "r": 16,
        "lora_alpha": 16,
        "epochs": 5,
        "lr": 2e-4,
        "wd": 1e-2,
        "gc": 1.0,
        "batch_size": 1,
        "grad_accum_steps": 64,
    }

    model, tokenizer = get_model(
        MODEL_NAME,
        test=False,
        model_name=model_config["model_name"],
        pt_model_name=model_config["pt_model_name"],
        r=model_config["r"],
        lora_alpha=model_config["lora_alpha"]
    )
    
    peft_model = PeftModel.from_pretrained(model, model_id=lora_model_path, is_trainable=False)
    peft_model.to("cuda")
    peft_model.eval()

    predictions = []
    labels = []

    # Store raw inputs for debugging misclassifications
    raw_inputs = []
    decoded_inputs = []
    # Now doing inference
    for batch in test_loader:
        input_ids = batch["input_ids"].to("cuda")
        attn_mask = batch["attention_mask"].to("cuda")
        batch_labels = batch["labels"]
        raw_inputs.append(batch)  # Store full batch input for debugging
        decoded_inputs.append(tokenizer.batch_decode(input_ids, skip_special_tokens=True))
        with torch.no_grad():
            model_out = peft_model.generate(
                input_ids, 
                attention_mask=attn_mask, 
                pad_token_id=tokenizer.pad_token_id, 
                do_sample=False,  # Greedy decoding
                max_new_tokens=1000
            )

            new_tokens = model_out[:, batch["input_ids"].shape[-1]:]
            prediction_text = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)

            for i in range(len(prediction_text)):
                print(f"Input: {decoded_inputs[-1]}")
                print(f"Prediction: {prediction_text[i]}")
                print(f"Label: {batch_labels[i]}")

            predictions.extend(prediction_text)
            labels.extend(batch_labels)


    metrics = compute_metrics(predictions, labels)
    print(metrics)
    # if PRED_LABEL_NAME == "teacher_move_type" or PRED_LABEL_NAME == "future_teacher_move_type":
    #     evaluate_multi_label_safe(labels, predictions)
    # Call function to analyze misclassifications
    # true_positive_true_negative(labels, predictions, decoded_inputs)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, choices=["mathdial", "anation"], required=True)
    parser.add_argument("--pred_label_name", type=str, required=True)
    parser.add_argument("--include_prev", action="store_true")
    args = parser.parse_args()

    test_lora_model(args.dataset_name, args.pred_label_name, args.include_prev)