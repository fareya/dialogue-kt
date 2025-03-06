# import torch
# from tqdm import tqdm
# from data_loaders import read_jsonl, group_data_by_id, get_test_formatted, split_train_val, DialogueDatasetUnpacked, DialogueCollatorUnpacked
# from transformers import AutoModelForCausalLM, AutoTokenizer
# from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
# from torch.utils.data import Dataset, DataLoader
# from torch.utils.data import DataLoader
# from transformers import AdamW
# from transformers import get_scheduler
# from sklearn.metrics import f1_score
# from huggingface_hub import login
# import numpy as np 

# MATHDIAL_DIALOGUE_DESC = "the student is attempting to solve a math problem."
# DATA_PATH = '/work/pi_andrewlan_umass_edu/fikram_umass-edu/dialogue-kt/teacher_moves/processed_data/test_check.jsonl'
# MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"

# def get_base_model(base_model_name: str, tokenizer: AutoTokenizer, quantize: bool):
#     base_model = AutoModelForCausalLM.from_pretrained(
#         base_model_name,
#         pad_token_id=tokenizer.pad_token_id
#     )
#     base_model.config.use_cache = False
#     base_model.config.pretraining_tp = 1 # What is this?
#     return base_model

# #copied over and modified from lm.py
# def get_model(base_model_name: str, test: bool,
#               model_name: str = None, pt_model_name: str = None,
#               r: int = None, lora_alpha: int = None,
#               quantize: bool = True, use_gradient_checkpointing: bool = True):
#     print("Initializing tokenizer")
#     tokenizer = AutoTokenizer.from_pretrained(base_model_name, padding_side="right")
#     tokenizer.pad_token = tokenizer.bos_token # Have to pick some token, and eos triggers a warning
#     # retreive the base model
#     print("Initializing model")
#     model = get_base_model(base_model_name, tokenizer, quantize)
#     return model, tokenizer

# def compute_metrics(preds, labels):
#     # compute the accuracy
#     count = 0
#     for i in range(len(preds)):
#         if preds[i] == labels[i]:
#             count += 1
#     accuracy = count / len(preds)
#     f1 = f1_score(labels, preds, average="macro")
#     return {"accuracy": accuracy, "f1": f1}

# import numpy as np
# from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# # Replace these with your actual predictions and ground truth labels
# # y_true: Ground truth labels
# # y_pred: Model predictions
# # Example:
# # y_true = np.array([0, 1, 2, 2, 0, 1])
# # y_pred = np.array([0, 1, 2, 1, 0, 0])

# # Define your class names (for clarity)

# def true_positive_true_negative(y_true, y_pred):
#     class_names = ['generic', 'focus', 'telling', 'probing']
#     tp = {'generic': 0, 'focus': 0, 'telling': 0, 'probing': 0}
#     fn = {'generic': 0, 'focus': 0, 'telling': 0, 'probing': 0}
#     fp = {'generic': 0, 'focus': 0, 'telling': 0, 'probing': 0}
#     misclassifications = {'generic': [], 'focus': [], 'telling': [], 'probing': []}
#     for i in range(len(y_true)):  
#         try:
#             assert y_pred[i] in class_names
#             assert y_true[i] in class_names
#         except:
#             print(f"y_pred[i]: {y_pred[i]}")
#             print(f"y_true[i]: {y_true[i]}")
#             continue
#         if y_true[i] == y_pred[i]:
#             tp[y_true[i]] += 1
#         else:
#             fn[y_true[i]] += 1
#             fp[y_pred[i]] += 1
#             misclassifications[y_true[i]].append(y_pred[i])
#     # print the general data distribution
#     print("True Positives: ", tp)
#     print("False Negatives: ", fn)
#     print("False Positives: ", fp)
#     print("Misclassifications: ", misclassifications)

# def test_lora_model():
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     print(f"Using device: {device}")
#     data = read_jsonl(DATA_PATH)
#     grouped_data = group_data_by_id(data)
#     test_data = get_test_formatted(grouped_data)
#     # print(grouped_data)

#     PRED_LABEL_NAME = "teacher_move_type"
#     print(f"Using label: {PRED_LABEL_NAME}")
#     print(f"Model: {MODEL_NAME}")   
#     tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
#     test_dataset = DialogueDatasetUnpacked(test_data, tokenizer, model_name = MODEL_NAME, output_key=PRED_LABEL_NAME)
#     collator = DialogueCollatorUnpacked(tokenizer, device, is_test=True)
#     test_loader = DataLoader(test_dataset, batch_size=1, collate_fn=collator)
#     # first read the base model and tokenizer 
    

#     model_config = {
#         "model_name": MODEL_NAME,
#         "pt_model_name": "trainable_lora_model",
#         "r": 16,
#         "lora_alpha": 16,
#         "epochs": 5,
#         "lr": 2e-4,
#         "wd": 1e-2,
#         "gc": 1.0,
#         "batch_size": 1,
#         "grad_accum_steps": 64,
#     }

#     model, tokenizer = get_model(
#         MODEL_NAME,
#         test=False,
#         model_name=model_config["model_name"],
#         pt_model_name=model_config["pt_model_name"],
#         r=model_config["r"],
#         lora_alpha=model_config["lora_alpha"]
#     )

#     # lora_model_path = "/work/pi_andrewlan_umass_edu/fikram_umass-edu/dialogue-kt/saved_models/model_llama32bv2_129"
#     # lora_model_path= "/work/pi_andrewlan_umass_edu/fikram_umass-edu/dialogue-kt/teacher_moves/model_llama32bv2_5"
#     lora_model_path = "/work/pi_andrewlan_umass_edu/fikram_umass-edu/dialogue-kt/teacher_moves/model_llama32bv2_5_lr_0.0003_r_16"
#     print(f"Loading model from {lora_model_path}")
#     peft_model = PeftModel.from_pretrained(model, model_id=lora_model_path, is_trainable=False)
#     peft_model.to("cuda")
#     peft_model.eval()


#     predictions = []
#     labels = []
#     # now doing inference
#     for batch in test_loader:
#         input_ids = batch["input_ids"].to("cuda")
#         attn_mask = batch["attention_mask"].to("cuda")
#         batch_labels = batch["labels"]
#         with torch.no_grad():
#             # outputs = peft_model(input_ids, attention_mask=attn_mask, labels=labels)
#             # Need to do greedy 
#             model_out = peft_model.generate(input_ids, 
#                                 attention_mask=attn_mask, 
#                                 pad_token_id=tokenizer.pad_token_id, 
#                                 do_sample=False, # this ensure greedy decoding 
#                                 max_new_tokens=1000)
           
#             # just look at the first output
#             new_tokens = model_out[:, batch["input_ids"].shape[-1]:]
#             prediction_text = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
#             for i in range(len(prediction_text)):
#                 print(f"Prediction: {prediction_text[i]}")
#                 print(f"Label: {batch_labels[i]}")
#             predictions.extend(prediction_text)
#             labels.extend(batch_labels)
#     metrics = compute_metrics(predictions, labels)
#     print(metrics)
#     true_positive_true_negative(labels, predictions)

# test_lora_model()

import torch
from tqdm import tqdm
from data_loaders import read_jsonl, group_data_by_id, get_test_formatted, split_train_val, DialogueDatasetUnpacked, DialogueCollatorUnpacked
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import DataLoader
from transformers import AdamW
from transformers import get_scheduler
from sklearn.metrics import f1_score
import numpy as np

MATHDIAL_DIALOGUE_DESC = "the student is attempting to solve a math problem."
DATA_PATH = '/work/pi_andrewlan_umass_edu/fikram_umass-edu/dialogue-kt/teacher_moves/processed_data/test_check4.jsonl'
MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"

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

def test_lora_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    data = read_jsonl(DATA_PATH)
    grouped_data = group_data_by_id(data)
    test_data = get_test_formatted(grouped_data)

    PRED_LABEL_NAME = "teacher_move_type"
    print(f"Using label: {PRED_LABEL_NAME}")
    print(f"Model: {MODEL_NAME}")   
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    test_dataset = DialogueDatasetUnpacked(test_data, tokenizer, model_name=MODEL_NAME, output_key=PRED_LABEL_NAME)
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

    # lora_model_path = "/work/pi_andrewlan_umass_edu/fikram_umass-edu/dialogue-kt/saved_models/model_llama32bv2_5_lr_0.0003_r_16"
    lora_model_path = "/work/pi_andrewlan_umass_edu/fikram_umass-edu/dialogue-kt/teacher_moves/model_llama32b_224"
    lora_model_path = 
    print(f"Loading model from {lora_model_path}")
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
                print(f"Prediction: {prediction_text[i]}")
                print(f"Label: {batch_labels[i]}")

            predictions.extend(prediction_text)
            labels.extend(batch_labels)

    metrics = compute_metrics(predictions, labels)
    print(metrics)

    # Call function to analyze misclassifications
    true_positive_true_negative(labels, predictions, decoded_inputs)

test_lora_model()
