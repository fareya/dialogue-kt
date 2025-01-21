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
from huggingface_hub import login

MATHDIAL_DIALOGUE_DESC = "the student is attempting to solve a math problem."
DATA_PATH = '/work/pi_andrewlan_umass_edu/fikram_umass-edu/dialogue-kt/teacher_moves/processed_data/test.jsonl'
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
    print("Initializing trainable model with new LoRA adapters")
    peft_config = LoraConfig(
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
        inference_mode=False,
        )
    model = get_peft_model(model, peft_config)
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

def test_lora_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    data = read_jsonl(DATA_PATH)
    grouped_data = group_data_by_id(data)
    test_data = get_test_formatted(grouped_data)
    # print(grouped_data)

    PRED_LABEL_NAME = "teacher_move_type"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    test_dataset = DialogueDatasetUnpacked(test_data, tokenizer, output_key=PRED_LABEL_NAME)
    collator = DialogueCollatorUnpacked(tokenizer, device, is_test=True)
    test_loader = DataLoader(test_dataset, batch_size=1, collate_fn=collator)
    # first read the base model and tokenizer 
    

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

    # now read the PEFT model 
    # lora_model_path = "/work/pi_andrewlan_umass_edu/fikram_umass-edu/dialogue-kt/teacher_moves/model_checkpoints/"
    # lora_model_path="/work/pi_andrewlan_umass_edu/fikram_umass-edu/dialogue-kt/teacher_moves/model_llama32b"
    lora_model_path="/work/pi_andrewlan_umass_edu/fikram_umass-edu/dialogue-kt/teacher_moves/model_llama32"
   
    print(f"Loading model from {lora_model_path}")
    peft_model = PeftModel.from_pretrained(model, model_id=lora_model_path, is_trainable=False)
    peft_model.to("cuda")
    peft_model.eval()


    predictions = []
    labels = []
    # now doing inference
    for batch in test_loader:
        input_ids = batch["input_ids"].to("cuda")
        attn_mask = batch["attention_mask"].to("cuda")
        batch_labels = batch["labels"]
        with torch.no_grad():
            # outputs = peft_model(input_ids, attention_mask=attn_mask, labels=labels)
            model_out = peft_model.generate(input_ids, 
                                attention_mask=attn_mask, 
                                pad_token_id=tokenizer.pad_token_id, 
                                max_new_tokens=1000)
           
            # just look at the first output
            new_tokens = model_out[:, batch["input_ids"].shape[-1]:]
            prediction_text = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
            for i in range(len(prediction_text)):
                print(f"Prediction: {prediction_text[i]}")
                print(f"Label: {batch_labels[i]}")
            predictions.extend(prediction_text)
            labels.extend(batch_labels)
    metrics = compute_metrics(predictions, labels)
    print(metrics)

test_lora_model()

