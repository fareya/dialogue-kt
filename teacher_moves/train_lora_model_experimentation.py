import argparse
import torch
from tqdm import tqdm
from ultimate_data_loader import read_jsonl, group_data_by_id, split_train_val, DialogueDatasetUnpacked, DialogueCollatorUnpacked
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import DataLoader
from transformers import AdamW, get_scheduler

MATHDIAL = "mathdial"
ANATION = "anation"

def get_base_model(base_model_name: str, tokenizer: AutoTokenizer, quantize: bool):
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        pad_token_id=tokenizer.pad_token_id,
        quantization_config=None if not quantize else bnb_config,  # bnb_config should be defined if quantize=True
        torch_dtype=torch.float32 if quantize else torch.bfloat16,
        device_map={"": 0}
    )
    base_model.config.use_cache = False
    base_model.config.pretraining_tp = 1
    return base_model

def get_model(base_model_name: str, test: bool,
              model_name: str = None, pt_model_name: str = None,
              r: int = None, lora_alpha: int = None,
              quantize: bool = False, use_gradient_checkpointing: bool = True):
    print("Initializing tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, padding_side="right")
    tokenizer.pad_token = tokenizer.bos_token
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

def fine_tune_llama_with_lora(
    tokenizer,
    model,
    device,
    train_dataset,
    val_dataset,
    collate_fn,
    output_dir="./fine_tuned_model_with_lora",
    epochs=2,
    learning_rate=1e-4,
    batch_size=1,
    grad_accum_steps=32,
    use_lr_scheduler=False,
    wandb=None,
    early_stopping=False,
    patience=2
):
    model.train()
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    num_training_steps = len(train_dataloader) * epochs
    lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

    best_val_loss = float("inf")
    epochs_no_improvement = 0

    for epoch in range(epochs):
        total_loss = 0.0
        model.train()
        print(f"Epoch {epoch + 1}/{epochs}")
        for batch_idx, batch in tqdm(enumerate(train_dataloader), desc="Training Batches"):
            outputs = model(**batch)
            loss = outputs.loss
            total_loss += loss.item()
            loss = loss / grad_accum_steps
            loss.backward()

            if (batch_idx + 1) % grad_accum_steps == 0 or batch_idx == len(train_dataloader) - 1:
                optimizer.step()
                optimizer.zero_grad()
                if use_lr_scheduler:
                    lr_scheduler.step()

        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch + 1}/{epochs}, Average Loss: {avg_loss:.4f}")

        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for _, batch in tqdm(enumerate(val_dataloader), desc="Validation Batches"):
                outputs = model(**batch)
                total_val_loss += outputs.loss.item()

        val_loss = total_val_loss / len(val_dataloader)
        print(f"Epoch {epoch + 1}/{epochs}, Validation Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improvement = 0
            model.save_pretrained(f"{output_dir}")
            print(f"New best model saved at epoch {epoch + 1}")
        else:
            epochs_no_improvement += 1
            print(f"No validation improvement for {epochs_no_improvement} epochs.")

        if early_stopping and epochs_no_improvement >= patience:
            print(f"No validation improvement for {patience} epochs. Stopping early.")
            break

        if wandb:
            wandb.log({
                "epoch": epoch + 1,
                "average_loss": avg_loss,
                "validation_loss": val_loss
            })

    print("Finished fine-tuning.")
    return best_val_loss

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", choices=["mathdial", "anation"], required=True, help="Name of the dataset")
    parser.add_argument("--pred_label_name", type=str, required=True, help="Label to predict")
    parser.add_argument("--include_prev", action="store_true", help="Include previous turn types")

    args = parser.parse_args()

    DATASET_NAME = args.dataset_name
    PRED_LABEL_NAME = args.pred_label_name
    INCLUDE_PREV = args.include_prev

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"

    if DATASET_NAME == MATHDIAL:
        DATA_PATH = "/work/pi_andrewlan_umass_edu/fikram_umass-edu/dialogue-kt/teacher_moves/processed_data/train_check4.jsonl"
    else:
        DATA_PATH = "/work/pi_andrewlan_umass_edu/fikram_umass-edu/dialogue-kt/teacher_moves/processed_data/anation_train_data_final.jsonl"

    MODEL_SAVE_PATH = f"/work/pi_andrewlan_umass_edu/fikram_umass-edu/dialogue-kt/teacher_moves/46_runs/{MODEL_NAME}_{DATASET_NAME}_{PRED_LABEL_NAME}_include_labels_{str(INCLUDE_PREV)}"

    print(f"Using dataset: {DATASET_NAME}")
    print(f"Using data: {DATA_PATH}")
    print(f"Using model: {MODEL_NAME}")
    print(f"Using prediction label: {PRED_LABEL_NAME}")
    print(f"Using include_labels: {INCLUDE_PREV}")
    print(f"Using device: {device}")
    print(f"Saving model to: {MODEL_SAVE_PATH}")

    data = read_jsonl(DATA_PATH)
    grouped_data = group_data_by_id(data)
    train_data, val_data = split_train_val(grouped_data)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_dataset = DialogueDatasetUnpacked(train_data, tokenizer, model_name=MODEL_NAME, output_key=PRED_LABEL_NAME, include_previous_turn_types=INCLUDE_PREV, dataset_name=DATASET_NAME)
    val_dataset = DialogueDatasetUnpacked(val_data, tokenizer, model_name=MODEL_NAME, output_key=PRED_LABEL_NAME, include_previous_turn_types=INCLUDE_PREV, dataset_name=DATASET_NAME)
    collator = DialogueCollatorUnpacked(tokenizer, device)

    train_config = {
        "model_name": MODEL_NAME,
        "pt_model_name": "trainable_lora_model",
        "r": 8,
        "lora_alpha": 8,
        "epochs": 5,
        "lr": 0.0003,
        "wd": 1e-2,
        "gc": 1.0,
        "batch_size": 1,
        "grad_accum_steps": 64,
        "model_save_path": MODEL_SAVE_PATH,
    }

    model, tokenizer = get_model(
        MODEL_NAME,
        test=False,
        model_name=train_config["model_name"],
        pt_model_name=train_config["pt_model_name"],
        r=train_config["r"],
        lora_alpha=train_config["lora_alpha"]
    )
    model.to(device)

    fine_tune_llama_with_lora(
        tokenizer,
        model,
        device,
        train_dataset,
        val_dataset,
        collator,
        output_dir=train_config["model_save_path"],
        epochs=train_config["epochs"],
        learning_rate=train_config["lr"],
        batch_size=train_config["batch_size"],
        grad_accum_steps=train_config["grad_accum_steps"],
        use_lr_scheduler=False,
        wandb=None,
        early_stopping=False,
        patience=2
    )

if __name__ == "__main__":
    main()

# python /work/pi_andrewlan_umass_edu/fikram_umass-edu/dialogue-kt/teacher_moves/train_lora_model_experimentation.py --dataset_name anation --pred_label_name final_correctness --include_prev
