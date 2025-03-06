import torch
from tqdm import tqdm
from data_loaders_5 import read_jsonl, group_data_by_id, split_train_val, DialogueDatasetUnpacked, DialogueCollatorUnpacked
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import DataLoader
from transformers import AdamW
from transformers import get_scheduler
import itertools



def get_base_model(base_model_name: str, tokenizer: AutoTokenizer, quantize: bool):
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        pad_token_id=tokenizer.pad_token_id,
        quantization_config=bnb_config if quantize else None,
        # f32 seems helpful for train/test time consistency when quantizing, bf16 performs best for non-quantized
        torch_dtype=torch.float32 if quantize else torch.bfloat16,
        device_map={"": 0}
    )
    base_model.config.use_cache = False
    base_model.config.pretraining_tp = 1
    return base_model

#copied over and modified from lm.py
def get_model(base_model_name: str, test: bool,
              model_name: str = None, pt_model_name: str = None,
              r: int = None, lora_alpha: int = None,
              quantize: bool = False, use_gradient_checkpointing: bool = True):
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

# def get_class_tokens(tokenizer, class_labels = ["generic", "focus", "telling", "probing"]):
#     """
#     Returns the token IDs for all class labels.
#     Args:
#         tokenizer: The tokenizer for the language model.
#         class_labels: A list of class labels (strings).
#     Returns:
#         class_tokens: A dictionary mapping class labels to their token IDs.
#     """
#     class_tokens = {
#         label: tokenizer(label, return_tensors="pt").input_ids.squeeze()
#         for label in class_labels
#     }
#     return class_tokens


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
    """
    Fine-tunes a LLaMA model using LoRA for efficient adaptation.

    Args:
        model_name (str): The base model name (e.g., 'meta-llama/Meta-Llama-3.1-8B-Instruct').
        tokenizer (transformers.AutoTokenizer): Tokenizer corresponding to the base model.
        model (transformers.AutoModelForCausalLM): The pre-trained LLaMA model.
        dataset (List[Dict[str, str]]): The fine-tuning dataset with 'input' and 'output' pairs.
        output_dir (str, optional): Directory to save the fine-tuned model. Defaults to './fine_tuned_model_with_lora'.
        lora_r (int, optional): Rank of the LoRA adapter matrices.
        lora_alpha (int, optional): Scaling factor for LoRA.
        lora_dropout (float, optional): Dropout rate for LoRA layers.
        task_type (TaskType, optional): The task type for LoRA. Defaults to CAUSAL_LM.
        epochs (int, optional): Number of epochs to fine-tune. Defaults to 1.
        learning_rate (float, optional): Learning rate for the optimizer. Defaults to 5e-5.

    Returns:
        None: Saves the fine-tuned model to the specified output directory.
    """

    # Prepare data
    model.train()
    # collator = (tokenizer, device)

    # modify this code 
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # Set up optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    num_training_steps = len(train_dataloader) * epochs
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    print(f"Starting LoRA fine-tuning for {epochs} epoch(s) with {len(train_dataset)} samples.")

    best_val_loss = float("inf")
    epochs_no_improvement = 0

    # Training loop
    for epoch in range(epochs):
        total_loss = 0.0
        model.train()
        print(f"Epoch {epoch + 1}/{epochs}")
        # for batch in train_dataloader:
        for batch_idx, batch in tqdm(enumerate(train_dataloader), desc="Training Batches"):
            # print(batch)
            outputs = model(**batch)
            loss = outputs.loss
            total_loss += loss.item() # Accumulate actual loss for logging. Think this was not in order.
            loss = loss / grad_accum_steps
            loss.backward()

            if (batch_idx + 1) % grad_accum_steps == 0 or batch_idx == len(train_dataloader) - 1:
                optimizer.step()
                optimizer.zero_grad()

                if use_lr_scheduler:
                    lr_scheduler.step()
                    
        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch + 1}/{epochs}, Average Loss: {avg_loss:.4f}")

        # Validation
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for batch_idx, batch in tqdm(enumerate(val_dataloader), desc="Validation Batches"):
                outputs = model(**batch)
                total_val_loss += outputs.loss.item()

        val_loss = total_val_loss / len(val_dataloader)
        print(f"Epoch {epoch + 1}/{epochs}, Validation Loss: {val_loss:.4f}")

        # Early stopping check: if no improvement, increment counter
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improvement = 0

            # Save the best model so far
            model.save_pretrained(f"{output_dir}")
            print(f"New best model saved at epoch {epoch + 1}")
        else:
            epochs_no_improvement += 1
            print(f"No validation improvement for {epochs_no_improvement} epochs.")

        # Early stopping condition
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
    # DATA_PATH = '/work/pi_juanzhai_umass_edu/fareya_workspace/dialogue-kt/teacher_moves/processed_data/train.jsonl'
    DATA_PATH = "/work/pi_andrewlan_umass_edu/fikram_umass-edu/dialogue-kt/teacher_moves/processed_data/train_check4.jsonl"
    print(f"Using data: {DATA_PATH}")
    MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
    # MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
    print(f"Using model: {MODEL_NAME}")
    MODEL_SAVE_PATH = "/work/pi_andrewlan_umass_edu/fikram_umass-edu/dialogue-kt/teacher_moves/model_llama32bv2_227_future_correctness_with_prev_moves"
    print(f"Saving model to: {MODEL_SAVE_PATH}")
    ACCESS_TOKEN = "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"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # PRED_LABEL_NAME = "future_teacher_move_type"
    
    PRED_LABEL_NAME = "future_correctness_annotation"
    print(f"Using prediction label: {PRED_LABEL_NAME}")
    print(f"Using device: {device}")

    data = read_jsonl(DATA_PATH)
    grouped_data = group_data_by_id(data)
    train_data, val_data = split_train_val(grouped_data)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_dataset = DialogueDatasetUnpacked(train_data, tokenizer, model_name = MODEL_NAME, output_key = PRED_LABEL_NAME)
    val_dataset = DialogueDatasetUnpacked(val_data, tokenizer, model_name = MODEL_NAME, output_key = PRED_LABEL_NAME)
    collator = DialogueCollatorUnpacked(tokenizer, device)
 
    #
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

    print("Model Configuration:")
    print(train_config)

    model, tokenizer = get_model(
        MODEL_NAME,
        test=False,
        model_name=train_config["model_name"],
        pt_model_name=train_config["pt_model_name"],
        r=train_config["r"],
        lora_alpha=train_config["lora_alpha"]
    )
    model.to(device)  # Move model to device


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

def hyperparam_sweep():
    DATA_PATH = "/work/pi_andrewlan_umass_edu/fikram_umass-edu/dialogue-kt/teacher_moves/processed_data/train_check.jsonl"
    MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
    MODEL_SAVE_PATH = "/work/pi_andrewlan_umass_edu/fikram_umass-edu/dialogue-kt/teacher_moves/model_llama32bv2_25_2_4"
    PRED_LABEL_NAME = "teacher_move_type"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using data: {DATA_PATH}")
    print(f"Using model: {MODEL_NAME}")
    print(f"Using device: {device}")
    print(f"Using prediction label: {PRED_LABEL_NAME}")
    print(f"Using device: {device}")

    data = read_jsonl(DATA_PATH)
    grouped_data = group_data_by_id(data)
    train_data, val_data = split_train_val(grouped_data)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_dataset = DialogueDatasetUnpacked(train_data, tokenizer, model_name = MODEL_NAME, output_key = PRED_LABEL_NAME)
    val_dataset = DialogueDatasetUnpacked(val_data, tokenizer, model_name = MODEL_NAME, output_key = PRED_LABEL_NAME)
    collator = DialogueCollatorUnpacked(tokenizer, device)

    # Define hyperparameter grid
    learning_rates = [5e-5, 1e-4, 2e-4, 3e-4]
    lora_ranks = [4, 8, 16, 32]
    best_val_loss = float("inf")
    best_config = None
    results = []

    for lr, r in itertools.product(learning_rates, lora_ranks):
        print(f"Training with lr={lr}, r={r}")
        save_path = MODEL_SAVE_PATH + f"_lr_{lr}_r_{r}"
        # Update training configuration
        train_config = {
            "model_name": MODEL_NAME,
            "pt_model_name": "trainable_lora_model",
            "r": r,
            "lora_alpha": r,  # Keeping alpha same as r, can be changed
            "epochs": 3,
            "lr": lr,
            "wd": 1e-2,
            "gc": 1.0,
            "batch_size": 1,
            "grad_accum_steps": 64,
            "model_save_path": save_path,
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

        val_loss = fine_tune_llama_with_lora(
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

        # Store results
        results.append({"lr": lr, "r": r, "val_loss": val_loss})

        # Track best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_config = train_config
            print(f"New best model with val_loss={val_loss} for lr={lr}, r={r}")

    print("Hyperparameter tuning completed.")
    print(f"Best config: {best_config} with val_loss={best_val_loss}")
    return results, best_config

if __name__ == "__main__":
    main()
    # hyperparam_sweep()