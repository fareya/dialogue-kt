import torch
import itertools
import pandas as pd
from ast import literal_eval
from tqdm import tqdm
from teacher_moves.data_loaders import read_jsonl, group_data_by_id, split_train_val, DialogueDatasetUnpacked, DialogueCollatorUnpacked
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import DataLoader
from transformers import AdamW
from transformers import get_scheduler
from teacher_moves.train_lora_model import get_base_model, get_model
from kt_data_loading import (LMKTDatasetUnpacked, LMKTCollatorUnpacked, LMKTDatasetPacked, LMKTCollatorPacked,
                             DKTDataset, DKTCollator, get_dataloader, apply_annotations)
from prompting import get_true_false_tokens
# from training import get_lmkt_loss_unpacked, get_lmkt_loss_packed
# from data_loading import load_annotated_data
# Read train data for bothe dialogue kt and teacher moves 
# Completely consolidated standalone functions that do their own thing and load it to the training batch 

# Next is to pick a model - it can the same for both as both use the causal LM model 
# The data loader should alternate between the two datasets when training the finetuning model 

def get_lmkt_loss_unpacked(model, batch, true_token, false_token, args):
    # Get logits at last token of each sequence
    model_output = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
    batch_size = model_output.logits.shape[0]
    logits = model_output.logits[torch.arange(batch_size), batch["last_idxs"]]
    # Return probability of True token over False token for each sequence
    logits = torch.stack([logits[:, true_token], logits[:, false_token]], dim=1)
    kc_probs = torch.softmax(logits, dim=1)[torch.arange(batch_size), 0]
    # Get probability that all KCs are True for each turn in the batch
    num_kc_counter = 0
    kc_probs_grouped = []
    corr_probs = []
    for num_kcs in batch["num_kcs"]:
        kc_probs_grouped.append(kc_probs[num_kc_counter : num_kc_counter + num_kcs].tolist())
        if args.agg == "prod":
            prob = kc_probs[num_kc_counter : num_kc_counter + num_kcs].prod()
        elif args.agg == "mean-ar":
            prob = kc_probs[num_kc_counter : num_kc_counter + num_kcs].mean()
        elif args.agg == "mean-geo":
            prob = kc_probs[num_kc_counter : num_kc_counter + num_kcs].prod() ** (1 / num_kcs)
        corr_probs.append(prob)
        num_kc_counter += num_kcs
    corr_probs = torch.stack(corr_probs)
    # Get BCE loss with correctness labels and predicted probabilities
    loss = torch.nn.BCELoss()(corr_probs, batch["labels"])
    return loss, kc_probs_grouped, corr_probs

def get_lmkt_loss_packed(model, batch, true_token, false_token, args, device):
    # Invert attention mask
    attention_mask = batch["attention_mask"]
    min_dtype = torch.finfo(model.dtype).min
    attention_mask[attention_mask == 0] = min_dtype
    attention_mask[attention_mask == 1] = 0
    attention_mask = attention_mask.type(model.dtype)
    # Get logits at last token of each sequence
    model_output = model(input_ids=batch["input_ids"], attention_mask=attention_mask, position_ids=batch["position_ids"])
    batch_size = model_output.logits.shape[0]
    logits = model_output.logits[torch.arange(batch_size).unsqueeze(1), batch["last_idxs"]]
    # Return probability of True token over False token for each sequence
    logits = torch.stack([logits[:, :, true_token], logits[:, :, false_token]], dim=2)
    kc_probs = torch.softmax(logits, dim=2)[:, :, 0]
    # Get probability that all KCs are True for each turn in the batch
    kc_probs_grouped = [probs[:num_kcs].tolist() for probs, num_kcs in zip(kc_probs, batch["num_kcs"])]
    # Set probs on padded indices
    padding_val = 0 if args.agg == "mean-ar" else 1
    kc_probs = torch.masked_scatter(kc_probs, batch["last_idxs"].to(device) == 0, torch.full_like(kc_probs, padding_val).to(device))
    # Get BCE loss with correctness labels and predicted probabilities
    if args.agg == "prod":
        corr_probs = kc_probs.prod(dim=1)
    elif args.agg == "mean-ar":
        corr_probs = kc_probs.sum(dim=1) / batch["num_kcs"]
    elif args.agg == "mean-geo":
        corr_probs = kc_probs.prod(dim=1) ** (1 / batch["num_kcs"])
    loss = torch.nn.BCELoss()(corr_probs, batch["labels"])
    return loss, kc_probs_grouped, corr_probs


def load_annotated_data(data_path):
    def pass_typical_threshold(row, typical_cutoff=1):
        # the default value of typical_cutoff is 1
        return (row["meta_data"]["self_typical_confusion"] >= typical_cutoff and
                row["meta_data"]["self_typical_interactions"] >= typical_cutoff)
    train_df = pd.read_csv(data_path, converters={col: literal_eval for col in ["dialogue", "meta_data", "annotation"]})
    train_df = train_df.sample(frac=1, random_state=221)
    train_df = train_df[train_df.apply(pass_typical_threshold, axis=1)]
    return (
        train_df[:int(.8 * len(train_df))],
        train_df[int(.8 * len(train_df)):],
    )
    raise Exception(f"Loading not supported for {args.dataset}")

def load_dialogue_kt_data(data_path, model_name, args):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    file_name = "/work/pi_andrewlan_umass_edu/fikram_umass-edu/dialogue-kt/data/annotated/mathdial_train_atc.csv"
    train_df, val_df = load_annotated_data(file_name)

    print(args)
    # train_dataset = LMKTDatasetUnpacked(train_df, tokenizer, args)
    train_dataset = LMKTDatasetPacked(train_df, tokenizer, args)
    print("Train dataset")
    print(train_dataset[:5])
    # val_dataset = LMKTDatasetUnpacked(val_df, tokenizer, args)
    val_dataset = LMKTDatasetPacked(val_df, tokenizer, args)
    # collator = LMKTCollatorUnpacked(tokenizer)
    collator = LMKTCollatorPacked(tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=1, collate_fn=collator)
    val_loader = DataLoader(val_dataset, batch_size=1, collate_fn=collator)

    # for idx, batch in enumerate(train_loader):
    #     print(batch)
    #     if idx == 5:
    #         break
    return train_loader, val_loader
    # return train_dataset, val_dataset,  

def load_teacher_moves_data(data_path, model_name, pred_label_name, device):
    data = read_jsonl(data_path)
    grouped_data = group_data_by_id(data)
    train_data, val_data = split_train_val(grouped_data)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_dataset = DialogueDatasetUnpacked(train_data, tokenizer, output_key = pred_label_name)
    val_dataset = DialogueDatasetUnpacked(val_data, tokenizer, output_key = pred_label_name)
    collator = DialogueCollatorUnpacked(tokenizer, device)
    return train_dataset, val_dataset, collator

def load_data():
    # create a data loader for dialogue kt data, both training and validation
    # create a data loader for teacher moves data, both training and validation
    # returns 4 values 
    pass

def train_model():
    # load the model 
    # load the data 
    # train the model 
    pass



def fine_tune_alternate_llama_with_lora(
    tokenizer,
    model,
    device,
    train_dataset_one,
    train_dataset_two,
    val_dataset_one,
    val_dataset_two,
    collate_fn_1,
    collate_fn_2,
    output_dir="./fine_tuned_model_with_lora",
    epochs=2,
    learning_rate=1e-4,
    batch_size=1,
    grad_accum_steps=32,
    use_lr_scheduler=False,
    wandb=None,
    early_stopping=False,
    patience=2, 
    args = None,
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
    train_dataloader_one = DataLoader(train_dataset_one, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_1)
    val_dataloader_one = DataLoader(val_dataset_one, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_1)

    # modify this code 
    train_dataloader_two = train_dataset_two # DataLoader(train_dataset_two, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_2)
    val_dataloader_two = val_dataset_two #DataLoader(val_dataset_two, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_2)

    # for idx, batch in enumerate(train_dataloader_two):
    #     print(batch)
    #     if idx == 5:
    #         break
    # Set up optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    true_token, false_token = get_true_false_tokens(tokenizer)
    # num_training_steps = len(train_dataloader) * epochs
    # lr_scheduler = get_scheduler(
    #     name="linear",
    #     optimizer=optimizer,
    #     num_warmup_steps=0,
    #     num_training_steps=num_training_steps,
    # )

    print(f"Starting LoRA fine-tuning for {epochs} epoch(s) with {len(train_dataloader_one)+len(train_dataloader_two)} samples.")

    best_val_loss = float("inf")
    epochs_no_improvement = 0

    # Training loop 
    for epoch in range(epochs):
        total_loss = 0.0
        model.train()
        print(f"Epoch {epoch + 1}/{epochs}")
        max_train_batches = max(len(train_dataloader_one), len(train_dataloader_two)) 
        total_training_batches = len(train_dataloader_one) + len(train_dataloader_two)
        iterations = 0 
        for batch_idx, (batch1, batch2) in tqdm( enumerate(itertools.zip_longest(train_dataloader_one, train_dataloader_two, fillvalue=None)), total=total_training_batches, desc="Processing Training Batches",):
            if batch1 is not None:
                # Process batch from dataloader1
                # print("Batch 1")
                # print(batch1)
                outputs = model(**batch1)
                loss = outputs.loss
                total_loss += loss.item() # Accumulate actual loss for logging. Think this was not in order.
                loss = loss / grad_accum_steps
                loss.backward()
                iterations += 1
            if batch2 is not None:
                # Process batch from dataloader2
                # print("Batch 2")
                # print(batch2)
                # print(batch2.keys())
                # print(len(batch2["input_ids"]))
                # print(batch2["input_ids"])
                # print(len(batch2["labels"]))
                # print(batch2["labels"])
                loss, _, _ = get_lmkt_loss_packed(model, batch2, true_token, false_token, args, device)
                total_loss += loss.item() 
                loss = loss / grad_accum_steps
                loss.backward()
                iterations += 1
            # using batch_idx*2 to determine when to step the optimizer, multiplyin by 2 because we are processing two batches at a time 
            if (iterations + 1) % grad_accum_steps == 0 or batch_idx == max_train_batches - 1:
                optimizer.step()
                optimizer.zero_grad()

                # if use_lr_scheduler:
                #     lr_scheduler.step()  

        avg_loss = total_loss / total_train_batches
        print(f"Epoch {epoch + 1}/{epochs}, Average Loss: {avg_loss:.4f}")


    # Training loop
    # for epoch in range(epochs):
    #     total_loss = 0.0
    #     model.train()
    #     print(f"Epoch {epoch + 1}/{epochs}")
    #     # for batch in train_dataloader:
    #     for batch_idx, batch in tqdm(enumerate(train_dataloader), desc="Training Batches"):
    #         # print(batch)
    #         outputs = model(**batch)
    #         loss = outputs.loss
    #         total_loss += loss.item() # Accumulate actual loss for logging. Think this was not in order.
    #         loss = loss / grad_accum_steps
    #         loss.backward()

    #         if (batch_idx + 1) % grad_accum_steps == 0 or batch_idx == len(train_dataloader) - 1:
    #             optimizer.step()
    #             optimizer.zero_grad()

    #             if use_lr_scheduler:
    #                 lr_scheduler.step()
                    
        

        # Validation Loop
        model.eval()
        total_val_loss = 0.0

        total_val_batches = len(val_dataloader_one) + len(val_dataloader_two)
        with torch.no_grad():
            for batch_idx, (batch1, batch2) in tqdm( enumerate(itertools.zip_longest(dataloader1, dataloader2, fillvalue=None)), total=total_val_batches, desc="Processing Validation Batches",):
                if batch1 is not None:
                    outputs = model(**batch1)
                    total_val_loss += outputs.loss.item()
                if batch2 is not None:
                    outputs = model(**batch2)
                    total_val_loss += outputs.loss.item()

        val_loss = total_val_loss /total_val_batches
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


def main():
    MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
    PRED_LABEL_NAME = "teacher_move_type"
    ACCESS_TOKEN = "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"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DATA_PATH_ONE = "/work/pi_andrewlan_umass_edu/fikram_umass-edu/dialogue-kt/teacher_moves/processed_data/train.jsonl"
    DATA_PATH_TWO = "/work/pi_andrewlan_umass_edu/fikram_umass-edu/dialogue-kt/data/annotated/mathdial_train_atc.csv"
    MODEL_SAVE_PATH = "/work/pi_andrewlan_umass_edu/fikram_umass-edu/dialogue-kt/teacher_moves/model_llama_mixed_4gpu"

    print(f"Using model: {MODEL_NAME}")
    print(f"Using teacher moves data: {DATA_PATH_ONE}")
    print(f"Using kc moves data: {DATA_PATH_TWO}")
    print(f"Saving model to: {MODEL_SAVE_PATH}")
    print(f"Using prediction label: {PRED_LABEL_NAME}")
    print(f"Using device: {device}")

    class SubstituteArgs:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
    args = SubstituteArgs(dataset = "mathdial", prompt_inc_labels=True, agg = "prod") 

    train_dataset_one, val_dataset_one, collator_one = load_teacher_moves_data(DATA_PATH_ONE, MODEL_NAME, PRED_LABEL_NAME, device)
    # train_dataset_two, val_dataset_two, collator_two = load_dialogue_kt_data(DATA_PATH_TWO, MODEL_NAME)
    train_dataset_two, val_dataset_two = load_dialogue_kt_data(DATA_PATH_TWO, MODEL_NAME, args)

    # for idx, batch in enumerate(train_dataset_two):
        # print(batch)
        # if idx == 5:
        #     break

    train_config = {
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


    fine_tune_alternate_llama_with_lora(
        tokenizer,
        model,
        device,
        train_dataset_one,
        train_dataset_two,
        val_dataset_one,
        val_dataset_two,
        collator_one,
        None,
        output_dir=train_config["model_save_path"],
        epochs=train_config["epochs"],
        learning_rate=train_config["lr"],
        batch_size=train_config["batch_size"],
        grad_accum_steps=train_config["grad_accum_steps"],
        use_lr_scheduler=False,
        wandb=None,
        early_stopping=False,
        patience=2, 
        args = args,
    )

if __name__ == "__main__":
    main()