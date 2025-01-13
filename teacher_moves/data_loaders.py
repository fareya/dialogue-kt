import json
import torch
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
import pandas as pd 

# Constants and Global Variables
RANDOM_SEED = 42
MATHDIAL_DIALOGUE_DESC = "the student is attempting to solve a math problem."
DATA_PATH = '/work/pi_juanzhai_umass_edu/fareya_workspace/dialogue-kt/teacher_moves/processed_data/train.jsonl'
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
SYSTEM_PROMPT_TEMPLATE = (
    "You are an experienced math teacher. You are given a dialogue between a student and teacher where {desc} "
    "Your job is to predict the next teacher move. There are four teacher move types: generic, focus, telling, probing. "
    "Return the next teacher move type in the dialogue."
    "The dialogue is as follows:"
)

# Read JSONL File
def read_jsonl(data_path):
    with open(data_path) as f:
        return [json.loads(line) for line in f]

def group_data_by_id(data):
    grouped_data = defaultdict(list)
    
    for entry in data:
        grouped_data[entry["id"]].append(entry)
    
    return grouped_data

def split_train_val(grouped_data):
    train_data, val_data = train_test_split(list(grouped_data.values()), test_size=0.2, random_state=RANDOM_SEED)
    return train_data, val_data

def dialogue_apply_chat_template(dialogue, labels, ids, tokenizer):
    """
    Mnemonic generation prompt. 
    Constructs the prompt for LLaMA fine-tuning using special tokens.
    """
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(desc=MATHDIAL_DIALOGUE_DESC)
    user_message = dialogue

    messages = [
        {
            "role": "system", 
            "content": system_prompt
        },
        {"role": "user", "content": user_message},
    ]
    # Ensuring that we are adding the generator prompt here 
    tokenized_chat = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")

    # print(tokenizer.decode(tokenized_chat[0]))
    return tokenizer.decode(tokenized_chat[0])


def format_dialogue(data):
    formatted_outputs = []
    labels = [] 
    ids = [] 
    for conversation in data: 
        dialogue_text = []
        for i, turn in enumerate(conversation):
            dialogue_text.append(f"\nTeacher Turn {turn['turn']}: {turn['teacher_move']} \nStudent Turn  {turn['turn']}: {turn['student_move']}")
            user_dialogue = " ".join(dialogue_text)
            user_prompt = "[BEGIN DIALOGUE]" + user_dialogue + "\n[END DIALOGUE]"       
            formatted_outputs.append(user_prompt)
            labels.append(turn["future_teacher_move_type"])
            ids.append(turn["id"])
        # if len(labels) > 6:
        #     break
    return formatted_outputs, labels, ids


class DatasetBase(Dataset):
    def __getitem__(self, index: int):
        return self.data[index]

    def __len__(self):
        return len(self.data)

class DialogueDatasetUnpacked(DatasetBase):
    def __init__(self, data: list, tokenizer, skip_first_turn: bool = False):
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME) 
        formatted_outputs, labels, ids = format_dialogue(data)
        chat_formatted_dialogue = [dialogue_apply_chat_template(formatted_outputs[i], labels[i],ids[i], tokenizer) for i in range(len(formatted_outputs))]
        self.data = []
        for i in range(len(chat_formatted_dialogue)): 
            # modify this 
            self.data.append({
                    "id": ids[i], 
                    "prompt":chat_formatted_dialogue[i],
                    "label": labels[i],
                    })
        print("Processed data")
        print(f"Number of data points: {len(self.data)}")

# class DialogueCollatorUnpacked:
#     def __init__(self, tokenizer):
#         self.tokenizer = tokenizer

#     def __call__(self, batch):
#         all_prompts = [prompt for sample in batch for prompt in sample["prompts"]]
#         prompts_tokenized = self.tokenizer(all_prompts, return_tensors="pt", padding=True).to(device)

#         return {
#             "input_ids": prompts_tokenized.input_ids,
#             "attention_mask": prompts_tokenized.attention_mask,
#             "last_idxs": prompts_tokenized.attention_mask.sum(dim=-1) - 2, # Take index of token before eos
#             "labels": torch.Tensor([sample["label"] for sample in batch]).to(device),
#             "meta_data": batch
#         }


# Use this tokenizer 
class DialogueCollatorUnpacked:
    def __init__(self, tokenizer, device):
        self.tokenizer = tokenizer
        self.device = device

    def __call__(self, batch):
        tokenizer = self.tokenizer
        device = self.device

        # Construct prompts, use output labels if given
        # has_labels = isinstance(batch[0], dict)
        # is there a better way to check if a batch has labels? 

        # print(batch[0])
        # prompts = [example["prompt"] + example["label"] + tokenizer.eos_token for example in batch] # if has_labels else batch

        prompts = [
            example["prompt"] + tokenizer.decode(tokenizer(example["label"], add_special_tokens=False).input_ids) + tokenizer.eos_token
            for example in batch
        ]

        # Batch tokenize all sequences with padding
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right" #if has_labels else "left" # Right at train-time and left at test-time
        tokenized_seqs = tokenizer(
            prompts,
            padding=True,
            return_tensors="pt",
            add_special_tokens=False, # Since apply_chat_template already adds special tokens
        ).to(device)

        input_ids = tokenized_seqs.input_ids
        attn_mask = tokenized_seqs.attention_mask
        # if not has_labels:
        #     return {
        #         "input_ids": input_ids,
        #         "attention_mask": attn_mask
        #     }

        # Create labels
        labels = input_ids.clone()
        labels[attn_mask == 0] = -100
        input_end_idxs = torch.stack([
            (seq_ids == self.tokenizer.eos_token_id).nonzero()[1] # First eos after system msg, second eos after user msg
            for seq_ids in input_ids
        ])
        label_mask = torch.arange(input_ids.shape[1]).repeat(input_ids.shape[0], 1).to(device) <= input_end_idxs
        labels[label_mask] = -100
        return {
            "input_ids": input_ids,
            "attention_mask": attn_mask,
            "labels": labels
        }

# Usage example:
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_data = read_jsonl(DATA_PATH)
    grouped_data = group_data_by_id(train_data) 
    train_data, val_data = split_train_val(grouped_data)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME) 
    train_dataset = DialogueDatasetUnpacked(train_data, tokenizer)
    collator = DialogueCollatorUnpacked(tokenizer, device) 
    train_loader = DataLoader(train_dataset, batch_size=1, collate_fn=collator)
    # for batch in train_loader:
    #     print(batch)
    #     break

# Data outputs 
# {
#     'id': '5001055_3', 
#     'prompts': ["""<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n
#                 Cutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\n
#                 You are an experienced math teacher. You are given a dialogue between a 
#                 student and teacher where the student is attempting to solve a math problem. 
#                 Your job is to predict the next teacher move. There are four teacher moves: 
#                 generic, focus, telling, probing. The dialogue is as follows:<|eot_id|>
#                 <|start_header_id|>user<|end_header_id|>\n
#                 \n[BEGIN DIALOGUE]
#                 \nTeacher Turn 0:  Hi Riya can you explain your thinking to me?
#                 \nStudent Turn  0:  Sure. I figured that since each person needed two s'mores,
#                  we would need two-thirds of a chocolate bar per person. Then I multiplied that 
#                  by 15 (the number of scouts) to get the total number of s'mores needed, which 
#                  was 30. Then I multiplied that by two-thirds again to get the number of chocolate
#                  bars needed, which was 20. Finally, I multiplied that by $1.50 (the cost of one 
#                  chocolate bar) to get the total cost, which was $30.
#                 \n[END DIALOGUE]<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n<|eot_id|>"""], 
#     'label': 'focus'
#     }
