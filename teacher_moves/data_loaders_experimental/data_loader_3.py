# Goal: To create a dataloader for the teacher moves dataset 
# Labels: teacher move type, and input will be just the teacher move 

import json
import torch
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
import pandas as pd 

# Constants and Global Variables
RANDOM_SEED = 42

# PROMPT SETUP
MATHDIAL_DIALOGUE_DESC = "the student is attempting to solve a math problem."
SYSTEM_PROMPT_TEMPLATE= (
    "You are an experienced math teacher. You are given teacher dialogue {desc} "
    "Your job is to detect the teacher move type given teacher moves. There are four teacher moves: generic, focus, telling, probing."
    "The dialogue is as follows:"

)

# Read JSONL File
def read_jsonl(data_path):
    with open(data_path) as f:
        return [json.loads(line) for line in f]

def group_data_by_id(data):
    # add all ids to a list
    grouped_data = defaultdict(list)
    for entry in data:
        grouped_data[entry["id"]].append(entry)
    return grouped_data

def split_train_val(grouped_data):
    train_data, val_data = train_test_split(list(grouped_data.values()), test_size=0.2, random_state=RANDOM_SEED)
    return train_data, val_data

def get_test_formatted(grouped_data):
    test_data = list(grouped_data.values())
    return test_data

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
    tokenized_chat = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) #, return_tensors="pt")

    # print(tokenizer.decode(tokenized_chat[0]))
    return tokenized_chat


# TODO: Fix this for when we provide last teacher turn 
def format_dialogue(data, pred_label_name):
    formatted_outputs = []
    labels = [] 
    ids = [] 
    print("Length of data: ", len(data))
    for conversation in data: 
        dialogue_text = []
        for i, turn in enumerate(conversation):
            labels.append(turn["teacher_move_type"])
            formatted_outputs.append(turn["teacher_move"])
            ids.append(turn["id"])
            

    print("Length of formatted outputs: ", len(formatted_outputs))
    
    for i in range(15):
        print("Formatted outputs: ", formatted_outputs[i])
        print("Labels: ", labels[i])
        print("IDs: ", ids[i])


    assert len(formatted_outputs) == len(labels)
    return formatted_outputs, labels, ids


class DatasetBase(Dataset):
    def __getitem__(self, index: int):
        return self.data[index]

    def __len__(self):
        return len(self.data)

class DialogueDatasetUnpacked(DatasetBase):
    def __init__(self, data: list, tokenizer, skip_first_turn: bool = False, model_name: str = None, output_key: str = None):
        tokenizer = AutoTokenizer.from_pretrained(model_name) 
        # print(OUTPUT_KEY)
        # print(output_key)
        
        # assert output_key == OUTPUT_KEY
        formatted_outputs, labels, ids = format_dialogue(data, output_key)
        print("Formatted data")
        # for i in range(50):
        #     print(formatted_outputs[i])
        #     print(labels[i])
        chat_formatted_dialogue = [dialogue_apply_chat_template(formatted_outputs[i], labels[i],ids[i], tokenizer) for i in range(len(formatted_outputs))]
        self.data = []
        self.data_len = []
        for i in range(len(chat_formatted_dialogue)): 
            # modify this 
            if len(chat_formatted_dialogue[i]) < 7500:
                self.data_len.append(len(chat_formatted_dialogue[i]))
                self.data.append({
                        "id": ids[i], 
                        "prompt":chat_formatted_dialogue[i],
                        "label": labels[i],
                        })
        print("Processed data")
        print(f"Number of data points: {len(self.data)}")
        #Average length of the data points
        print(f"Average length of the data points: {sum(self.data_len)/len(self.data_len)}")
        # Maximum length of the data points
        print(f"Maximum length of the data points: {max(self.data_len)}")

# Use this tokenizer 
class DialogueCollatorUnpacked:
    def __init__(self, tokenizer, device, is_test=False):
        self.tokenizer = tokenizer
        self.device = device
        self.is_test = is_test
    def __call__(self, batch):
        tokenizer = self.tokenizer
        device = self.device
        is_test = self.is_test

        if is_test:
            prompts = [
                example["prompt"] 
                for example in batch
            ]
        else:
            # todo: rerun for train 
            prompts = [
                example["prompt"] + example["label"] + tokenizer.eos_token
                for example in batch
            ]

        # Batch tokenize all sequences with padding
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right" if not is_test else "left" # Right at train-time and left at test-time
        
        tokenized_seqs = tokenizer(
            prompts,
            padding=True,
            return_tensors="pt",
            add_special_tokens=False, # Since apply_chat_template already adds special tokens
        ).to(device)

        input_ids = tokenized_seqs.input_ids
        attn_mask = tokenized_seqs.attention_mask
        if is_test:
            return {
                "input_ids": input_ids,
                "attention_mask": attn_mask,
                "labels": [example['label'] for example in batch] 
            }

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
    output_key = "teacher_move_type"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_path = "/work/pi_andrewlan_umass_edu/fikram_umass-edu/dialogue-kt/teacher_moves/processed_data/test_check.jsonl"
    incoming_test_data = read_jsonl(data_path)
    grouped_data = group_data_by_id(incoming_test_data) 
    test_data = get_test_formatted(grouped_data)
    MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME) 
    train_dataset = DialogueDatasetUnpacked(test_data, tokenizer, output_key = output_key)
    collator = DialogueCollatorUnpacked(tokenizer, device) 
    train_loader = DataLoader(train_dataset, batch_size=10, collate_fn=collator)
    for batch in train_loader:
        print(batch)
        break

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
