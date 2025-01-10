import json
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
ACCESS_TOKEN = "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"
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
        self.data = []
        for i in range(len(formatted_outputs)): 
            self.data.append({
                    "id": ids[i], 
                    "prompts":[
                        tokenizer.apply_chat_template([
                            {"role": "system", "content": SYSTEM_PROMPT_TEMPLATE.format(desc=MATHDIAL_DIALOGUE_DESC)},
                            {"role": "user", "content": formatted_outputs[i]},
                            {"role": "assistant", "content": f"\n"} # Newline would precede True or False prediction
                        ], tokenize=False)
                    ],
                    "label": labels[i],
                    })
        print("Processed data")
        print(f"Number of data points: {len(self.data)}")

class DialogueCollatorUnpacked:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        all_prompts = [prompt for sample in batch for prompt in sample["prompts"]]
        prompts_tokenized = self.tokenizer(all_prompts, return_tensors="pt", padding=True).to(device)

        return {
            "input_ids": prompts_tokenized.input_ids,
            "attention_mask": prompts_tokenized.attention_mask,
            "last_idxs": prompts_tokenized.attention_mask.sum(dim=-1) - 2, # Take index of token before eos
            "labels": torch.Tensor([sample["label"] for sample in batch]).to(device),
            "meta_data": batch
        }


# Usage example:
if __name__ == "__main__":
    train_data = read_jsonl(DATA_PATH)
    grouped_data = group_data_by_id(train_data) 
    train_data, val_data = split_train_val(grouped_data)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME) 
    train_dataset = DialogueDatasetUnpacked(train_data, tokenizer)
    collator = DialogueCollatorUnpacked(tokenizer) 
    train_loader = DataLoader(train_dataset, batch_size=1, collate_fn=collator)


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
