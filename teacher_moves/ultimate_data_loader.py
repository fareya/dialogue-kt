import json
import torch
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
import pandas as pd 

# Constants and Global Variables
RANDOM_SEED = 42

MATHDIAL = "mathdial"
ANATION = "anation"

# PROMPT SETUP
MATHDIAL_DIALOGUE_DESC = "the student is attempting to solve a math problem."
SYSTEM_PROMPT_TEMPLATE= (
    "You are an experienced math teacher. You are given a dialogue between a student and teacher where {desc} "
    "Your job is to predict student move correctness. There are four teacher move types: generic, focus, telling, probing."
    "You will output true, false, or n/a. "
    "The dialogue is as follows:"
)
SYSTEM_PROMPT_TEMPLATE2= (
    "You are an experienced math teacher. You are given a dialogue between a student and teacher where {desc} "
    "Your job is to predict the next teacher move types."
    "You will output the next teacher move only. "
    "The dialogue is as follows:"
)

# print("Model Name: ", MODEL_NAME)
# print("System Prompt: ", SYSTEM_PROMPT_TEMPLATE)
# Read JSONL File
def read_jsonl(data_path):
    with open(data_path) as f:
        return [json.loads(line) for line in f]

def group_data_by_id(data):
    # add all ids to a list
    grouped_data = defaultdict(list)
    for entry in data:
        grouped_data[entry["id"]].append(entry)
    
    # remove duplicates 
    # for key in grouped_data.keys():
    #     grouped_data[key] = pd.DataFrame(grouped_data[key]).drop_duplicates(subset=['teacher_move', 'student_move']).to_dict('records')
    return grouped_data

def split_train_val(grouped_data):
    train_data, val_data = train_test_split(list(grouped_data.values()), test_size=0.2, random_state=RANDOM_SEED)
    return train_data, val_data

def get_test_formatted(grouped_data):
    test_data = list(grouped_data.values())
    return test_data

def dialogue_apply_chat_template(dialogue, labels, ids, tokenizer, dataset_name = "mathdial"):
    """
    Mnemonic generation prompt. 
    Constructs the prompt for LLaMA fine-tuning using special tokens.
    """
    if dataset_name == MATHDIAL:
        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(desc=MATHDIAL_DIALOGUE_DESC)
    elif dataset_name == ANATION: 
        system_prompt = SYSTEM_PROMPT_TEMPLATE2
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


def format_dialogue_mathdial(data, pred_label_name, include_previous_turn_types=False):
    formatted_outputs = []
    labels = [] 
    ids = [] 
    print("Length of data: ", len(data))
    for conversation in data: 
        # print("Length of conversation: ", len(conversation))
        dialogue_text = []
        # print(conversation)
        # print(len(conversation))
        for i, turn in enumerate(conversation):
            if pred_label_name == "future_teacher_move_type":
                teacher_type = turn['teacher_move_type']
                if turn['future_teacher_move_type']:
                    if include_previous_turn_types:
                        dialogue_text.append(f"\nTeacher Turn {turn['turn']}: {turn['teacher_move']} ({teacher_type}) \nStudent Turn  {turn['turn']}: {turn['student_move']}")
                    else:
                        dialogue_text.append(f"\nTeacher Turn {turn['turn']}: {turn['teacher_move']} \nStudent Turn  {turn['turn']}: {turn['student_move']}")
                    user_dialogue = "".join(dialogue_text) # replaced 
                    user_prompt = "[BEGIN DIALOGUE]" + user_dialogue + "\n[END DIALOGUE]"       
                    formatted_outputs.append(user_prompt)
                    labels.append(turn[pred_label_name])
                    ids.append(turn["id"])
            
            elif pred_label_name == "teacher_move_type":
                teacher_type = turn['teacher_move_type']
                if turn['teacher_move_type'] and i < len(conversation)-1:
                    curr_teacher_move_model= conversation[i+1]['teacher_move']
                    curr_teacher_turn = conversation[i+1]['turn']
                    curr_teacher_type = conversation[i+1]['teacher_move_type']
                    if include_previous_turn_types:
                        additional_text = f"\nTeacher Turn {curr_teacher_turn}: {curr_teacher_move_model}"
                        dialogue_text.append(f"\nTeacher Turn {turn['turn']}: {turn['teacher_move']} ({teacher_type})\nStudent Turn  {turn['turn']}: {turn['student_move']}")
                    else:
                        additional_text = f"\nTeacher Turn {curr_teacher_turn}: {curr_teacher_move_model}"
                        dialogue_text.append(f"\nTeacher Turn {turn['turn']}: {turn['teacher_move']} \nStudent Turn  {turn['turn']}: {turn['student_move']}")
                    user_dialogue = "".join(dialogue_text) # replaced 
                    user_prompt = "[BEGIN DIALOGUE]" + user_dialogue + additional_text + "\n[END DIALOGUE]"       
                    formatted_outputs.append(user_prompt)
                    labels.append(curr_teacher_type)
                    ids.append(turn["id"])
            elif pred_label_name == "correctness_annotation":
                    teacher_type = turn['teacher_move_type']
                    if include_previous_turn_types:
                        dialogue_text.append(f"\nTeacher Turn {turn['turn']}:  {turn['teacher_move']} ({teacher_type})\nStudent Turn  {turn['turn']}: {turn['student_move']}")
                    else:
                        dialogue_text.append(f"\nTeacher Turn {turn['turn']}: {turn['teacher_move']} \nStudent Turn  {turn['turn']}: {turn['student_move']}")
                    user_dialogue = "".join(dialogue_text) # replaced 
                    user_prompt = "[BEGIN DIALOGUE]" + user_dialogue + "\n[END DIALOGUE]"       
                    formatted_outputs.append(user_prompt)
                    labels.append(turn[pred_label_name])
                    ids.append(turn["id"])
            elif pred_label_name == "future_correctness_annotation":
                if turn['teacher_move_type'] and i < len(conversation)-1:
                    curr_teacher_move_model= conversation[i+1]['teacher_move']
                    curr_teacher_turn = conversation[i+1]['turn']
                    curr_teacher_type = conversation[i+1]['teacher_move_type']
                    future_correctness_annotation = conversation[i+1]['correctness_annotation']
                    teacher_type = turn['teacher_move_type']
                    if include_previous_turn_types:
                        additional_text = f"\nTeacher Turn {curr_teacher_turn}: {curr_teacher_move_model}"
                        dialogue_text.append(f"\nTeacher Turn {turn['turn']}: {turn['teacher_move']} ({teacher_type})\nStudent Turn  {turn['turn']}: {turn['student_move']}")
                    else:
                        additional_text = f"\nTeacher Turn {curr_teacher_turn}: {curr_teacher_move_model}"
                        dialogue_text.append(f"\nTeacher Turn {turn['turn']}: {turn['teacher_move']} \nStudent Turn  {turn['turn']}: {turn['student_move']}")
                    user_dialogue = "".join(dialogue_text) # replaced 
                    user_prompt = "[BEGIN DIALOGUE]" + user_dialogue + additional_text + "\n[END DIALOGUE]"       
                    formatted_outputs.append(user_prompt)
                    labels.append(future_correctness_annotation)
                    ids.append(turn["id"])

            elif pred_label_name == "final_correctness":
                    teacher_type = turn['teacher_move_type']
                    if include_previous_turn_types:
                        dialogue_text.append(f"\nTeacher Turn {turn['turn']}: {turn['teacher_move']}({teacher_type})  \nStudent Turn  {turn['turn']}: {turn['student_move']}")
                    else:
                        dialogue_text.append(f"\nTeacher Turn {turn['turn']}: {turn['teacher_move']} \nStudent Turn  {turn['turn']}: {turn['student_move']}")
                    user_dialogue = "".join(dialogue_text) # replaced 
                    user_prompt = "[BEGIN DIALOGUE]" + user_dialogue + "\n[END DIALOGUE]"       
                    formatted_outputs.append(user_prompt)
                    labels.append(turn[pred_label_name])
                    ids.append(turn["id"])
            

    print("Length of formatted outputs: ", len(formatted_outputs))

    # print 5 examples and their labels
    for i in range(5):
        print("Formatted Output: ", formatted_outputs[i])
        print("Label: ", labels[i])
        print("ID: ", ids[i])
    assert len(formatted_outputs) == len(labels)
    return formatted_outputs, labels, ids


# def format_dialogue_anation(data, label_type = "teacher_move_type", add_prev_labels = True): 
#     formatted_outputs = []
#     labels = []
#     ids = [] 
#     for i, conversation in enumerate(data):
#         dialogue = []
#         tutor_labels = []
#         is_tutor_list = [] 
#         correctness = [] 
#         for j, turn in enumerate(conversation): 
#             dialogue.append(turn['processed_text'])
#             if turn["Success"]:
#                 correctness.append("true") 
#             else: correctness.append("true") 
#             if turn['is_tutor']: 
#                 tutor_labels.append(turn['list_of_labels'])
#                 is_tutor_list.append(True)
#             else: 
#                 tutor_labels.append(None)
#                 is_tutor_list.append(False)

#         final_dialogue = "[BEGIN DIALOGUE]"
#         for k in range(len(conversation)): 
#             if k>0 and add_prev_labels:
#                 prepend_text = "" if not is_tutor_list[k-1] else str(tutor_labels[k-1])
#             else: 
#                 prepend_text = ""
#             final_dialogue = final_dialogue + prepend_text + "\n" + dialogue[k]
        
#             if is_tutor_list[k] and label_type == "teacher_move_type":
#                 formatted_outputs.append(final_dialogue + "\n[END DIALOGUE]")
#                 labels.append(tutor_labels[k])
#                 ids.append(str(conversation[k]["id"])+str(conversation[k]["order"]))
            
#             if is_tutor_list[k] and label_type == "final_correctness":
#                 formatted_outputs.append(final_dialogue + "\n[END DIALOGUE]")
#                 labels.append(correctness[k])
#                 ids.append(str(conversation[k]["id"])+str(conversation[k]["order"]))

#             if k < len(conversation) - 1: 
#                 if is_tutor_list[k+1] and label_type == "future_teacher_move_type":
#                     formatted_outputs.append(final_dialogue + "\n[END DIALOGUE]")
#                     labels.append(tutor_labels[k+1])
#                     ids.append(str(conversation[k]["id"])+str(conversation[k]["order"]))

#     for i in range(10):
#         print(formatted_outputs[i])
#         print(labels[i])
#     return formatted_outputs, labels, ids 

def format_dialogue_anation(data, label_type = "teacher_move_type", add_prev_labels = True): 
    formatted_outputs = []
    labels = []
    ids = [] 
    for i, conversation in enumerate(data):
        dialogue = []
        tutor_labels = []
        is_tutor_list = [] 
        correctness = [] 
        for j, turn in enumerate(conversation): 
            dialogue.append(turn['processed_text'])
            if turn["Success"]:
                correctness.append("true") 
            else: 
                correctness.append("false") 
            if turn['is_tutor']: 
                tutor_labels.append(turn['list_of_labels'])
                is_tutor_list.append(True)
            else: 
                tutor_labels.append(None)
                is_tutor_list.append(False)

        final_dialogue = "[BEGIN DIALOGUE]"
        for k in range(len(conversation)): 
            if k>0 and add_prev_labels:
                prepend_text = "" if not is_tutor_list[k-1] else str(tutor_labels[k-1])
            else: 
                prepend_text = ""
            final_dialogue = final_dialogue + prepend_text + "\n" + dialogue[k]
        
            if is_tutor_list[k] and label_type == "teacher_move_type":
                formatted_outputs.append(final_dialogue + "\n[END DIALOGUE]")
                labels.append(tutor_labels[k])
                ids.append(str(conversation[k]["id"])+str(conversation[k]["id2"]))
            
            if is_tutor_list[k] and label_type == "final_correctness":
                formatted_outputs.append(final_dialogue + "\n[END DIALOGUE]")
                labels.append(correctness[k])
                ids.append(str(conversation[k]["id"])+str(conversation[k]["id2"]))

            if k < len(conversation) - 1: 
                if is_tutor_list[k+1] and label_type == "future_teacher_move_type":
                    formatted_outputs.append(final_dialogue + "\n[END DIALOGUE]")
                    labels.append(tutor_labels[k+1])
                    ids.append(str(conversation[k]["id"])+str(conversation[k]["id2"]))
    final_formatted_outputs = []
    final_labels = []
    final_ids = [] 

    for i in range(len(formatted_outputs)): 
        if not(labels[i] == "[]"):
            final_formatted_outputs.append(formatted_outputs[i])
            final_labels.append(labels[i])
            final_ids.append(ids[i])
    for i in range(10):
        print(final_formatted_outputs[i])
        print(final_labels[i])
    return final_formatted_outputs, final_labels, final_ids 


class DatasetBase(Dataset):
    def __getitem__(self, index: int):
        return self.data[index]

    def __len__(self):
        return len(self.data)

class DialogueDatasetUnpacked(DatasetBase):
    def __init__(self, data: list, tokenizer, skip_first_turn: bool = False, model_name: str = None, output_key: str = None, include_previous_turn_types = False, dataset_name = MATHDIAL):
        # tokenizer = AutoTokenizer.from_pretrained(model_name) 
        # print(OUTPUT_KEY)
        # print(output_key)
        
        # assert output_key == OUTPUT_KEY
        if dataset_name == MATHDIAL:
            formatted_outputs, labels, ids = format_dialogue_mathdial(data, output_key, include_previous_turn_types)
        elif dataset_name == ANATION: 
            formatted_outputs, labels, ids = format_dialogue_anation(data, output_key, include_previous_turn_types)

        print("Formatted data")
        # for i in range(50):
        #     print(formatted_outputs[i])
        #     print(labels[i])
        chat_formatted_dialogue = [dialogue_apply_chat_template(formatted_outputs[i], labels[i],ids[i], tokenizer, dataset_name) for i in range(len(formatted_outputs))]
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
    INCLUDE_PREV = True # True/False 
    print(f"Include previous: {INCLUDE_PREV}")
    PRED_LABEL_NAME = "final_correctness" # options: future_teacher_move_type, teacher_move_type, correctness_annotation, future_correctness_annotation, final_correctness
    print(f"Label Name: {PRED_LABEL_NAME}")
    MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
    DATASET_NAME = "mathdial"# anation, mathdial
    print(f"Dataset: {DATASET_NAME}")

    if DATASET_NAME == ANATION: 
        DATA_PATH = "/work/pi_andrewlan_umass_edu/fikram_umass-edu/dialogue-kt/teacher_moves/processed_data/anation_val_data.jsonl"
    if DATASET_NAME == MATHDIAL: 
        DATA_PATH = "/work/pi_andrewlan_umass_edu/fikram_umass-edu/dialogue-kt/teacher_moves/processed_data/test_check4.jsonl"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    data = read_jsonl(DATA_PATH)
    grouped_data = group_data_by_id(data)
    test_data = get_test_formatted(grouped_data)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME) 
    test_dataset = DialogueDatasetUnpacked(test_data, tokenizer, output_key = PRED_LABEL_NAME, include_previous_turn_types=INCLUDE_PREV, dataset_name = DATASET_NAME)
    collator = DialogueCollatorUnpacked(tokenizer, device) 
    test_loader = DataLoader(test_dataset, batch_size=1, collate_fn=collator)



    for batch in test_loader:
        print(batch) 
        break