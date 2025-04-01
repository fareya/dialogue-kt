import json 
from collections import defaultdict 
from sklearn.model_selection import train_test_split
from openai_api import OpenAIClient
from sklearn.metrics import f1_score, accuracy_score
import os 
import re 

label_to_category = {
    'seek_strategy': 'focus',
    'guiding_student_focus': 'focus',
    'recall_relevant_information': 'focus',
    'asking_for_explanation': 'probing',
    'seeking_self_correction': 'probing',
    'perturbing_the_question': 'probing',
    'seeking_world_knowledge': 'probing',
    'revealing_strategy': 'telling',
    'revealing_answer': 'telling',
    'greeting_fairwell': 'generic',
    'general_inquiry': 'generic', 
    'generic':'generic',
}

SYSTEM = "You are a math teacher who tutors student on a variety of problems."

PROMPT2 = """Your task is to classify the text into one of four following categories: Focus, Probing, Telling, Generic. These four categories are described below.
- Focus:
  - Seek Strategy: Ex. So what should you do next?
  - Guiding Student Focus: Ex. Can you calculate ...?
  - Recall Relevant Information: Ex. Can you reread the question and tell me what is ...?
- Probing:
  - Asking for Explanation: Ex. Why do you think you need to add these numbers?
  - Seeking Self Correction: Ex. Are you sure you need to add here?
  - Perturbing the Question: Ex. How would things change if they had ...items instead?
  - Seeking World Knowledge: Ex. How do you calculate the perimeter of a square
- Telling:
  - Revealing Strategy: Ex. You need to add ...to ...to get your answer.
  - Revealing Answer: Ex. No, he had ...items.
- Generic:
  - Greeting/Fairwell: Ex. Hi ..., how are you doing with the word problem? Good Job! Is there anything else I can help with?
  - General Inquiry: Ex.  Can you go walk me through your solution?

Given the student teacher dialogue below, classify each teacher move into the sub-categories of the four categories. Return a json object where the key is the teacher move number and the value is the sub category. For example, a return json type would be {'0': 'greeting', '1':'seek_strategy'}.
Here is how the sub categories should be mapped to their key names. 
- Seek Strategy -> seek_strategy
- Guiding Student -> guiding_student_focus,
- Recall Relevant Information -> recall_relevant_information
- Asking for Explanation -> asking_for_explanation
- Seeking Self Correction -> seeking_self_correction
- Perturbing the Question -> perturbing_the_question
- Seeking World Knowledge -> seeking_world_knowledge,
- Revealing Strategy -> revealing_strategy
- Revealing Answer -> revealing_answer
- Greeting/Fairwell -> greeting_fairwell
- General Inquiry -> general_inquiry

Please categorize the teacher moves in the following conversation: 
"""


PROMPT1 = """Your task is to classify the text into one of four following categories: Focus, Probing, Telling, Generic. These four categories are described below.
- Focus:
  - Seek Strategy: Ex. So what should you do next?
  - Guiding Student Focus: Ex. Can you calculate ...?
  - Recall Relevant Information: Ex. Can you reread the question and tell me what is ...?
- Probing:
  - Asking for Explanation: Ex. Why do you think you need to add these numbers?
  - Seeking Self Correction: Ex. Are you sure you need to add here?
  - Perturbing the Question: Ex. How would things change if they had ...items instead?
  - Seeking World Knowledge: Ex. How do you calculate the perimeter of a square
- Telling:
  - Revealing Strategy: Ex. You need to add ...to ...to get your answer.
  - Revealing Answer: Ex. No, he had ...items.
- Generic:
  - Greeting/Fairwell: Ex. Hi ..., how are you doing with the word problem? Good Job! Is there anything else I can help with?
  - General inquiry: Ex.  Can you go walk me through your solution?

Given the student teacher dialogue below, classify each teacher move into the one of the four categories. Return a json object where the key is the teacher move number and the value is the category. For example, a return json type would be {'0': 'telling', '1':'probing'}.

Please categorize the teacher moves in the following conversation: 
"""


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
    RANDOME_SEED = 42
    train_data, val_data = train_test_split(list(grouped_data.values()), test_size=0.2, random_state=42)
    return train_data, val_data

def get_test_formatted(grouped_data):
    test_data = list(grouped_data.values())
    return test_data

def extract_labels(data_one, data_two):
    labels_ground_truth = []
    labels_prediction = []
    for i in range(len(data_one)):
      if len(data_one[i].values()) == len(data_two[i].values()):
        labels_ground_truth.extend(data_one[i].values())
        labels_prediction.extend(data_two[i].values())
    return labels_ground_truth, labels_prediction

def write_to_jsonl(filename, data):
    with open(filename, 'w') as file:
        for item in data:
            # Use regex to extract JSON content
            match = re.search(r'\{.*\}', item, re.DOTALL)
            if match:
                json_str = match.group(0)
                try:
                    json_obj = json.loads(json_str)
                    file.write(json.dumps(json_obj) + '\n')
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON: {e}")
            else:
                print("No valid JSON found in the item.")

def map_to_category(labels):
    return [label_to_category[label] for label in labels]

def format_dialogue(data):
    formatted_outputs = []
    labels = []
    print("Length of data: ", len(data))
    for conversation in data: 
        # print("Length of conversation: ", len(conversation))
        dialogue_text = []
        curr_label_map = {} 
        for i, turn in enumerate(conversation):
            curr_label_map[i] = turn['teacher_move_type']
            teacher_text = f"Teacher Turn {i}: {turn['teacher_move']}\n"
            student_text = f"Student Turn {i}: {turn['student_move']}\n" if turn['student_move'] else ""
            dialogue_text.append(teacher_text)
            dialogue_text.append(student_text)
        
        final_formatting = "[BEGIN DIALOGUE]\n"
        for d in dialogue_text: 
            final_formatting = final_formatting + d
        final_formatting = final_formatting + "[END DIALOGUE]"
        formatted_outputs.append(final_formatting)
        labels.append(curr_label_map)
    print("Length of formatted outputs: ", len(formatted_outputs))
    assert len(formatted_outputs) == len(labels)
    return formatted_outputs, labels

use_prompt_one = False 

if use_prompt_one: 
  print("Using Prompt One")
else: 
  print("Using Prompt Two")
data_path = "/work/pi_andrewlan_umass_edu/fikram_umass-edu/dialogue-kt/teacher_moves/processed_data/train_check4.jsonl"
if use_prompt_one: 
  out_path = "/work/pi_andrewlan_umass_edu/fikram_umass-edu/dialogue-kt/results/gpt_out/out_prompt1.jsonl"
else: 
    out_path = "/work/pi_andrewlan_umass_edu/fikram_umass-edu/dialogue-kt/results/gpt_out/out_prompt2.jsonl"

data = read_jsonl(data_path)
grouped_data = get_test_formatted(group_data_by_id(data))
formatted_outputs, labels = format_dialogue(grouped_data)

if use_prompt_one: 
  prompts = [PROMPT1 + out for out in formatted_outputs]
else: 
  prompts = [PROMPT2 + out for out in formatted_outputs]

model = "gpt-4o" 
max_tokens = 512
batch_size = 1
temperature = 0

print(os.getenv("AZURE_OPENAI_API_KEY"))
print(os.getenv("AZURE_OPENAI_ENDPOINT"))


oac = OpenAIClient(use_azure_client=True)
responses = oac.get_batched_responses(prompts, model, max_tokens, batch_size, temperature, system_message= SYSTEM)
write_to_jsonl(out_path, responses)
prediction_data = read_jsonl(out_path)
# extracted_labels = extract_labels(prediction_data)
# extracted_ground_truth_labels = extract_labels(labels)
extracted_labels, extracted_ground_truth_labels = extract_labels(prediction_data,labels)
print(set(extracted_labels))
print(set(extracted_ground_truth_labels))
if use_prompt_one:
  mapped_labels = extracted_labels # map_to_category(extracted_labels)
else: 
  mapped_labels = map_to_category(extracted_labels)


f1 = f1_score(extracted_ground_truth_labels, mapped_labels, average='weighted')
accuracy = accuracy_score(extracted_ground_truth_labels, mapped_labels)
print(f1)
print(accuracy)