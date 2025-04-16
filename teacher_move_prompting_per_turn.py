import json 
from collections import defaultdict 
from sklearn.model_selection import train_test_split
from openai_api import OpenAIClient
from sklearn.metrics import f1_score, accuracy_score
from teacher_moves.ultimate_data_loader import format_dialogue_mathdial, format_dialogue_anation
import os 
import re 
import ast 
from sklearn.metrics import f1_score
from sklearn.preprocessing import MultiLabelBinarizer

SYSTEM = "You are a math teacher who tutors student on a variety of problems."
PROMPT_MATHDIAL = """Your task is to classify the text into one of four following categories: Focus, Probing, Telling, Generic. These four categories are described below.
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

Given the student teacher dialogue below, classify the final teacher move. Please return a json object that has the dialogue id as the key and the the teacher turn type as the value. 
For example, the output would look like {"1234" : "focus"}. The key and the value should both be strings. 

Please categorize the final teacher move in the following conversation: 
"""

PROMPT_ANATION =  """Your task is to classify the text into one or more of the following categories: confirmatory_feedback, negative_feedback, correcting, giving_instruction, giving_explanation, giving_explanation, providing_further_references, questioning, asking_for_elaboration, praising_and_encouraging, managing_frustration, managing_discussions, giving_answers, encouraging_peer_tutoring, guiding_peer_tutoring, acknowledging_tutor_issue, and other. 
The categories are described below:

- confirmatory_feedback : Whether a reply provides confirmatory feedback about an answer's correctness.
- negative_feedback: Whether a reply states that an answer is incorrect.
- correcting: Whether a reply addresses errors in the student's problem-solving approach.
- giving_instruction: Whether a reply breaks down a task, performs a part, or initiates a task for the student to complete.
- giving_explanation: Whether a reply explains concepts, principles, or provides additional information 
- providing_further_references: Whether a reply includes additional resources or references related to the topic.
- questioning: Whether a reply asks questions to stimulate thought or constructive discussion.
- asking_for_elaboration: Whether a reply requests further details or explanation from the student.
- praising_and_encouraging: Whether a reply praises or encourages the student for their efforts or successes.
- managing_frustration: Whether a reply addresses the student's negative emotions or frustration.
- managing_discussions: Whether a reply organizes the flow of discussion or adjusts the direction of inquiry.
- giving_answers Whether a reply directly provides an answer to the posed question.
- encouraging_peer_tutoring: Whether a reply promotes tutoring interactions among peers.
- guiding_peer_tutoring: Whether a reply provides feedback on peer tutoring interactions.
- acknowledging_tutor_issue: W tutor's uncertainty in their reply. 
- other: Binary indicator for tutoring strategies not classified under the existing labels.

Given the student teacher dialogue below, classify the final teacher move. Please return a json object that has the dialogue id as the key and the the teacher turn type as the value. 
For example, the output would look like {"1234" : ["confirmatory_feedback", "correcting"]}. Please use snake case for the categories. The key and the value should both be strings. 

Please categorize the final teacher move in the following conversation: 
"""

def convert_string_to_list(string):
    """
    Converts a string representation of a list into an actual Python list.
    
    Args:
        string (str): The string representation of a list.
        
    Returns:
        list: The converted Python list.
    """
    try:
        return ast.literal_eval(string)
    except (ValueError, SyntaxError) as e:
        print(f"Error converting string to list: {e}")
        return []

def read_jsonl(data_path):
    with open(data_path) as f:
        return [json.loads(line) for line in f]

def group_data_by_id(data):
    grouped_data = defaultdict(list)
    for entry in data:
        grouped_data[entry["id"]].append(entry)
    return grouped_data

def get_test_formatted(grouped_data):
    return list(grouped_data.values())

def extract_labels(data_one, data_two, turn_ids):
    """
    Extracts and matches the values from two lists of dictionaries based on the same order of IDs.

    Args:
        data_one (list): First list of dictionaries with IDs as keys and values as lists or strings.
        data_two (list): Second list of dictionaries with IDs as keys and values as lists or strings.
        turn_ids (list): List of IDs to match the order.

    Returns:
        tuple: Two lists of values corresponding to the matched IDs.
    """
    labels_ground_truth = []
    labels_prediction = []

    # Convert lists of dictionaries into mappings of id to value
    data_one_map = {list(entry.keys())[0]: list(entry.values())[0] for entry in data_one}
    data_two_map = {list(entry.keys())[0]: list(entry.values())[0] for entry in data_two}

    # Match the values based on the order of turn_ids
    for id_ in turn_ids:
        if id_ in data_one_map and id_ in data_two_map:
            labels_ground_truth.append(data_one_map[id_])
            labels_prediction.append(data_two_map[id_])
        else:
            print(f"Warning: ID {id_} not found in one of the datasets.")

    return labels_ground_truth, labels_prediction

from collections import Counter
from itertools import zip_longest

def analyze_labels(ground_truth, predictions):
    """
    Analyze the distribution of labels and misclassifications.

    Args:
        ground_truth (list): List of ground truth labels.
        predictions (list): List of predicted labels.

    Returns:
        None: Prints the analysis results.
    """
    # Flatten lists if they are multi-label
    ground_truth_flat = [label for sublist in ground_truth for label in sublist]
    predictions_flat = [label for sublist in predictions for label in sublist]

    # (1) Distribution of labels
    ground_truth_distribution = Counter(ground_truth_flat)
    predictions_distribution = Counter(predictions_flat)

    print("Ground Truth Label Distribution:")
    for label, count in ground_truth_distribution.items():
        print(f"{label}: {count}")

    print("\nPredicted Label Distribution:")
    for label, count in predictions_distribution.items():
        print(f"{label}: {count}")

    # (2) Misclassifications
    misclassifications = []
    for gt, pred in zip_longest(ground_truth, predictions, fillvalue=[]):
        if set(gt) != set(pred):
            misclassifications.append((gt, pred))

    misclassification_counter = Counter()
    for gt, pred in misclassifications:
        for label in set(gt) - set(pred):  # Missed labels
            misclassification_counter[f"Missed: {label}"] += 1
        for label in set(pred) - set(gt):  # Incorrectly added labels
            misclassification_counter[f"Incorrect: {label}"] += 1

    print("\nMost Common Misclassifications:")
    for label, count in misclassification_counter.most_common(10):
        print(f"{label}: {count}")

def evaluate_multi_label_safe(y_true_raw, y_pred_raw):
    import ast
    from sklearn.preprocessing import MultiLabelBinarizer
    from sklearn.metrics import f1_score, accuracy_score, hamming_loss

    all_labels = [
        'confirmatory_feedback', 'negative_feedback', 'correcting',
        'giving_instruction', 'giving_explanation',
        'providing_further_references', 'questioning', 'asking_for_elaboration',
        'praising_and_encouraging', 'managing_frustration',
        'managing_discussions', 'giving_answers', 'encouraging_peer_tutoring',
        'guiding_peer_tutoring', 'acknowledging_tutor_issue', 'other'
    ]
    # Parse and clean
    y_true = [ast.literal_eval(s) if isinstance(s, str) else s for s in y_true_raw]
    y_pred = [ast.literal_eval(s) if isinstance(s, str) else s for s in y_pred_raw]

    print(y_true)
    print(y_pred)
    # Strip spaces
    y_true = [[label.strip() for label in ex] for ex in y_true]
    y_pred = [[label.strip() for label in ex] for ex in y_pred]

    total = 0 
    for i in range(len(y_true)): 
        # if y_true[i] == []: 
        #     y_true[i] == ['none']
        # if y_pred[i] == []:
        #     y_pred[i] = ['none']
        if y_pred[i] == y_true[i]:
            print(i)
            print(y_pred[i])
            total = total + 1 
    print(y_true)
    print(y_pred)

    print(total/ len(y_true))
    # Initialize binarizer with fixed class order
    mlb = MultiLabelBinarizer()
    mlb.fit(y_true + y_pred)
    # mlb.fit([])

    y_true_bin = mlb.transform(y_true)
    y_pred_bin = mlb.transform(y_pred)

    # Sanity check
    if y_pred_bin.shape != y_true_bin.shape:
        raise ValueError("Shape mismatch between predictions and ground truth after binarization.")

    # Compute metrics
    f1 = f1_score(y_true_bin, y_pred_bin, average='samples')
    exact_match_acc = accuracy_score(y_true_bin, y_pred_bin)

    print(f"F1 Score (samples): {f1:.4f}")
    print(f"Exact Match Accuracy: {exact_match_acc:.4f}")

    return f1, exact_match_acc

def write_to_jsonl(filename, data):
    with open(filename, 'w') as file:
        for item in data:
            match = re.search(r'\{.*\}', item, re.DOTALL)
            if match:
                json_str = match.group(0)
                json_obj = json.loads(json_str)
                file.write(json.dumps(json_obj) + '\n')
            else:
                print("No valid JSON found in the item.")


def map_to_category(labels):
    return [label_to_category[label] for label in labels]

def format_dialogue(data):
    formatted_outputs = []
    labels = []
    turn_ids = []

    for conversation in data:
        dialogue_text = []
        curr_label_map = {}
        curr_turn_ids = []

        for i, turn in enumerate(conversation):
            curr_label_map[i] = turn['teacher_move_type']
            curr_turn_ids.append((turn['id'], turn['turn']))

            teacher_text = f"Teacher Turn {i}: {turn['teacher_move']}\n"
            student_text = f"Student Turn {i}: {turn['student_move']}\n" if turn['student_move'] else ""
            dialogue_text.append(teacher_text)
            dialogue_text.append(student_text)

        final_formatting = "[BEGIN DIALOGUE]\n" + "".join(dialogue_text) + "[END DIALOGUE]"
        formatted_outputs.append(final_formatting)
        labels.append(curr_label_map)
        turn_ids.append(curr_turn_ids)

    return formatted_outputs, labels, turn_ids

def inject_gpt_labels_by_turn_ids(grouped_data, gpt_labels, turn_ids, label_key="gpt_teacher_move_type"):
    id_turn_to_data = {(item["id"], item["turn"]): item for group in grouped_data for item in group}

    for label_dict, id_turn_list in zip(gpt_labels, turn_ids):
        for i, (id_, turn_idx) in enumerate(id_turn_list):
            label = label_dict.get(str(i))
            if label is not None:
                id_turn_to_data[(id_, turn_idx)][label_key] = label
            else:
                print(f"Warning: No GPT label for id={id_}, turn={turn_idx}")

    return list(id_turn_to_data.values())


# -------- Main Script --------

use_mathdial = False
pred_label = "teacher_move_type"
if use_mathdial:
    dataset = "MATHDIAL"
    data_path = "/work/pi_andrewlan_umass_edu/fikram_umass-edu/dialogue-kt/teacher_moves/processed_data/test_check4.jsonl"
    format_fn = format_dialogue_mathdial
else: 
    dataset = "ANATION"
    data_path = "/work/pi_andrewlan_umass_edu/fikram_umass-edu/dialogue-kt/teacher_moves/processed_data/anation_val_data_final.jsonl"
    format_fn = format_dialogue_anation

print(f"Using the dataset:{dataset}")
print(f"Data Path:{data_path}")

data = read_jsonl(data_path)
grouped_data = get_test_formatted(group_data_by_id(data))

formatted_outputs, labels, turn_ids = format_fn(grouped_data,pred_label)
labels_ground_truth = [{turn_ids[i]:labels[i]} for i in range(len(turn_ids))]
prompts = ["Dialogue ID:"+ str(turn_ids[i])+"\n" + PROMPT_MATHDIAL + formatted_outputs[i]  if use_mathdial else  "Dialogue ID:"+ str(turn_ids[i])+"\n" + PROMPT_ANATION + formatted_outputs[i]  for i in range(len(formatted_outputs))]

print(turn_ids[1])
print(type(turn_ids[1]))
print(prompts[1])
print(labels[1])
# store the prompts in a local file 
prompts_output_path = "/work/pi_andrewlan_umass_edu/fikram_umass-edu/dialogue-kt/results/prompts"+"_"+ dataset +"_"+ pred_label +".jsonl"

# Write the prompts to the file
with open(prompts_output_path, 'w') as file: 
    for prompt in prompts:
        file.write(json.dumps({"prompt": prompt}) + '\n')

print(f"Prompts have been written to: {prompts_output_path}")

results_output_path = "/work/pi_andrewlan_umass_edu/fikram_umass-edu/dialogue-kt/results/prompts"+"_"+ dataset +"_"+ pred_label +"_results.jsonl"
model = "gpt-4o"
batch_size = 10

print(os.getenv("AZURE_OPENAI_API_KEY"))
print(os.getenv("AZURE_OPENAI_ENDPOINT"))

# generation_args = {"max_tokens": 1000, "response_format": {"type": "json_object"}, "temperature":0}

# client = OpenAIClient(use_azure_client=True)
# responses = client.get_batched_responses(prompts, model, 10, generation_args, system_message=SYSTEM)

# print("RESPONSES")
# print(len(responses))
# for i in range(len(responses)):
#     print(i)
#     print(responses[i])
# write_to_jsonl(results_output_path, responses)

prediction_data = read_jsonl(results_output_path)
print("labels")
print(labels_ground_truth[:10])
extracted_ground_truth_labels, extracted_labels  = extract_labels(labels_ground_truth, prediction_data, turn_ids)

print("len(extracted_labels)")
print(len(extracted_labels))
print(extracted_labels[:10])
print("len(extracted_ground_truth_labels)")
print(len(extracted_ground_truth_labels))
print(extracted_ground_truth_labels[:10])

if dataset == "ANATION":
   extracted_labels = [
        [l] if isinstance(l, str) else l for l in extracted_labels
    ]
   print(extracted_labels[:10])
   extracted_ground_truth_labels = [ast.literal_eval(l) for l in extracted_ground_truth_labels]
   evaluate_multi_label_safe(extracted_ground_truth_labels, extracted_labels)

else: 
    f1 = f1_score(extracted_ground_truth_labels, extracted_labels, average='weighted')
    accuracy = accuracy_score(extracted_ground_truth_labels, extracted_labels)
    print(f"Accuracy: {accuracy}")
    print(f"F1 Score: {f1}")


# Example usage
analyze_labels(extracted_ground_truth_labels, extracted_labels)

# # Inject GPT labels and future labels
# label_key = "gpt_teacher_move_type" if use_prompt_one else "gpt_teacher_move_subtype"
# updated_data = inject_gpt_labels_by_turn_ids(grouped_data, prediction_data, turn_ids, label_key=label_key)
# updated_data = add_future_gpt_move_type(updated_data, label_key=label_key)

# # Save final data
# write_dicts_to_jsonl(output_gpt_labeled_data, updated_data)
# print(f"Wrote GPT-labeled training data to: {output_gpt_labeled_data}")
