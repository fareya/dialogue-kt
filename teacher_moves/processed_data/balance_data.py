import json
import random
from collections import defaultdict


def balance_teacher_moves(data, output_file):
    """
    Balances the dataset to ensure equal representation of future_teacher_move_type.

    Args:
        data (list): A list of dictionaries containing the dataset.
        output_file (str): The file to save the balanced dataset.
    """
    # Group data by future_teacher_move_type
    grouped_data = defaultdict(list)
    for entry in data:
        grouped_data[entry["future_teacher_move_type"]].append(entry)

    # Print the count of samples for each move type
    # for move_type, entries in grouped_data.items():
    #     print(f"{move_type}: {len(entries)}")

    # filter out the move types with less than 100 samples
    filtered_data = {move_type: entries for move_type, entries in grouped_data.items() if len(entries) >= 100}

    # Print the count of samples for each move type after filtering
    for move_type, entries in filtered_data.items():
        print(f"{move_type}: {len(entries)}")

    # Find the minimum count of samples across move types
    min_count = min(len(entries) for entries in filtered_data.values())
    print(f"Minimum count: {min_count}")
    # Subsample each group to the minimum count

    balanced_data = []
    for move_type, entries in filtered_data.items():
        balanced_data.extend(random.sample(entries, min_count))

    # Shuffle the balanced data
    random.shuffle(balanced_data)

    # Write to the output file
    # write each entry as a json object on a new line
    with open(output_file, "w") as f:
        for entry in balanced_data:
            f.write(json.dumps(entry) + "\n")   

    print(f"Balanced dataset written to {output_file}")


# read input jsonl file 
data_path = "/work/pi_andrewlan_umass_edu/fikram_umass-edu/dialogue-kt/teacher_moves/processed_data/train.jsonl"
output_file = "/work/pi_andrewlan_umass_edu/fikram_umass-edu/dialogue-kt/teacher_moves/processed_data/balanced_train.jsonl"

with open(data_path, "r") as f:
    data = [json.loads(line) for line in f]
    balance_teacher_moves(data, output_file)
