import json 
from collections import defaultdict
DATA_PATH = '/work/pi_andrewlan_umass_edu/fikram_umass-edu/dialogue-kt/teacher_moves/processed_data/test_check4.jsonl'

def read_jsonl(data_path):
    with open(data_path) as f:
        return [json.loads(line) for line in f]

def split_train_val(grouped_data):
    train_data, val_data = train_test_split(list(grouped_data.values()), test_size=0.2, random_state=RANDOM_SEED)
    return train_data, val_data

def group_data_by_id(data):
    # add all ids to a list
    grouped_data = defaultdict(list)
    for entry in data:
        grouped_data[entry["id"]].append(entry)
    
    # remove duplicates 
    # for key in grouped_data.keys():
    #     grouped_data[key] = pd.DataFrame(grouped_data[key]).drop_duplicates(subset=['teacher_move', 'student_move']).to_dict('records')
    return grouped_data


def get_test_formatted(grouped_data):
    test_data = list(grouped_data.values())
    return test_data

def get_test():
    data = read_jsonl(DATA_PATH)
    grouped_data = group_data_by_id(data)
    test_data = get_test_formatted(grouped_data)

    return test_data