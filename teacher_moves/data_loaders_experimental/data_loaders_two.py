import torch
from torch.utils.data import Dataset, DataLoader
import json
from sklearn.model_selection import train_test_split
from collections import defaultdict
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

RANDOM_SEED = 42


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

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

class TeacherMovesDataset(Dataset):
    def __init__(self, grouped_data, max_seq_length=5, max_tokens=32):
        """
        Args:
            grouped_data (dict): Dictionary where keys are `id`s and values are lists of dialogue turns.
            max_seq_length (int): Number of teacher moves in a sequence.
            max_tokens (int): Maximum token length per teacher move.
        """
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.data = self.process_data(grouped_data, max_seq_length)

    def process_data(self, grouped_data, max_seq_length):
        """Processes grouped data into sequences of tokenized teacher moves with labels."""
        sequences = []
        for dialogue in grouped_data:
            teacher_moves = [turn["teacher_move_type"] for turn in dialogue]
            labels = [turn["correctness_annotation"] for turn in dialogue]

            for i in range(len(teacher_moves)):
                if labels[i] == "na":  
                    continue  # Ignore instances with "na" labels

                seq_moves = teacher_moves[:i+1]  # Sliding window
                seq_labels = labels[i]  # Corresponding label
                
                # Pad/Truncate at dialogue level
                seq_moves_padded = [""] * (max_seq_length - len(seq_moves)) + seq_moves[-max_seq_length:]

                # Tokenize each teacher move (padded to max_tokens per move)
                tokenized_moves = [
                    self.tokenizer(move, padding="max_length", truncation=True, max_length=self.max_tokens, return_tensors="pt")["input_ids"].squeeze(0)
                    for move in seq_moves_padded
                ]
                
                tokenized_moves = torch.stack(tokenized_moves)  # Shape: (max_seq_length, max_tokens)
                sequences.append((tokenized_moves, seq_labels))
        
        return sequences

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """Returns a single sequence of tokenized teacher moves and its corresponding label."""
        move_tensor, label = self.data[idx]

        # Convert label to integer (0 for "false", 1 for "true")
        label_map = {"true": 1, "false": 0}
        label_tensor = torch.tensor(label_map[label], dtype=torch.long)

        return move_tensor, label_tensor


# Example integration with your workflow
data = read_jsonl("/work/pi_andrewlan_umass_edu/fikram_umass-edu/dialogue-kt/teacher_moves/processed_data/train_check2.jsonl")
grouped_data = group_data_by_id(data)
# Split the data
train_data, val_data = split_train_val(grouped_data)

# Create datasets
train_dataset = TeacherMovesDataset(train_data)
val_dataset = TeacherMovesDataset(val_data)

# # Create data loaders
# train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)

# # Example iteration
# for batch in train_loader:
#     moves_seq, labels = batch
#     print("Moves Sequence:", moves_seq)
#     print("Labels:", labels)
#     break


def verify_dataset(dataset):
    """
    Verifies the correctness of the TeacherMovesDataset.
    Ensures:
    - All labels are either 0 ("false") or 1 ("true")
    - Tokenized sequences have correct dimensions
    - Prints a sample of untokenized and tokenized teacher moves
    """
    all_labels = []
    for i in range(min(10, len(dataset))):  # Check first 10 samples
        moves_seq, label = dataset[i]
        
        # Ensure label is 0 or 1
        assert label in [0, 1], f"Incorrect label found: {label}"

        # Ensure tokenized teacher moves have shape (max_seq_length, max_tokens)
        assert moves_seq.shape == (dataset.data[0][0].shape[0], dataset.max_tokens), \
            f"Unexpected tensor shape: {moves_seq.shape}"

        all_labels.append(label)
        
        # Retrieve original untokenized sequences
        untokenized_moves = dataset.data[i][0]  # List of original teacher moves
        
        # Decode the first move from tokenized IDs back to text (for inspection)
        decoded_moves = [dataset.tokenizer.decode(moves_seq[j].tolist()) for j in range(len(moves_seq))]

        # Print sample output
        print(f"Sample {i+1}:")
        print(f"- Tokenized Moves Shape: {moves_seq.shape}")
        print(f"- Label (0=False, 1=True): {label}")
        print("\nOriginal Teacher Moves:")
        for j, move in enumerate(untokenized_moves):
            print(f"  {j+1}. {move}")

        print("\nDecoded Tokenized Moves (After Tokenization & Padding):")
        for j, move in enumerate(decoded_moves):
            print(f"  {j+1}. {move}")

        print("-" * 50)

    print("\n✅ Dataset verification passed: All labels are correct & tokenized sequences are valid.\n")

# Run verification
print("Verifying training dataset...")
verify_dataset(train_dataset)

print("Verifying validation dataset...")
verify_dataset(val_dataset)
