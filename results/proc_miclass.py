# reads a jsonl file and output the results 
import json 
import sys 

def read_jsonl(file_path): 
    with open(file_path, "r") as f: 
        return [json.loads(line) for line in f]

def iterate_through_classes(line):
    classes = ["generic", "focus", "telling", "probing"] 
    # now iterate through the list to get the misclassification rate 
    for cl in classes: 
        list_of_mis = line[cl]
        misclassification_rate = {cl: 0 for cl in classes}

        for mis in list_of_mis:
            misclassification_rate[mis] += 1
        print(f"Misclassification rate for {cl}: {misclassification_rate}")

def main():
    file = "/work/pi_andrewlan_umass_edu/fikram_umass-edu/dialogue-kt/results/misclassification_rates_32bv2.jsonl"
    lines = read_jsonl(file)
    iterate_through_classes(lines[0])
        
main()