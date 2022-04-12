# import torch
import sys
import pandas as pd
from parser import extract_all

def extract_targets(conversation_ids, fluent=False):
    targets = []
    for conversation_id in conversation_ids:
        targets.extend(extract_all.extract(conversation_id, return_fluent=fluent))
    
    return pd.DataFrame(targets)

if __name__ == "__main__":
    conversation_ids_file = sys.argv[1]

    with open(conversation_ids_file, 'r') as f:
        conversation_ids = f.read().splitlines()
    print(extract_targets(conversation_ids))