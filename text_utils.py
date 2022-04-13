# import torch
import sys
import pandas as pd
from parser import extract_all

def extract(conversation_ids, fluent=False, chars_to_ignore=[]):
    targets = []
    for conversation_id in conversation_ids:
        targets.extend(extract_all.extract(conversation_id, return_fluent=fluent, chars_to_ignore=chars_to_ignore))
    
    return pd.DataFrame(targets)

def get_conversation_ids_from_file(filepath):
    """
    fetches conversation_ids from text file containing one id per line.
    """
    with open(filepath, 'r') as f:
        conversation_ids = f.read().splitlines()
    return conversation_ids

if __name__ == "__main__":
    conversation_ids_file = sys.argv[1]

    conversation_ids = get_conversation_ids_from_file(conversation_ids_file)
    print(extract(conversation_ids))