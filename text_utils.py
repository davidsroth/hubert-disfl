# import torch
import sys
import pandas as pd
from audio_utils import get_conversation_filepath
from parser import extract_all

def extract(conversation_ids, fluent=False, min_length=0, max_length=20, verbose=False):
    """
    Extracts all target labels for each sentence using parser/extract_all.py
    Params:
        conversation_ids: List of conversation ids 
        fluent: Boolean - if only fluent labels required
        chars_to_ignore: List of any words to ignore
        min_length: Float - minimum length utterance to include
        max_length: Float - maximum length utterance to include
    """
    targets = []
    for conversation_id in conversation_ids:
        sentences = extract_all.extract(conversation_id, return_fluent=fluent, min_length=min_length, max_length=max_length, verbose=verbose)
        for sentence in sentences:
            sentence["audio"] = get_conversation_filepath(conversation_id)
        targets.extend(sentences)
    
    return pd.DataFrame(targets)

def extract_w_disfluency_tags(conversation_ids, verbose=False):
    targets = []
    for conversation_id in conversation_ids:
        sentences = extract_all.extract_w_tags(conversation_id, verbose=verbose)
        targets.extend(sentences)
    return pd.DataFrame(targets)

def get_conversation_ids_from_file(filepath):
    """
    fetches conversation_ids from text file containing one id per line.
    Params:
        filepath: str - filepath containing all conversation ids
    """
    with open(filepath, 'r') as f:
        conversation_ids = f.read().splitlines()
    return conversation_ids

if __name__ == "__main__":
    conversation_ids_file = sys.argv[1]

    conversation_ids = get_conversation_ids_from_file(conversation_ids_file)
    print(extract(conversation_ids))