import pandas as pd
import audio_utils
import text_utils
from torch.utils.data import Dataset

class SwitchboardDisfluencyDataset(Dataset):
    """
    A torch.utils.data.Dataset wrapper for the Switchboard dataset with Disfluency annotations.
    """
    def __init__(self, dataframe, audio_tokenizer, text_tokenizer):
        self.input_ids = audio_utils.encode_data(dataframe, audio_tokenizer)
        self.targets = text_utils.extract_targets(dataframe)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, i):
        return {
            'input_ids': self.input_ids[i],
            'target': self.targets[i]
        }
