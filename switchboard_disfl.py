import audio_utils
import text_utils
from torch.utils.data import Dataset

class SwitchboardDisfluencyDataset(Dataset):
    """
    A torch.utils.data.Dataset wrapper for the Switchboard dataset with Disfluency annotations.
    """
    def __init__(self, conversation_ids):
        self.targets = text_utils.extract_targets(conversation_ids)
        self.input = audio_utils.extract_inputs(self.targets)
        assert len(self.targets) == len(self.input)
        

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, i):
        return {
            'input_ids': self.input_ids[i],
            'target': self.targets[i]
        }
