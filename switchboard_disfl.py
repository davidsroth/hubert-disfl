import audio_utils
import text_utils
from torch.utils.data import Dataset

class SwitchboardDisfluencyDataset(Dataset):
    """
    A torch.utils.data.Dataset wrapper for the Switchboard dataset with Disfluency annotations.
    """
    def __init__(self, conversation_ids, target_sr=16000):
        self.target_sr = 16000
        self.data = text_utils.extract(conversation_ids)
        # assert len(self.targets) == len(self.input)
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        target = self.data[i]
        return {
            'audio': audio_utils.get_conversation_slice(
                target["conversation_id"],
                target['start_time'],
                target['end_time'],
                self.target_sr
            ),
            'target_text': self.data[i]["target_text"],
        }
