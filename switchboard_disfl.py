from distutils import text_file
import audio_utils
import text_utils
import re
from torch.utils.data import Dataset

class SwitchboardDisfluencyDataset(Dataset):
    """
    A torch.utils.data.Dataset wrapper for the Switchboard dataset with Disfluency annotations.
    """
    def __init__(self, conversation_ids, target_sr=16000, chars_to_ignore=None):
        self.target_sr = 16000
        self.data = text_utils.extract(conversation_ids)
        self.chars_to_ignore = chars_to_ignore
        self.chars_to_ignore_regex = (
                f'[{"".join(self.chars_to_ignore)}]' if self.chars_to_ignore is not None else None
        )
        # assert len(self.targets) == len(self.input)
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        target = self.data[i]
        target_text = re.sub(self.chars_to_ignore_regex, "", target['target_text']).lower() + " " if self.chars_to_ignore is not None else target['target_text']
        return {
            'audio': audio_utils.get_conversation_slice(
                target["conversation_id"],
                target['start_time'],
                target['end_time'],
                self.target_sr
            ),
            'target_text': target_text,
        }

# def remove_special_characters(batch):
#         if chars_to_ignore_regex is not None:
#             batch["target_text"] = re.sub(chars_to_ignore_regex, "", batch[text_column_name]).lower() + " "
#         else:
#             batch["target_text"] = batch[text_column_name].lower() + " "
#         return batch

def get_switchboard_disfluency_dataset(conversation_ids, target_sr):
    df = text_utils.extract(conversation_ids)
    df['audio'] = df.apply(lambda x: audio_utils.get_conversation_slice(x['conversation_id'], x['start_time'], x['end_time'], target_sr), axis=1)
    return df