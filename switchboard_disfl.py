from concurrent.futures import process
from distutils import text_file
from audio_utils import get_conversation_slice
from text_utils import extract
import re
from torch.utils.data import Dataset

class SwitchboardDisfluencyDataset(Dataset):
    """
    A torch.utils.data.Dataset wrapper for the Switchboard dataset with Disfluency annotations.
    """
    def __init__(self, conversation_ids, processor, chars_to_ignore=None, fluent=False, min_length=0, max_length=20):
        self.data = extract(conversation_ids, chars_to_ignore=chars_to_ignore, fluent=fluent, min_length=min_length, max_length=max_length)
        self.processor = processor
        # assert len(self.targets) == len(self.input)
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        data = self.data[i]
        audio = get_conversation_slice(data["conversation_id"], data["start_time"], data["end_time"])

        encoded_audio = self.processor(audio, sampling_rate=16_000)["input_values"][0]

        with self.processor.as_target_processor():
            encoded_text = self.processor(data["target_text"]).input_ids
        return {
            "input_values": encoded_audio,
            "input_length": len(encoded_audio),
            "labels": encoded_text
        }

# def remove_special_characters(batch):
#         if chars_to_ignore_regex is not None:
#             batch["target_text"] = re.sub(chars_to_ignore_regex, "", batch[text_column_name]).lower() + " "
#         else:
#             batch["target_text"] = batch[text_column_name].lower() + " "
#         return batch

# def get_switchboard_disfluency_dataset(conversation_ids, target_sr, chars_to_ignore):
#     """
#     fetch dataframe with extracted audio labels with audio array
#     Params:
#         conversation_ids: List of conversation ids
#         target_sr: val - target sample rate
#         chars_to_ignore: List of any words to ignore
#     Returns:
#         df: pd.DataFrame
#     """
#     # print("Extracting text segments")
#     df = text_utils.extract(conversation_ids, chars_to_ignore=chars_to_ignore)
#     # print("Extracting audio segments.")
#     # df['audio'] = df.apply(lambda x: audio_utils.get_conversation_slice(x['conversation_id'], x['start_time'], x['end_time']), axis=1)
#     # print("Done extracting audio segments.")
#     return df