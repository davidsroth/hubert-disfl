# -*- coding: utf-8 -*-
# !pip install librosa
from doctest import OutputChecker
from responses import target
import librosa
import soundfile as sf
# import librosa.display # You need this in librosa to be able to plot
import os
# import matplotlib.pyplot as plt

# clip_dir = os.path.join("sw02005.sph")

PROJECT_ROOT = os.environ.get("PROJECT_ROOT")
SWB_ROOT = f"{PROJECT_ROOT}/switchboard"
SWB_DATA_ROOT = f"{SWB_ROOT}/disks"
UPSAMPLED_DATA_ROOT = f"{SWB_ROOT}/upsampled/disks"
DISK_MAP_PATH = f"{SWB_ROOT}/docs/swb1_all.dvd.tbl"
  
def build_disk_lookup_table(file_path):
  """
  Creating lookup table (dict) with keys = conversation number (conv_num)
  and value = disk
  Params:
    file_path: str - .tbl file
  Returns:
    lookup: dict()
  """
  with open(file_path, 'r') as f:
    content = f.read().splitlines()
  
  lookup = {}
  for line in content:
    disk, conversation_id = line.split()
    conversation_id = conversation_id.split(".")[0]
    conv_num = conversation_id.partition("0")[-1]
    lookup[f"{conv_num}"] = disk
  return lookup

lookup_table = build_disk_lookup_table(DISK_MAP_PATH)

def get_conversation_filepath(conversation_id):
  """
  fetching conversation wav file
  Params:
    conversation_id: str - conversation id (like sw2005)
  Returns:
    file_path: str - wav file_path
  """
  conv_num = conversation_id[-4:]
  if not os.path.exists(os.path.join(UPSAMPLED_DATA_ROOT, lookup_table[conv_num], "data", f"sw0{conv_num}.wav")):
    result = preprocess(os.path.join(SWB_DATA_ROOT, lookup_table[conv_num], "data", f"sw0{conv_num}.sph"), os.path.join(UPSAMPLED_DATA_ROOT, lookup_table[conv_num], "data", f"sw0{conv_num}.wav"))
    if result is not None:
      raise Exception(f"Preprocessing failed for {conversation_id}")
  
  file_path = os.path.join(UPSAMPLED_DATA_ROOT, lookup_table[conv_num], "data", f"sw0{conv_num}.wav")
  return file_path

def get_conversation_slice(conversation_id, start, end):
  """
  To splice audio start=start time , end=end time in seconds
  Params:
    conversation_id: str - conversation id (like sw2005)
    start: start time
    end: end time of slice
  Returns:
    audio: np array - sliced audio from start -> end
  """
  file_path = get_conversation_filepath(conversation_id)
  audio, sr = librosa.load(file_path, sr=16_000, offset=start, duration=end) # audio is a numpy array
  return audio
  # return resample(audio, sr, target_sr)

def resample(audio, source_sr, target_sr):
  """
  resample audio with new target sample rate
  Params:
    audio: np aray
    source_sr: original sample rate
    target_sr: target sample rate
  Returns:
    audio: np array - audio with target sample rate
  """
  if source_sr == target_sr:
    return audio
  else:
    return librosa.resample(audio, source_sr, target_sr)

def extract_inputs(dataset, target_sr):
  inputs = []
  for conversation_id, start_time, end_time in dataset[['conversation_id', 'start_time', 'end_time']].itertuples():
    audio, sr = get_conversation_slice(conversation_id, start_time, end_time)
    sample = resample(audio, sr, target_sr)
    inputs.append(sample)
  
  return inputs

def preprocess(in_path, out_path, target_sr=16_000):
  """
  Preprocessing for a conversation_id - writing to out_path with target sample rate
  Params:
    in_path: str - sph file path
    out_path: str - wav file path
  """
  disk_path, file = os.path.split(out_path)
  print(f"Preprocessing {file}",end='\r')
  if not os.path.isdir(disk_path):
    os.makedirs(disk_path)
  audio, sr = librosa.load(in_path, sr=8_000)
  resampled = resample(audio, sr, target_sr)
  sf.write(out_path, resampled, target_sr)

  return None
  
#example
# audio,sr=load(1,4) # loads audio from t=1 to t=4 seconds
# fig, ax = plt.subplots(figsize=(15,8))
# librosa.display.waveplot(audio, sr=sr, ax=ax)
# ax.set(title="sw02005.sph waveform")