# -*- coding: utf-8 -*-
# !pip install librosa
import librosa
import librosa.display # You need this in librosa to be able to plot
import os
# import matplotlib.pyplot as plt

clip_dir = os.path.join("sw02005.sph")

def encode_data(dataset, audio_tokenizer):
  pass

def load_audio_slice(filepath, start, end):#function to splice audio start=start time , end = end time in seconds
  audio,sr = librosa.load(clip_dir,sr=8000,offset=start,duration=end) # audio is a numpy array
  return audio,sr

#example
# audio,sr=load(1,4) # loads audio from t=1 to t=4 seconds
# fig, ax = plt.subplots(figsize=(15,8))
# librosa.display.waveplot(audio, sr=sr, ax=ax)
# ax.set(title="sw02005.sph waveform")