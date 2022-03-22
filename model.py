from transformers import Wav2Vec2Processor, HubertModel, HubertConfig
from datasets import load_dataset
import soundfile as sf


processor = Wav2Vec2Processor.from_pretrained('facebook/hubert-large-ls960-ft')
model = HubertModel.from_pretrained("facebook/hubert-large-ls960-ft")

dataset = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")
dataset=dataset.sort("id")
sampling_rate = dataset.features["audio"].sampling_rate

# audio file is decoded on the fly
inputs = processor(dataset[0]["audio"]["array"], sampling_rate=sampling_rate, return_tensors="pt")
with torch.no_grad():
    logits = model(**inputs).logits

predicted_ids = torch.argmax(logits, dim=-1)

# transcribe speech
transcription = processor.batch_decode(predicted_ids)
print(transcription[0])

