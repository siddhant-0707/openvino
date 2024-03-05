from transformers import AutoProcessor, MusicgenForConditionalGeneration
import torch
import scipy
import openvino.torch

processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")

inputs = processor(
    text=["trap song with piano, heavy drums and an 808 bass"],
    padding=True,
    return_tensors="pt",
)

compiled_model = torch.compile(model, backend="openvino")
with torch.no_grad():
    # output = compiled_model(**inputs, max_new_tokens=256)
    output = compiled_model.generate(**inputs, max_new_tokens=256)

sampling_rate = model.config.audio_encoder.sampling_rate
scipy.io.wavfile.write("musicgen_out.wav", rate=sampling_rate, data=output[0, 0].numpy())
