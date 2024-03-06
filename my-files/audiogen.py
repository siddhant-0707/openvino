from audiocraft.models import AudioGen
from audiocraft.data.audio import audio_write
import os
import torch._dynamo

os.environ["PYTORCH_TRACING_MODE"] = "TORCHFX"
os.environ["OPENVINO_DEVICE"] = "CPU"
os.environ["TORCH_LOGS"] = "+dynamo"
os.environ["TORCHDYNAMO_VERBOSE"] = "1"

torch._dynamo.config.suppress_errors = True

model = AudioGen.get_pretrained('facebook/audiogen-medium', device='cpu')
model.set_generation_params(duration=5)  # generate 8 seconds.
# wav = model.generate_unconditional(1)    # generates 1 unconditional audio samples
descriptions = ['dog barking', 'sirene of an emergency vehicle', 'footsteps in a corridor']
wav = model.generate(descriptions)  # generates 3 samples.

# melody, sr = torchaudio.load('./assets/bach.mp3')
# generates using the melody from the given audio and the provided descriptions.
# wav = model.generate_with_chroma(descriptions, melody[None].expand(3, -1, -1), sr)

for idx, one_wav in enumerate(wav):
    # Will save under {idx}.wav, with loudness normalization at -14 db LUFS.
    audio_write(f'{idx}', one_wav.cpu(), model.sample_rate, strategy="loudness", loudness_compressor=True)
