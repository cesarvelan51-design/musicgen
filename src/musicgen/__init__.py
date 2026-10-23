import torch
from scipy.io.wavfile import write
from transformers import AutoProcessor, MusicgenForConditionalGeneration

MODEL_ID="facebook/musicgen-small"
DEVICE="cpu"

PROMPT="Smooth romantic reggaeton instrumental, 92 BPM, warm electric piano, soft nylon guitar, gentle dembow groove, round 808 bass, lush pads, airy vocal textures, nostalgic and heartfelt mood, clean radio-ready mix."

print("Cargando MusicGen...")

processor = AutoProcessor.from_pretrained(MODEL_ID)
model = MusicgenForConditionalGeneration.from_pretrained(MODEL_ID)

model = model.to(DEVICE)
model.eval()

print("Preparando el prompt")

imputs = processor(text=[PROMPT], padding=True, return_tensors="pt").to(DEVICE)

print("Generando musica...")

with torch.inference_mode():
    audio=model.generate(
        **inputs,
        do_sample=True,
        guidance_scale=3.0,
        max_new_tokens=200 #4s 750 ~ 15s
    )

sample_rate=model.config.audio_encoder.sampling_rate # ~32000 Hz
waveform = audio[0,0].cpu().float().numpy()

write("musica.wav", sample_rate, waveform)
print("Musica generada")
