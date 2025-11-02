# layer_check.py
import torch
import torch.nn as nn
import scipy.io.wavfile
from diffusers import AudioLDM2Pipeline
from typing import Optional

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load pipeline
pipe = AudioLDM2Pipeline.from_pretrained(
    "cvssp/audioldm2",
    torch_dtype=torch.float32,
).to(device)

pipe.unet.requires_grad_(False)
pipe.vae.requires_grad_(False)

# ----------------------------------------------------------------------
# 1. Projection Layer
# ----------------------------------------------------------------------
class ClipShapeProjection(nn.Module):
    def __init__(self, clap_dim=512, t5_dim=1024, target_seq=77, target_dim=768):
        super().__init__()
        self.clap_proj = nn.Linear(clap_dim, target_dim, bias=False)
        self.t5_proj = nn.Linear(t5_dim, target_dim, bias=False)
        self.target_seq = target_seq

    def forward(self, clap_hidden, t5_hidden):
        B = clap_hidden.shape[0]
        clap_proj = self.clap_proj(clap_hidden)
        t5_proj = self.t5_proj(t5_hidden)
        concat = torch.cat([clap_proj, t5_proj], dim=1)

        if concat.shape[1] > self.target_seq:
            concat = concat[:, :self.target_seq, :]
        else:
            pad = self.target_seq - concat.shape[1]
            padding = torch.zeros(B, pad, 768, device=concat.device, dtype=concat.dtype)
            concat = torch.cat([concat, padding], dim=1)
        return concat

projection = ClipShapeProjection().to(device)

# ----------------------------------------------------------------------
# 2. PATCH USING MONKEY-PATCH WITH CORRECT SELF
# ----------------------------------------------------------------------
original_encode_prompt = pipe.encode_prompt

def make_patched_encode_prompt(original):
    def patched(self, *args, **kwargs):
        # Extract args properly
        prompt = kwargs.get("prompt") or args[0]
        device = kwargs.get("device") or args[1]
        num_waveforms_per_prompt = kwargs.get("num_waveforms_per_prompt") or args[2]
        do_classifier_free_guidance = kwargs.get("do_classifier_free_guidance", True)
        negative_prompt = kwargs.get("negative_prompt")
        prompt_embeds = kwargs.get("prompt_embeds")
        negative_prompt_embeds = kwargs.get("negative_prompt_embeds")

        if prompt_embeds is not None:
            return prompt_embeds, None, negative_prompt_embeds or prompt_embeds

        # Call original
        result = original(self, prompt, device, num_waveforms_per_prompt,
                          do_classifier_free_guidance, negative_prompt)
        clap_hidden, t5_hidden, clap_pooled, t5_pooled = result

        # ------------------- PRINT BEFORE -------------------
        print("\n" + "="*60)
        print("BEFORE PROJECTION")
        print(f"CLAP hidden shape : {clap_hidden.shape}")
        print(f"T5 hidden shape   : {t5_hidden.shape}")

        # ------------------- APPLY PROJECTION -------------------
        projected = projection(clap_hidden, t5_hidden)

        # ------------------- PRINT AFTER -------------------
        print("\nAFTER PROJECTION")
        print(f"Projected shape   : {projected.shape}  ← matches CLIP Text (ViT-L/14)")
        print("="*60 + "\n")

        return projected, None, projected

    return patched

# Apply patch correctly
pipe.encode_prompt = make_patched_encode_prompt(original_encode_prompt)

# ----------------------------------------------------------------------
# 3. Run inference
# ----------------------------------------------------------------------
prompt = "a serene forest with birds chirping at dawn"
generator = torch.manual_seed(42)

print("Starting generation...")
try:
    audio = pipe(
        prompt,
        num_inference_steps=50,
        audio_length_in_s=10.0,
        guidance_scale=3.5,
        generator=generator,
    ).audios[0]

    # Ensure audio is numpy + float32
    if isinstance(audio, torch.Tensor):
        audio = audio.cpu().numpy()
    if audio.dtype != 'float32':
        audio = audio.astype('float32')

    scipy.io.wavfile.write("audioldm2_clipshaped.wav", rate=16000, data=audio)
    print("\nAudio saved as audioldm2_clipshaped.wav")

except Exception as e:
    print(f"Generation failed: {e}")
    # Save silence
    silence = torch.zeros(160000).numpy()
    scipy.io.wavfile.write("audioldm2_clipshaped.wav", rate=16000, data=silence)
    print("Saved silence due to error.")