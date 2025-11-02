# shape_test.py  (fixed)
import torch
from diffusers import StableDiffusionPipeline, AudioLDM2Pipeline

prompts = ["a serene forest with birds chirping at dawn"]

# --- SD (CLIP) ---
sd = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float32)
tok_clip = sd.tokenizer
enc_clip = sd.text_encoder.eval()

ids_clip = tok_clip(prompts, padding="max_length",
                    max_length=tok_clip.model_max_length,
                    truncation=True, return_tensors="pt")
with torch.no_grad():
    h_clip = enc_clip(input_ids=ids_clip["input_ids"],
                      attention_mask=ids_clip["attention_mask"]).last_hidden_state
print("SD tokens:", ids_clip["input_ids"].shape)   # [B, 77]
print("SD hidden:", h_clip.shape)                  # [B, 77, 768]

# --- AudioLDM2 (CLAP) ---
aud = AudioLDM2Pipeline.from_pretrained("cvssp/audioldm2", torch_dtype=torch.float32)
tok_clap = aud.tokenizer            # RobertaTokenizer fast, typically
clap = aud.text_encoder.eval()      # transformers.ClapModel

ids_clap = tok_clap(prompts, padding=True, truncation=True, return_tensors="pt")
with torch.no_grad():
    # IMPORTANT: use the text-only API
    z_clap = clap.get_text_features(input_ids=ids_clap["input_ids"],
                                    attention_mask=ids_clap["attention_mask"])
print("CLAP pooled text:", z_clap.shape)           # [B, D_clap]  (often 512)
