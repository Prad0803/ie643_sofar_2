# --------------------------------------------------------------
# cross_modal_generator.py  ← FIXED VERSION
# --------------------------------------------------------------
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import T5EncoderModel, T5Tokenizer, CLIPTextModel, CLIPTokenizer
from diffusers import StableDiffusionPipeline, AudioLDM2Pipeline
import numpy as np
import os
import librosa
import scipy.io.wavfile  # ← MOVED HERE
from multiprocessing import Process

# ------------------- 1. Modular Adapter -----------------------
class ModularAdapter(nn.Module):
    def __init__(self, t5_dim=1024, clip_dim=768, num_heads=8):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim=t5_dim, num_heads=num_heads, batch_first=True)
        self.audio_proj = nn.Linear(t5_dim, t5_dim)
        self.image_proj = nn.Linear(t5_dim, clip_dim)

    def forward(self, t5_embeds):
        attn_out, _ = self.attention(t5_embeds, t5_embeds, t5_embeds)
        audio_embeds = self.audio_proj(attn_out)
        image_embeds = self.image_proj(attn_out)
        return audio_embeds, image_embeds

# ------------------- 2. Dataset -------------------------------
class TextAudioDataset(Dataset):
    def __init__(self, csv_file, audio_dir, t5_tokenizer, clip_tokenizer, max_samples=400):
        df = pd.read_csv(csv_file).head(max_samples)
        self.data = df
        self.audio_dir = audio_dir
        self.t5_tokenizer = t5_tokenizer
        self.clip_tokenizer = clip_tokenizer

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        text = row['text']
        audio_path = row['audio_path'].replace("clotho_16k/", "", 1)
        audio_path = os.path.join(self.audio_dir, audio_path)

        audio, sr = librosa.load(audio_path, sr=None)
        if sr != 16000:
            raise ValueError(f"Wrong sr {sr}")

        if idx < 5 or idx % 100 == 0:
            print(f"[LOAD] {os.path.basename(audio_path)} | sr={sr}")

        audio = (audio * 32767).astype(np.int16)

        t5_in = self.t5_tokenizer(text, return_tensors="pt", padding="max_length", max_length=77, truncation=True)
        clip_in = self.clip_tokenizer(text, return_tensors="pt", padding="max_length", max_length=77, truncation=True)

        return {
            "text": text,
            "t5_inputs": {k: v.squeeze(0) for k, v in t5_in.items()},
            "clip_inputs": {k: v.squeeze(0) for k, v in clip_in.items()},
            "audio": audio
        }

# ------------------- 3. Collate -------------------
def collate_fn(batch):
    t5_inputs = {k: torch.stack([b["t5_inputs"][k] for b in batch]) for k in batch[0]["t5_inputs"]}
    clip_inputs = {k: torch.stack([b["clip_inputs"][k] for b in batch]) for k in batch[0]["clip_inputs"]}
    audios = [b["audio"] for b in batch]
    max_len = max(a.shape[0] for a in audios)
    padded = np.array([np.pad(a, (0, max_len - a.shape[0])) for a in audios])
    audios_tensor = torch.from_numpy(padded).float()
    return {"t5_inputs": t5_inputs, "clip_inputs": clip_inputs, "audio": audios_tensor}

# ------------------- 4. Loss -------------------
def contrastive_loss(a, b, temp=0.07):
    a = a.mean(dim=1); b = b.mean(dim=1)
    a = a / a.norm(dim=-1, keepdim=True)
    b = b / b.norm(dim=-1, keepdim=True)
    logits = torch.matmul(a, b.T) / temp
    labels = torch.arange(logits.size(0), device=a.device)
    loss_i = nn.CrossEntropyLoss()(logits, labels)
    loss_j = nn.CrossEntropyLoss()(logits.T, labels)
    return (loss_i + loss_j) / 2

# ------------------- 5. Generation -------------------
def generate_image(sd_pipe, image_embeds, out_path):
    # DISABLE SAFETY CHECKER
    sd_pipe.safety_checker = None
    sd_pipe.requires_safety_checker = False
    img = sd_pipe(prompt_embeds=image_embeds,
                  num_inference_steps=50,
                  guidance_scale=7.5).images[0]
    img.save(out_path)

def generate_audio(audioldm_pipe, prompt, out_path):
    import scipy.io.wavfile  # ← CRITICAL: Import inside process
    gen = torch.manual_seed(42)
    wav = audioldm_pipe(prompt,
                        num_inference_steps=200,
                        audio_length_in_s=10.0,
                        guidance_scale=3.5,
                        generator=gen).audios[0]
    scipy.io.wavfile.write(out_path, rate=16000, data=wav)

# ------------------- 6. MAIN -------------------
if __name__ == "__main__":
    device = "cpu"  # or "mps"

    # --- Load models ---
    t5_tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-large")
    t5_model = T5EncoderModel.from_pretrained("google/flan-t5-large").to(device).eval()
    clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    clip_model = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(device).eval()

    dtype = torch.float16 if device == "mps" else torch.float32
    sd_pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", torch_dtype=dtype).to(device)
    audioldm_pipe = AudioLDM2Pipeline.from_pretrained(
        "cvssp/audioldm2", torch_dtype=dtype).to(device)

    # Freeze
    for m in (t5_model, clip_model, sd_pipe.unet, sd_pipe.vae,
              audioldm_pipe.unet, audioldm_pipe.vae):
        m.requires_grad_(False)

    # --- Adapter ---
    adapter = ModularAdapter().to(device)

    if os.path.exists("adapter.pth"):
        adapter.load_state_dict(torch.load("adapter.pth", map_location=device, weights_only=True))
        print("Loaded adapter.pth")
    else:
        print("Training adapter...")
        dataset = TextAudioDataset("dataset/metadata.csv", "dataset/clotho_16k",
                                   t5_tokenizer, clip_tokenizer, max_samples=400)
        loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

        opt = torch.optim.AdamW(adapter.parameters(), lr=1e-4)
        adapter.train()

        for epoch in range(15):
            total = 0
            for batch in loader:
                t5_in = {k: v.to(device) for k, v in batch["t5_inputs"].items()}
                clip_in = {k: v.to(device) for k, v in batch["clip_inputs"].items()}
                with torch.no_grad():
                    t5_emb = t5_model(**t5_in).last_hidden_state
                    clip_emb = clip_model(**clip_in).last_hidden_state
                _, img_emb = adapter(t5_emb)
                loss = contrastive_loss(img_emb, clip_emb)
                opt.zero_grad(); loss.backward(); opt.step()
                total += loss.item()
            print(f"Epoch {epoch+1}/15 | Loss: {total/len(loader):.4f}")
        torch.save(adapter.state_dict(), "adapter.pth")
        print("Saved adapter.pth")

    # --- Inference ---
    adapter.eval()
    prompt = "a serene forest with birds chirping at dawn."
    t5_in = t5_tokenizer([prompt], return_tensors="pt", padding="max_length",
                         max_length=77, truncation=True).to(device)
    with torch.no_grad():
        t5_emb = t5_model(**t5_in).last_hidden_state
        audio_emb, image_emb = adapter(t5_emb)

    p1 = Process(target=generate_image, args=(sd_pipe, image_emb, "output_image.png"))
    p2 = Process(target=generate_audio, args=(audioldm_pipe, prompt, "output_audio.wav"))
    p1.start(); p2.start()
    p1.join(); p2.join()

    print("Done! → output_image.png & output_audio.wav")