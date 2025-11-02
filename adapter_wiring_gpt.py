# shared_text_adapter.py
# Requirements: diffusers>=0.29, transformers>=4.42, torch>=2.1, scipy
# Device note: keep float32 on CPU/MPS. Switch to "mps" manually if desired.

import math
import types
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers import StableDiffusionPipeline, AudioLDM2Pipeline
from transformers import (
    T5Tokenizer, T5EncoderModel,           # shared encoder
    ClapModel, AutoTokenizer,              # target for AudioLDM2 branch
)
torch.set_grad_enabled(True)


# -----------------------------
# 0) Small utils
# -----------------------------
def masked_mean(x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    """
    x: [B, L, D], mask: [B, L] (1 = keep), returns [B, D]
    """
    if mask is None:
        return x.mean(dim=1)
    w = mask.float()
    w = w / (w.sum(dim=1, keepdim=True).clamp(min=1))
    return (x * w.unsqueeze(-1)).sum(dim=1)


def cosine_loss(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return 1.0 - F.cosine_similarity(a, b, dim=-1).mean()


# -----------------------------
# 1) Adapters
# -----------------------------
class SeqProjectTo77(nn.Module):
    """
    Maps a long T5 token sequence [B, L_in, d_in] to the CLIP-like sequence [B, 77, 768].
    Lightweight: per-token projection + brief mixing + learned query pooling to exactly 77 tokens.
    """
    def __init__(self, d_in: int, d_out: int = 768, L_target: int = 77, n_heads: int = 8, n_layers: int = 2):
        super().__init__()
        self.L_target = L_target
        self.proj_in = nn.Linear(d_in, d_out)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_out, nhead=n_heads, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        # learned queries to pool down to 77
        self.q = nn.Parameter(torch.randn(1, L_target, d_out) / math.sqrt(d_out))
        self.norm = nn.LayerNorm(d_out)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: [B, L_in, d_in], mask: [B, L_in] (1=keep)
        h = self.proj_in(x)  # [B, L_in, d_out]
        pad_mask = (mask == 0) if mask is not None else None
        h = self.encoder(h, src_key_padding_mask=pad_mask)  # [B, L_in, d_out]

        # content pooling: queries attend over h
        attn = torch.softmax((self.q @ h.transpose(1, 2)) / math.sqrt(h.size(-1)), dim=-1)  # [1, 77, L_in]
        attn = attn.expand(h.size(0), -1, -1)  # [B, 77, L_in]
        y = attn @ h  # [B, 77, d_out]
        y = self.norm(y)
        return y


class T5toCLAP(nn.Module):
    """
    Pooled T5 → CLAP vector.
    """
    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_in),
            nn.Linear(d_in, d_in),
            nn.GELU(),
            nn.Linear(d_in, d_out),
        )

    def forward(self, t5_hidden: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        pooled = masked_mean(t5_hidden, mask)   # [B, d_in]
        return self.net(pooled)                  # [B, d_out]


# -----------------------------
# 2) Load models (frozen)
# -----------------------------
@dataclass
class ModelPack:
    sd: StableDiffusionPipeline
    sd_tokenizer: any
    sd_text_encoder: nn.Module

    aud: AudioLDM2Pipeline
    clap_tokenizer: AutoTokenizer
    clap_text_encoder: ClapModel

    t5_tokenizer: T5Tokenizer
    t5_encoder: T5EncoderModel

    d_t5: int
    d_clip: int
    d_clap: int
    L_sd: int

def build_models(device: str = "cpu") -> ModelPack:
    # SD
    sd = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float32
    ).to(device)
    sd.unet.requires_grad_(False)
    sd.vae.requires_grad_(False)
    sd_text_encoder = sd.text_encoder.eval()
    sd_tokenizer = sd.tokenizer
    L_sd = sd_tokenizer.model_max_length          # 77
    d_clip = sd_text_encoder.config.hidden_size   # 768

    # AudioLDM2
    aud = AudioLDM2Pipeline.from_pretrained(
        "cvssp/audioldm2", torch_dtype=torch.float32
    ).to(device)
    aud.unet.requires_grad_(False)
    aud.vae.requires_grad_(False)

    # CLAP (text tower)
    clap_tokenizer = AutoTokenizer.from_pretrained("laion/clap-htsat-unfused")
    clap_text_encoder = ClapModel.from_pretrained("laion/clap-htsat-unfused").eval().to(device)
    # pooled projection dim used by get_text_features()
    d_clap = clap_text_encoder.config.projection_dim

    # Shared T5
    t5_name = "google/flan-t5-large"
    t5_tokenizer = T5Tokenizer.from_pretrained(t5_name)
    t5_encoder = T5EncoderModel.from_pretrained(t5_name).eval().to(device)
    d_t5 = t5_encoder.config.d_model

    return ModelPack(
        sd=sd, sd_tokenizer=sd_tokenizer, sd_text_encoder=sd_text_encoder,
        aud=aud, clap_tokenizer=clap_tokenizer, clap_text_encoder=clap_text_encoder,
        t5_tokenizer=t5_tokenizer, t5_encoder=t5_encoder,
        d_t5=d_t5, d_clip=d_clip, d_clap=d_clap, L_sd=L_sd
    )


# -----------------------------
# 3) Targets for distillation
# -----------------------------
@torch.no_grad()
def target_clip_seq(sd_tokenizer, sd_text_encoder, prompts: List[str], device: str):
    ids = sd_tokenizer(
        prompts, padding="max_length", max_length=sd_tokenizer.model_max_length,
        truncation=True, return_tensors="pt"
    ).to(device)
    hid = sd_text_encoder(input_ids=ids["input_ids"], attention_mask=ids["attention_mask"]).last_hidden_state
    return hid, ids["attention_mask"]  # [B,77,768], [B,77]

@torch.no_grad()
def target_clap_vec(clap_tokenizer, clap_text_encoder, prompts: List[str], device: str):
    ids = clap_tokenizer(prompts, padding=True, truncation=True, return_tensors="pt").to(device)
    vec = clap_text_encoder.get_text_features(
        input_ids=ids["input_ids"], attention_mask=ids["attention_mask"]
    )
    return vec  # [B, d_clap]


# -----------------------------
# 4) Shared encoder forward
# -----------------------------
@torch.no_grad()
def shared_t5_hidden(t5_tokenizer, t5_encoder, prompts: List[str], device: str):
    ids = t5_tokenizer(prompts, padding="max_length", max_length=256,  # cap length safely
                       truncation=True, return_tensors="pt").to(device)
    out = t5_encoder(input_ids=ids["input_ids"], attention_mask=ids["attention_mask"])
    return out.last_hidden_state, ids["attention_mask"]  # [B, L_t5, d_t5], [B, L_t5]


# -----------------------------
# 5) Trainer
# -----------------------------
class AdapterTrainer:
    def __init__(self, pack: ModelPack, device: str = "cpu", lr: float = 1e-4, wd: float = 1e-2):
        self.pack = pack
        self.device = device

        self.A_sd = SeqProjectTo77(d_in=pack.d_t5, d_out=pack.d_clip, L_target=pack.L_sd).to(device)
        self.A_clap = T5toCLAP(d_in=pack.d_t5, d_out=pack.d_clap).to(device)

        self.opt = torch.optim.AdamW(list(self.A_sd.parameters()) + list(self.A_clap.parameters()),
                                     lr=lr, weight_decay=wd)

    def train_step(self, prompts: List[str]) -> float:
        # targets
        clip_seq, _ = target_clip_seq(self.pack.sd_tokenizer, self.pack.sd_text_encoder, prompts, self.device)
        clap_vec = target_clap_vec(self.pack.clap_tokenizer, self.pack.clap_text_encoder, prompts, self.device)

        # shared
        t5_hid, t5_mask = shared_t5_hidden(self.pack.t5_tokenizer, self.pack.t5_encoder, prompts, self.device)

        # preds
        y_sd = self.A_sd(t5_hid, t5_mask)               # [B,77,768]
        z_clap = self.A_clap(t5_hid, t5_mask)           # [B,d_clap]

        # losses
        loss_sd = F.mse_loss(y_sd, clip_seq) + 0.5 * cosine_loss(y_sd.flatten(1), clip_seq.flatten(1))
        loss_clap = F.mse_loss(z_clap, clap_vec) + 0.5 * cosine_loss(z_clap, clap_vec)
        loss = loss_sd + loss_clap

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return float(loss.item())

    # checkpoint helpers
    def state_dict(self):
        return {"A_sd": self.A_sd.state_dict(), "A_clap": self.A_clap.state_dict()}

    def load_state_dict(self, sd):
        self.A_sd.load_state_dict(sd["A_sd"])
        self.A_clap.load_state_dict(sd["A_clap"])


# -----------------------------
# 6) Inference wiring
# -----------------------------
def sd_generate_with_shared(pack: ModelPack, A_sd: SeqProjectTo77,
                            prompts: List[str], num_inference_steps=30, guidance_scale=7.5):
    # get shared T5
    t5_hid, t5_mask = shared_t5_hidden(pack.t5_tokenizer, pack.t5_encoder, prompts, device=str(next(A_sd.parameters()).device))
    # map to CLIP-like sequence
    h_clip_like = A_sd(t5_hid, t5_mask)  # [B,77,768]
    # SD accepts prompt_embeds directly
    images = pack.sd(
        prompt_embeds=h_clip_like,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale
    ).images
    return images


def _monkeypatch_audioldm_encode_prompt(aud_pipe: AudioLDM2Pipeline,
                                        clap_text_vec: torch.Tensor,
                                        t5_hidden: torch.Tensor):
    """
    AudioLDM2 internally encodes prompt using CLAP + T5 and then composes conditionings.
    We override its encoder to return our precomputed embeddings.

    This patch is purposely lightweight: it replaces the private encode method if present,
    else adds a new attribute consumed by a small wrapper around __call__.
    """
    # Many diffusers pipelines use a helper like self._encode_prompt or encode_prompt.
    # We replace it if found; otherwise we wrap __call__.
    def custom_encode_prompt(*args, **kwargs):
        # Return a tuple that matches what the original encode returns.
        # In AudioLDM2, the UNet consumes (cond_embeddings, text_embeddings, attention_mask, …).
        # We’ll pass both t5_hidden (seq) and clap_text_vec (pooled) in a dict for clarity, and
        # the patched __call__ branch will read them.
        return {
            "t5_hidden": t5_hidden,             # [B, L_t5, d_t5]
            "clap_text_vec": clap_text_vec      # [B, d_clap]
        }

    if hasattr(aud_pipe, "encode_prompt"):
        aud_pipe._orig_encode_prompt = aud_pipe.encode_prompt
        aud_pipe.encode_prompt = types.MethodType(lambda self, *a, **k: custom_encode_prompt(), aud_pipe)
        return "encode_prompt"
    elif hasattr(aud_pipe, "_encode_prompt"):
        aud_pipe._orig_encode_prompt = aud_pipe._encode_prompt
        aud_pipe._encode_prompt = types.MethodType(lambda self, *a, **k: custom_encode_prompt(), aud_pipe)
        return "_encode_prompt"
    else:
        # Fallback: wrap __call__ minimally and stash the tensors
        aud_pipe._shared_t5_hidden = t5_hidden
        aud_pipe._shared_clap_text = clap_text_vec
        return "fallback"


@torch.no_grad()
def audioldm_generate_with_shared(pack, A_clap, prompts,
                                  num_inference_steps=200, audio_length_in_s=10.0,
                                  guidance_scale=3.5, seed=42):
    device = str(next(A_clap.parameters()).device)

    # 1) Shared T5 features
    t5_hid, t5_mask = shared_t5_hidden(pack.t5_tokenizer, pack.t5_encoder, prompts, device)

    # 2) Map to CLAP space (pooled)
    clap_vec = A_clap(t5_hid, t5_mask)  # [B, d_clap]

    # Minimal, shape-stable packaging:
    # Use T5 sequence as the "prompt_embeds" and a real attention mask.
    # For "generated_prompt_embeds" provide the pooled CLAP vector (broadcast to seq len 1).
    B, L_t5, D_t5 = t5_hid.shape
    prompt_embeds = t5_hid
    attention_mask = t5_mask
    generated_prompt_embeds = clap_vec.unsqueeze(1)     # [B,1,d_clap]

    gen = torch.manual_seed(seed)
    out = pack.aud(
        # NOTE: pass the 3 tensors; the pipeline should skip encode_prompt internally
        prompt_embeds=prompt_embeds,
        attention_mask=attention_mask,
        generated_prompt_embeds=generated_prompt_embeds,
        num_inference_steps=num_inference_steps,
        audio_length_in_s=audio_length_in_s,
        guidance_scale=guidance_scale,
        generator=gen,
    )
    return out.audios


# -----------------------------
# 7) Demo / quick sanity run
# -----------------------------
if __name__ == "__main__":
    device = "cpu"  # change to "mps" on Apple Silicon if desired
    pack = build_models(device=device)

    # Quick shape probe
    sample = ["a serene forest with birds chirping at dawn"]
    with torch.no_grad():
        # SD targets
        clip_seq, _ = target_clip_seq(pack.sd_tokenizer, pack.sd_text_encoder, sample, device)
        print("SD target (CLIP) seq:", tuple(clip_seq.shape))  # (1,77,768)
        # CLAP target
        clap_vec = target_clap_vec(pack.clap_tokenizer, pack.clap_text_encoder, sample, device)
        print("CLAP target vec:", tuple(clap_vec.shape))       # (1, d_clap)
        # Shared T5
        t5_hid, t5_mask = shared_t5_hidden(pack.t5_tokenizer, pack.t5_encoder, sample, device)
        print("Shared T5 hidden:", tuple(t5_hid.shape))        # (1, L_t5, d_t5)

    # Build trainer + do a couple of steps just to verify it runs
    trainer = AdapterTrainer(pack, device=device, lr=1e-4, wd=1e-2)
    prompts_pool = [
        "a serene forest at dawn",
        "a busy city street with honking cars",
        "soft piano with gentle rain ambience",
        "a cat sleeping near a window, cinematic light",
        "epic orchestral music with heavy percussion",
    ]
    for step in range(3):
        batch = random.sample(prompts_pool, k=min(4, len(prompts_pool)))
        loss = trainer.train_step(batch)
        print(f"warmup step {step}: loss={loss:.4f}")

    # Inference example (will be mediocre until trained properly)
    imgs = sd_generate_with_shared(pack, trainer.A_sd, sample, num_inference_steps=20)
    imgs[0].save("shared_sd_test.png")
    print("Saved shared_sd_test.png")

    audios = audioldm_generate_with_shared(pack, trainer.A_clap, sample, num_inference_steps=50, audio_length_in_s=4.0)
    import scipy.io.wavfile
    scipy.io.wavfile.write("shared_audioldm_test.wav", rate=16000, data=audios[0])
    print("Saved shared_audioldm_test.wav")
