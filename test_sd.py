# test_sd.py
import torch
from diffusers import StableDiffusionPipeline

device = "cpu"   # change to "mps" if you want Apple-silicon GPU (optional)

# Load Stable Diffusion (no float16 on M1/M2)
sd_pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float32   # keep float32 on CPU/MPS
).to(device)

sd_pipe.unet.requires_grad_(False)
sd_pipe.vae.requires_grad_(False)

prompt = "a serene forest with birds chirping at dawn"
image = sd_pipe(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]
image.save("sd_test_image.png")
print("Stable Diffusion image saved as sd_test_image.png")