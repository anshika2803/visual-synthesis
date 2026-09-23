import torch
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained(
    "models/sd-turbo",
    variant="fp16",
    dtype=torch.float16,
    local_files_only=True
)

pipe = pipe.to("cuda")

prompt = input("Enter prompt: ")

image = pipe(
    prompt=prompt,
    num_inference_steps=1,
    guidance_scale=0.0,
    height=512,
    width=512
).images[0]

image.save("results/test.png")

print("Saved: results/test.png")
