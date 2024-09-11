import torch
from diffusers import StableDiffusionPipeline

model_path = "/workspace/diffusers/examples/text_to_image/sd-naruto-model"
pipe = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16)
pipe.to("cuda")

image = pipe(prompt="flash").images[0]
image.save("/workspace/images/yoda-naruto.png")

image = pipe(prompt="flash").images[0]
image.save("/workspace/images/yoda2-naruto.png")

image = pipe(prompt="flash").images[0]
image.save("/workspace/images/yoda3-naruto.png")

image = pipe(prompt="flash").images[0]
image.save("/workspace/images/yoda4-naruto.png")

image = pipe(prompt="superman").images[0]
image.save("/workspace/images/zelda-naruto.png")

image = pipe(prompt="superman").images[0]
image.save("/workspace/images/zelda2-naruto.png")

image = pipe(prompt="superman").images[0]
image.save("/workspace/images/zelda3-naruto.png")

image = pipe(prompt="superman").images[0]
image.save("/workspace/images/zelda4-naruto.png")

image = pipe(prompt="sonic").images[0]
image.save("/workspace/images/sponge-naruto.png")

image = pipe(prompt="sonic").images[0]
image.save("/workspace/images/sponge2-naruto.png")

image = pipe(prompt="sonic").images[0]
image.save("/workspace/images/sponge3-naruto.png")

image = pipe(prompt="sonic").images[0]
image.save("/workspace/images/sponge4-naruto.png")