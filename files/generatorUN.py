import torch
from diffusers import StableDiffusionPipeline, UNet2DConditionModel

model_path = "/workspace/diffusers/examples/text_to_image/sd-naruto-model"
unet = UNet2DConditionModel.from_pretrained(model_path + "/unet", torch_dtype=torch.float16)

pipe = StableDiffusionPipeline.from_pretrained("/workspace/diffusers/examples/text_to_image/sd-naruto-model", unet=unet, torch_dtype=torch.float16)
pipe.to("cuda")

image = pipe(prompt="zelda").images[0]
image.save("imagesUNET/zelda1-naruto.png")

image = pipe(prompt="zelda").images[0]
image.save("imagesUNET/zelda2-naruto.png")

image = pipe(prompt="zelda").images[0]
image.save("imagesUNET/zelda3-naruto.png")

image = pipe(prompt="zelda").images[0]
image.save("imagesUNET/zelda4-naruto.png")

image = pipe(prompt="zelda").images[0]
image.save("imagesUNET/zelda5-naruto.png")

image = pipe(prompt="zelda").images[0]
image.save("imagesUNET/zelda6-naruto.png")

image = pipe(prompt="zelda").images[0]
image.save("imagesUNET/zelda7-naruto.png")

image = pipe(prompt="zelda").images[0]
image.save("imagesUNET/zelda8-naruto.png")