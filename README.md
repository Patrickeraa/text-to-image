# Text-to-Image Stable Diffusion Custom Training

This tutorial is a simple and straightfoward way of training a custom Stable Difusion text-to-image model using the Diffusers repository by Hugging Face. It shows how to proper start the container, setup the ambient and get ready to use your own custom dataset to generate images.

## Container
The first step is making the docker image that will be used in the fine-tuning process, for this you can use ` docker build -t difusers_img . ` to create the image and then running the Dockerfile with: `docker run --name difusers --gpus all -d difusers_img`.

After running the container, you can access it's terminal with `docker exec -it difusers /bin/bash`.

## Diffusers Repository
Inside the container, you can follow the same steps as presented in the original diffusers repository. First, run:

```
git clone https://github.com/huggingface/diffusers
cd diffusers
pip install .
```
After installing the dependencies above, you need to proper install the Accelerate library that is used in the fine-tuning to make the process faster, to avoid confusions with the instalation (that can be overwhelming) you can instead just run the following command to set default values for it's parameters:

```
python -c "from accelerate.utils import write_basic_config; write_basic_config(mixed_precision='fp16')"
```

## Naruto BLIP Captions Dataset Example

In the original tutorial in the Diffusers repository, they utilize a script that already contains a dataset ready to fine-tune. You can test this dataset iff you will using the `train_naruto.sh` file, for this you can run from outside the container (to train detached from the terminal):

```
docker exec -d -w /workspace/files difusers bash train_naruto.sh
```
If you want to train without detached mode, navigate to `/workspace/diffusers/examples/text_to_image` and run (the `max_train_steps` is set to 1000 to train faster, the default value is 15000):

```
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export DATASET_NAME="lambdalabs/naruto-blip-captions"

accelerate launch --mixed_precision="fp16"  train_text_to_image.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$DATASET_NAME \
  --use_ema \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --max_train_steps=1000 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --output_dir="sd-naruto-model"
```

After running the fine-tune, the results will be saved in the `sd-naruto-model` folder created in the `text_to_image` example directory, with the checkpoints you can start generating images! For this example, you can run the file `generator_naruto.py`, or create a custom python file with:

```
import torch
from diffusers import StableDiffusionPipeline

model_path = "/workspace/diffusers/examples/text_to_image/sd-naruto-model"
pipe = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16)
pipe.to("cuda")

image = pipe(prompt="zelda").images[0]
image.save("zelda-naruto.png")
```

The last two lines of code can be repeated changing the prompt and file names to generate multple images at once, remember that each run will generate a different batch of images. 

### Fine-tune Results
The resulting images will be saved in the same folder, this is a example of how the images can look like:

![alt text](image.png)

## Using Custom Datasets

If you want to change the dataset used in the fine-tune, you can follow the general rules for anotated datasets found [here](https://huggingface.co/docs/datasets/v2.4.0/en/image_load#imagefolder-with-metadata). 

If you already have a dataset that are in the correct format you can just use the `custom_train.sh` and change the path line for where your dataset is. If you have just a image folder without captions you can run the `blip_generator.py` file to generate a metadata file using the BLIP captions pre-trained model. 

To make the anotated images, make shure to have a folder with all images inside of it without subfolders.

```
import os
import json
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def generate_caption(image_path):
    raw_image = Image.open(image_path).convert("RGB")
    inputs = processor(raw_image, return_tensors="pt")
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption


image_folder = "path_to_image_folder" # put your image folder path here
output_json = "image_metadata.json" # default name of the output json file

image_captions = {}
for image_filename in os.listdir(image_folder):
    if image_filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')):
        image_path = os.path.join(image_folder, image_filename)
        caption = generate_caption(image_path)
        image_captions[image_filename] = {"caption": caption}
        print(f"Processed {image_filename}: {caption}")

with open(output_json, 'w') as f:
    json.dump(image_captions, f, indent=4)

print(f"Captions saved to {output_json}")
```


