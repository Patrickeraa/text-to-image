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