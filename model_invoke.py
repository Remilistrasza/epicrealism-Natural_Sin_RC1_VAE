from diffusers import StableDiffusionPipeline
from huggingface_hub import hf_hub_download
import torch
import json

model = hf_hub_download(repo_id="Remilistrasza/ClustroAI_LoRA", filename="ghibli_style_offset.safetensors")
pipe = StableDiffusionPipeline.from_single_file(model,
                                                torch_dtype=torch.float16, 
                                                safety_checker=None)
pipe = pipe.to("cuda")

def invoke(input_text):
    input_json = json.loads(input_text)
    prompt = input_json['prompt']
    negative_prompt = input_json['negative_prompt'] if 'negative_prompt' in input_json else ""
    steps = int(input_json['steps']) if 'steps' in input_json else 50
    image = pipe(prompt=prompt, negative_prompt=negative_prompt, num_inference_steps=steps).images[0]
    image.save("generated_image.png")
    return "generated_image.png"
