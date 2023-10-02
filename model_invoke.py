from diffusers import StableDiffusionPipeline
import torch
import json

pipe = StableDiffusionPipeline.from_pretrained("Justin-Choo/epiCRealism-Natural_Sin_RC1_VAE",
                                                cache_dir="./.cache")
pipe = pipe.to("cuda")
pipe.safety_checker = None

def invoke(input_text):
    input_json = json.loads(input_text)
    prompt = input_json['prompt']
    negative_prompt = input_json['negative_prompt'] if 'negative_prompt' in input_json else ""
    steps = int(input_json['steps']) if 'steps' in input_json else 50
    image = pipe(prompt=prompt, negative_prompt=negative_prompt, num_inference_steps=steps).images[0]
    image.save("generated_image.png")
    return "generated_image.png"

invoke(
    "{\"prompt\":\"polaroid photo, night photo, photo of 24 y.o beautiful woman, pale skin, bokeh, motion blur\"," + 
    "\"negative_prompt\":\"(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime, mutated " + 
    "hands and fingers:1.4), (deformed, distorted, disfigured:1.3), poorly drawn, bad anatomy, wrong anatomy, " + 
    "extra limb, missing limb, floating limbs, disconnected limbs, mutation, mutated, ugly, disgusting, amputation\"}"
    )