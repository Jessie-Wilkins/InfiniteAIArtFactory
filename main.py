import sdkit
from sdkit.models import load_model, download_model, resolve_downloaded_model_path
from sdkit.generate import generate_images
from sdkit.utils import log
import random
import requests
import os
from tqdm import tqdm
import argparse
from InfiniteAiArtFactory.llm import prompt_generator

def download_file(url, destination):
    local_filename = url.split('/')[-1]+'.safetensors'

    if os.path.isfile(os.path.join(destination, local_filename)):
        print("File", url, "already exists.")
        return os.path.join(destination, local_filename)
    
    if not os.path.exists(destination):
        os.makedirs(destination)
    
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        
        total_size = int(r.headers.get('Content-Length', 0))
        
        # progress bar
        progress = tqdm(total=total_size, unit='B', unit_scale=True)
        
        with open(os.path.join(destination, local_filename), 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192): 
                if chunk: # filter out keep-alive new chunks
                    progress.update(len(chunk))
                    f.write(chunk)
        return os.path.join(destination, local_filename)

def generate_seed():
    return random.randint(0, 100) # Generate a random seed between 0 and 100

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--prompt', type=str, default='ignore this, please', help='The simple prompt to pass to the commandline')
arg_parser.add_argument('--auto', type=str, default=False, help='Option to turn on auto mode without prompts')
arg_parser.add_argument('--theme', type=str, default='random', help='Theme to pass to auto mode to direct what kind of images should be automatically created')

args = arg_parser.parse_args()

response = prompt_generator.generate_prompt(user_prompt=args.prompt, auto=args.auto, theme=args.theme)

sd_seed = generate_seed()


context = sdkit.Context()

# set the path to the model file on the disk (.ckpt or .safetensors file)
path = download_file('https://civitai.com/api/download/models/274039', 'models')
context.model_paths['stable-diffusion'] = path
load_model(context, 'stable-diffusion')
    
if not os.path.exists('images'):
    os.makedirs('images')

# generate the image
images = generate_images(context, 
                         prompt=response["positive"], 
                         negative_prompt=response["negative"],
                         seed=sd_seed, 
                         sampler_name='dpmpp_2m',
                         num_outputs=4,
                         num_inference_steps=35,
                         guidance_scale=7,
                         width=512, 
                         height=512)

# save the images
for i, img in enumerate(images):
    img.save(f"images/image{i+1}.png") # images is a list of PIL.Image

log.info("Generated images!")