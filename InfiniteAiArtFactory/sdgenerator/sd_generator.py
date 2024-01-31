import sdkit
from sdkit.models import load_model, download_model, resolve_downloaded_model_path
from sdkit.generate import generate_images
from sdkit.utils import log
import random
import requests
import os
from tqdm import tqdm
import time

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

def prep_and_generate_images(positive, negative):
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
                            prompt=positive, 
                            negative_prompt=negative,
                            seed=sd_seed, 
                            sampler_name='dpmpp_2m',
                            num_outputs=1,
                            num_inference_steps=35,
                            guidance_scale=7,
                            width=512, 
                            height=512)


    timestamp = int(time.time() * 1000)


    # save the images
    for i, img in enumerate(images):
        img.save(f"images/image#{i+1}@{timestamp}.png") # images is a list of PIL.Image

    log.info("Generated images!")