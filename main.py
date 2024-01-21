import sdkit
from sdkit.models import load_model, download_model, resolve_downloaded_model_path
from sdkit.generate import generate_images
from sdkit.utils import log
import random
import requests
import os
from tqdm import tqdm
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.llms import CTransformers
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
import json

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


parser = JsonOutputParser()

llm = CTransformers(
    model="TheBloke/Mistral-7B-Instruct-v0.2-GGUF", 
    model_file="mistral-7b-instruct-v0.2.Q4_K_M.gguf", 
    model_type="mistral", 
    callbacks=[StreamingStdOutCallbackHandler()],
    config={"context_length": 10000}
)


template = """You are a Stable Diffusion prompt creator. You construct detailed but brief prompts in JSON format based on what the user wants.
For example, let's say that the user says: I want a semi-realistic picture of a sorceress. 

Here's the kind of output you would give:
{{
    "positive": "a beautiful and powerful mysterious sorceress, smile, sitting on a rock, lightning magic, hat, 
                  detailed leather clothing with gemstones, dress, castle background, digital art, hyperrealistic, 
                  fantasy, dark art, artstation, highly detailed, sharp focus",
    "negative": "ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, extra limbs, 
                  disfigured, deformed, body out of frame, bad anatomy, watermark, signature, cut off, low contrast, 
                  underexposed, overexposed, bad art, beginner, amateur, distorted face"
}}

Only output the JSON and don't add any commentary. Stop once you have outputted the json.

Showtime!

{request}
"""

prompt = PromptTemplate(template=template, 
                        input_variables=["request"]
                        )

llm_chain = prompt | llm | parser

response = llm_chain.invoke({"request": "I want a photorealistic picture of a sea turtle."})

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