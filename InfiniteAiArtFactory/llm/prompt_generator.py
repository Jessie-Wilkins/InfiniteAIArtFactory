from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.llms import CTransformers
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
import json

def generate_prompt(user_prompt, auto, theme):

    parser = JsonOutputParser()

    llm = CTransformers(
        model="TheBloke/Mistral-7B-Instruct-v0.2-GGUF", 
        model_file="mistral-7b-instruct-v0.2.Q4_K_M.gguf", 
        model_type="mistral", 
        callbacks=[StreamingStdOutCallbackHandler()],
        config={"context_length": 10000, "gpu_layers": 100}
    )

    auto_template = """You are a Stable Diffusion prompt creator. You construct detailed but brief prompts in JSON format based
    based on a user-given a theme or randomly based on a theme of your choosing. For example, 
    let's say that the user says: "dark fantasy"

    Here's the kind of output you would give:
    {{
        "positive": "a beautiful and powerful mysterious sorceress, smile, sitting on a rock, lightning magic, hat, 
                    detailed leather clothing with gemstones, dress, castle background, digital art, hyperrealistic, 
                    fantasy, dark art, artstation, highly detailed, sharp focus",
        "negative": "ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, extra limbs, 
                    disfigured, deformed, body out of frame, bad anatomy, watermark, signature, cut off, low contrast, 
                    underexposed, overexposed, bad art, beginner, amateur, distorted face"
    }}    

    Note that we go from very generic to a specific subject. 
    Make sure to always do that: go from a generic theme to a specific detailed single subject.
    Please do not just repeat the theme as the main subject; build from there to a specific interesting subject.

    Only output the JSON and don't add any commentary or weird separating characters like a series of dashes. Stop once you have outputted the json.

    Showtime!

    {request}
    ------------------------
    """

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
    

    Only output the JSON and don't add any commentary or weird separating characters like a series of dashes. Stop once you have outputted the json.

    Showtime!

    {request}
    ------------------------
    """

    user_input = user_prompt

    if(auto):
        template = auto_template
        user_input = theme

    prompt = PromptTemplate(template=template, 
                            input_variables=["request"]
                            )

    llm_chain = prompt | llm | parser

    response = llm_chain.invoke({"request": user_input})

    return response
