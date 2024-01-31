# InfiniteAIArtFactory
A Stable Diffusion and Mistral powered python project that autonomously creates original art based on a theme.
It can also be used to expand a simple prompt into a more elaborate and detailed prompt.

## Requirements
* Python 3.10 or above
* Nvidia GPU 8 GB or more

## How To Install
* `pip install -r requirements`

## How to Run
* For prompt expansion: `python main.py --prompt "<your prompt here>"`
* The prompt can be something as simple as "brown dog"; the LLM will expand the prompt for you.
* For auto mode: `python main.py --auto True --theme "<your theme here>" to produce`
* Themes can be prompts such as "dark fantasy" or "impressionist landscape"
* See other options with the following command: `python main.py --help`

## Limitations
* Right now, the models are hard-coded (Mistral LLM: TheBloke/Mistral-7B-Instruct-v0.2-GGUF; Stable Diffusion model: Juggernaut). These cannot be changed via the command line.
* Resolution is fixed at 512*512.
* Various LLM and Stable Diffusion parameters are fixed.
* Per usual limitations of generative AI, generations may vary in quality and accuracy.