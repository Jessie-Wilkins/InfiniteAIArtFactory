import argparse
from InfiniteAiArtFactory.llm import prompt_generator
from InfiniteAiArtFactory.sdgenerator import sd_generator

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--prompt', type=str, default='ignore this, please', help='The simple prompt to pass to the commandline')
arg_parser.add_argument('--auto', type=str, default=False, help='Option to turn on auto mode without prompts')
arg_parser.add_argument('--theme', type=str, default='random', help='Theme to pass to auto mode to direct what kind of images should be automatically created')

args = arg_parser.parse_args()

response = prompt_generator.generate_prompt(user_prompt=args.prompt, auto=args.auto, theme=args.theme)

sd_generator.prep_and_generate_images(positive=response["positive"], negative=response["negative"])