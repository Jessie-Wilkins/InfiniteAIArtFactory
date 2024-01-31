import argparse
from InfiniteAiArtFactory.llm import prompt_generator
from InfiniteAiArtFactory.sdgenerator import sd_generator

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--prompt', type=str, default='ignore this, please', help='The simple prompt to pass to the commandline')
arg_parser.add_argument('--auto', type=bool, default=False, help='Option to turn on auto mode without prompts')
arg_parser.add_argument('--theme', type=str, default='random', help='Theme to pass to auto mode to direct what kind of images should be automatically created')
arg_parser.add_argument('--num_of_generations', type=int, default=1, help='Number of times to generate; set to -1 to generate infinitely.')

args = arg_parser.parse_args()

if(args.num_of_generations == -1):
    while True:
        response = prompt_generator.generate_prompt(user_prompt=args.prompt, auto=args.auto, theme=args.theme)

        sd_generator.prep_and_generate_images(positive=response["positive"], negative=response["negative"])

else:
    for i in range(args.num_of_generations):
        response = prompt_generator.generate_prompt(user_prompt=args.prompt, auto=args.auto, theme=args.theme)

        sd_generator.prep_and_generate_images(positive=response["positive"], negative=response["negative"])