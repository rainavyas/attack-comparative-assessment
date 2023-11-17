'''
    Find a sensible initialization phrase
'''

import random
from openai import OpenAI

from src.tools.args import core_args, attack_args, initialization_args
from src.data.load_data import load_data
from src.tools.tools import set_seeds

if __name__ == "__main__":

    # get command line arguments
    core_args, c = core_args()
    attack_args, a = attack_args()
    init_args, p = initialization_args()

    # set seeds
    set_seeds(core_args.seed)

    if init_args.bland:
        
        data, _ = load_data(core_args)

        # create a new combined context
        combined = ''
        for sample in data:
            context = sample.context
            sentences = context.split('.')
            sentence = random.sample(sentences, 1)[0] + '.'
            combined += sentence
        
        # chatgpt summary
        client = OpenAI()
        msgs = [{"role": "system", "content": "You are a summarization system."}]
        msgs.append({"role": "user", "content": f'Give a single 15 word phrase summary of:\n{combined}'})
        response = client.chat.completions.create(model='gpt-3.5-turbo', messages=msgs)
        summary = response.choices[0].message.content

        print(f'Combined context\n{combined}')
        print()
        print(f'Summary\n{summary}')