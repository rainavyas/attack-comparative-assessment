'''
    Find a sensible initialization phrase
'''

import random
import sys
import os
import torch
from openai import OpenAI
from tqdm import tqdm
import json

from src.tools.args import core_args, attack_args, initialization_args
from src.data.load_data import load_data
from src.models import load_model
from src.models.llama import LlamaBase
from src.tools.tools import get_default_device, set_seeds
from src.tools.saving import next_dir
from src.attacker.gcg import GCGAttacker
from src.tools.saving import base_path_creator

if __name__ == "__main__":

    # get command line arguments
    core_args, c = core_args()
    attack_args, a = attack_args()
    init_args, p = initialization_args()

    # set seeds
    set_seeds(core_args.seed)

    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/initialize.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')

    # Get the device
    if core_args.force_cpu:
        device = torch.device('cpu')
    else:
        device = get_default_device(core_args.gpu_id)
    print(device)

    if init_args.init_approach == 'bland':
        
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
    


    elif init_args.init_approach == 'bland2':

        '''Take average (across context) next token probability of summarisation system iteratively'''

        # load data
        data, _ = load_data(core_args)
        contexts = [s.context for s in data]

        # load summarization model
        summ_model = LlamaBase('llama-2-7b-chat-hf',device)

        bland_summ = ''
        # get each average predicted token iteratively
        for i in tqdm(range(attack_args.num_adv_tkns)):
            total_prob = None
            for j, context in enumerate(contexts):
                prompt = f'Summarise: {context} {bland_summ}'
                ids = summ_model.tokenizer(prompt, return_tensors='pt')['input_ids'][0].to(device)
                out = summ_model.forward(ids.unsqueeze(dim=0))
                import pdb; pdb.set_trace()




    
    elif init_args.init_approach == 'greedy':

        '''learn the next universal adversarial token to append greedily'''

        # load the train data
        data, _ = load_data(core_args)

        # load model
        model = load_model(model_name=core_args.model_name, device=device)

        # load the vocab
        import nltk
        nltk.download('words')
        from nltk.corpus import words
        word_list = words.words()

        # temporarily reduce word_list size - later will batch over vocab
        word_list = list(set(word_list))[:20000]

        attacker = GCGAttacker(attack_args, model)

        # score with no attack
        score_no_attack = attacker.sample_evaluate_uni_attack(data, init_args.prev_phrase, attack_type='A')

        word_2_score = {}
        for word in tqdm(word_list):
            if init_args.prev_phrase == '':
                adv_phrase = word
            else:
                adv_phrase = init_args.prev_phrase + ' ' + word
            score = attacker.sample_evaluate_uni_attack(data, adv_phrase, attack_type='A')
            word_2_score[word] = score
        
        # save
        pos = len(init_args.prev_phrase.split(' '))+1 if init_args.prev_phrase != '' else 1
        base_path = base_path_creator(core_args)
        path = next_dir(base_path, 'initialization')
        path = next_dir(path, 'greedy')
        path = next_dir(path, f'pos{pos}')

        fpath_prev = f'{path}/prev.txt'
        fpath_scores = f'{path}/scores.txt'

        with open(fpath_prev, 'w') as f:
            json.dump({'prev-adv-phrase': init_args.prev_phrase, 'score':score_no_attack}, f)
        with open(fpath_scores, 'w') as f:
            json.dump(word_2_score, f)




