'''
    Train adversarial phrase
'''

import random
import sys
import os
import torch
import torch.nn as nn
from tqdm import tqdm
import json

from src.tools.args import core_args, attack_args
from src.data.load_data import load_data
from src.models import load_model
from src.models.llama import LlamaBase
from src.tools.tools import get_default_device, set_seeds
from src.attacker.selector import select_train_attacker
from src.tools.saving import base_path_creator, attack_base_path_creator_train

if __name__ == "__main__":

    # get command line arguments
    core_args, c = core_args()
    attack_args, a = attack_args()

    # set seeds
    set_seeds(core_args.seed)
    base_path = base_path_creator(core_args)
    attack_base_path = attack_base_path_creator_train(attack_args, base_path)

    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/train_attack.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')

    # Get the device
    if core_args.force_cpu:
        device = torch.device('cpu')
    else:
        device = get_default_device(core_args.gpu_id)
    print(device)

    # load training data
    data, _ = load_data(core_args)


    if attack_args.attack_method == 'bland':

        from openai import OpenAI

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
    


    elif attack_args.attack_method == 'bland2':

        '''Take average (across context) next token probability of summarisation system iteratively'''

        # load data
        data, _ = load_data(core_args)
        contexts = [s.context for s in data]

        # load summarization model
        summ_model = LlamaBase('llama-2-7b-chat-hf', device)

        bland_summ = ''
        sf = nn.Softmax(dim=0)
        # get each average predicted token iteratively
        for i in tqdm(range(attack_args.num_adv_tkns)):
            total_prob = None
            for j, context in enumerate(contexts):
                # uisng Llama-2-chat prompt template
                prompt = f'[INST] Summarize the following text.\n{context} [/INST] {bland_summ}'
                ids = summ_model.tokenizer(prompt, return_tensors='pt')['input_ids'][0].to(device)
                with torch.no_grad():
                    next_tkn_logits = summ_model.forward(input_ids = ids.unsqueeze(dim=0))['logits'][0,-1,:].detach().cpu()
                    probs = sf(next_tkn_logits)

                    if j==0:
                        total_prob = probs
                    else:
                        total_prob += probs
            next_tkn_id = torch.argmax(total_prob, dim=0)
            bland_summ += f' {summ_model.tokenizer.decode(next_tkn_id)}'
            print(bland_summ)




    
    elif attack_args.attack_method == 'greedy' or attack_args.attack_method == 'greedy2':

        '''learn the next universal adversarial token to append greedily'''

        # load model
        model = load_model(model_name=core_args.model_name, device=device, assessment=core_args.assessment)

        # load the vocab
        fpath = 'experiments/words.txt'
        if os.path.isfile(fpath):
            with open(fpath, 'r') as f:
                word_list = json.load(f)
        else:
            import nltk
            nltk.download('words')
            from nltk.corpus import words
            word_list = words.words()
            word_list = list(set(word_list))[:20000]

            with open(fpath, 'w') as f:
                json.dump(word_list, f)
        
        # select vocab segment if array job
        if attack_args.array_job_id != -1:
            start = attack_args.array_job_id*attack_args.array_word_size
            end = start+attack_args.array_word_size
            word_list = word_list[start:end]

        # save scores for each word as the next word in the uni adv phrase
        attacker = select_train_attacker(attack_args, core_args, model, word_list=word_list)
        prev, word_2_score = attacker.next_word_score(data, attack_args.prev_phrase, attack_base_path, array_job_id=attack_args.array_job_id)
    


    elif attack_args.attack_method == 'gcg':

        # Load the model, tokenizer
        model = load_model(model_name=core_args.model_name, device=device, assessment=core_args.assessment)

        # universal attack (and cache)
        attacker = select_train_attacker(attack_args, core_args)
        adv_phrase = attacker.universal_attack(data, cache_path=attack_base_path)
        






