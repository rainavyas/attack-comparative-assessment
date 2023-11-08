from abc import ABC, abstractmethod
import json
import os
from tqdm import tqdm
import torch
import numpy as np

from src.data.templates import load_prompt_template

class Attacker(ABC):
    '''
    Base class for adversarial attacks
    '''
    def __init__(self, attack_args, model):
        self.attack_args = attack_args
        self.model = model
        self.tokenizer = self.model.tokenizer
        self.prompt_template = load_prompt_template()

        self.num_adv_tkns = attack_args.num_adv_tkns
        self.init_phrase = ';' * self.num_adv_tkns
    
    def universal_attack(self, data, cache_path=None):
        if not os.path.isdir(f'{cache_path}/{self.attack_args.attack_method}'):
            os.mkdir(f'{cache_path}/{self.attack_args.attack_method}')
        dir_path = f'{cache_path}/{self.attack_args.attack_method}'

        # try to load from cache
        fpath = f'{dir_path}/universal.txt'
        if os.path.isfile(fpath):
            with open(fpath, 'r') as f:
                adv_phrase = json.load(f)['adv-phrase']
            self.adv_phrase = adv_phrase
            return adv_phrase

        adv_phrase = self.init_phrase
        for _ in tqdm(range(self.attack_args.outer_steps)):
            adv_phrase = self.attack_batch(data, adv_phrase)

        # save
        with open(fpath, 'w') as f:
            json.dump({'adv-phrase': adv_phrase}, f)
        self.adv_phrase = adv_phrase
          
        return adv_phrase

    def evaluate_uni_attack(self, data, adv_phrase='', attack_type=None):
        '''
            List: [dict]
                Keys: 'prompt', 'prediction', 'adv_target', 'adv_prompt', 'adv_predicton'
        
            Returns a numpy array with cell i,j representing the fraction of samples
            for which system i was better than system j
        '''
        print('Evaluating')

        num_systems = 16
        result = np.zeros((num_systems, num_systems))
        
        for sample in tqdm(data):
            context = sample.context
            for i in range(num_systems):
                summA = sample.responses[i]
                if attack_type == 'A':
                    summA = summA + ' ' + adv_phrase
                for j in range(num_systems):
                    summB = sample.responses[j]
                    if attack_type == 'B':
                        summB = summB + ' ' + adv_phrase

                    with torch.no_grad():
                        input_ids = self.prep_input(context, summA, summB)
                        output = self.model.forward(input_ids=input_ids.unsqueeze(dim=0))
                        logits = output.logits.squeeze().cpu()

                    if logits[0] > logits[1]:
                        result[i][j] += 1

        return result/len(data)


    def prep_input(self, context, summary_A, summary_B):
        input_text = self.prompt_template.format(context=context, summary_A=summary_A, summary_B=summary_B)
        tok_input = self.tokenizer(input_text, return_tensors='pt').to(self.model.device)
        input_ids = tok_input['input_ids'][0]
        return input_ids

    @abstractmethod
    def attack_batch(self, batch, adv_phrase):
        '''
            Update the adversarial phrase, optimized on the batch of data

            batch: List[dict]:
                dict: {'context': , 'responses': []*12}
        '''
        raise NotImplementedError
