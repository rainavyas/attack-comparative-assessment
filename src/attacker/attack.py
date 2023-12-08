from abc import ABC, abstractmethod
import json
import os
from tqdm import tqdm
import torch
import numpy as np
import torch.nn.functional as F
import random

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
        self._load_init_phrase()
    
    def _load_init_phrase(self):
        if self.attack_args.init_phrase == 'semicolon':
            self.init_phrase = ';' * self.num_adv_tkns
        elif self.attack_args.init_phrase == 'bland':
            self.init_phrase = "Sterling's wife receives gifts after suing; endangered gray whales may be even fewer in number."
            self.num_adv_tkns = len(self.tokenizer(self.init_phrase, add_special_tokens=False, return_tensors='pt')['input_ids'].squeeze())
        elif self.attack_args.init_phrase == 'bland2':
            self.init_phrase = "A young man named Michael was driving home from work when he saw a group of people gathered around."
            self.num_adv_tkns = len(self.tokenizer(self.init_phrase, add_special_tokens=False, return_tensors='pt')['input_ids'].squeeze())
        elif self.attack_args.init_phrase == 'greedy':
            init_phrase = "resuggest concatenation relation ending"
            self.init_phrase = ' '.join(init_phrase.split()[:self.attack_args.num_init_phrase_words]) + '.'
            self.num_adv_tkns = len(self.tokenizer(self.init_phrase, add_special_tokens=False, return_tensors='pt')['input_ids'].squeeze())
    
    def universal_attack(self, data, cache_path=None):

        # try to load from cache
        fpath = f'{cache_path}/universal.txt'
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
        
            Returns a numpy array with cell i,j representing the probability
            system i is better than system j
        '''
        print('Evaluating')

        num_systems = 16
        result = np.zeros((num_systems, num_systems))
        
        for sample in tqdm(data):
            context = sample.context
            for i in range(num_systems):
                summi = sample.responses[i]
                if attack_type == 'A':
                    summi = summi + ' ' + adv_phrase
                for j in range(num_systems):
                    summj = sample.responses[j]

                    with torch.no_grad():
                        # attacked summ in position A
                        input_ids = self.prep_input(context, summi, summj)
                        output = self.model.forward(input_ids=input_ids.unsqueeze(dim=0))
                        logits = output.logits.squeeze().cpu()
                        prob1 = F.softmax(logits, dim=0)[0].item()

                        # attacked summ in position B
                        input_ids = self.prep_input(context, summj, summi)
                        output = self.model.forward(input_ids=input_ids.unsqueeze(dim=0))
                        logits = output.logits.squeeze().cpu()
                        prob2= F.softmax(logits, dim=0)[1].item()
                    
                    prob_i_better = 0.5*(prob1+prob2)
                    result[i][j] += prob_i_better

        return result/len(data)

    def sample_evaluate_uni_attack(self, data, adv_phrase='', attack_type=None):
        '''
            List: [dict]
                Keys: 'prompt', 'prediction', 'adv_target', 'adv_prompt', 'adv_predicton'
        
            Returns the average probability attacked system i is better than system j
            randomly samples the pair of summary systems i and j for each sample
            only consider the seen summarization systems
        '''

        result = 0
        
        for sample in data:
            context = sample.context
            summi, summj  = random.sample(sample.responses[:self.attack_args.num_systems_seen], 2)
            if attack_type == 'A':
                summi = summi + ' ' + adv_phrase

            with torch.no_grad():
                # attacked summ in position A
                input_ids = self.prep_input(context, summi, summj)
                output = self.model.forward(input_ids=input_ids.unsqueeze(dim=0))
                logits = output.logits.squeeze().cpu()
                prob1 = F.softmax(logits, dim=0)[0].item()

                # attacked summ in position B
                input_ids = self.prep_input(context, summj, summi)
                output = self.model.forward(input_ids=input_ids.unsqueeze(dim=0))
                logits = output.logits.squeeze().cpu()
                prob2= F.softmax(logits, dim=0)[1].item()
                    
                prob_i_better = 0.5*(prob1+prob2)
                result += prob_i_better

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
