from abc import ABC, abstractmethod
import json
import os
from tqdm import tqdm

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
        self.num_adv_tkns = len(self.attack_args.init_phrase.split(' '))
    
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

        adv_phrase = self.attack_args.init_phrase
        for _ in tqdm(range(self.attack_args.outer_steps)):
            adv_phrase = self.attack_batch(data, adv_phrase)

        # save
        with open(fpath, 'w') as f:
            json.dump({'adv-phrase': adv_phrase}, f)
        self.adv_phrase = adv_phrase
          
        return adv_phrase

    @staticmethod
    def evaluate_uni_attack(data, adv_phrase):
        '''
            List: [dict]
                Keys: 'prompt', 'prediction', 'adv_target', 'adv_prompt', 'adv_predicton'
        '''
       # TODO

    @abstractmethod
    def attack_batch(self, batch, adv_phrase):
        '''
            Update the adversarial phrase, optimized on the batch of data

            batch: List[dict]:
                dict: {'context': , 'responses': []*12}
        '''
        raise NotImplementedError
