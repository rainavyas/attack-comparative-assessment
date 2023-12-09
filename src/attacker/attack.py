from abc import ABC, abstractmethod
from tqdm import tqdm
import torch
import numpy as np
import torch.nn.functional as F


from src.data.templates import load_prompt_template

class BaseAttacker(ABC):
    '''
    Base class for adversarial attacks on LLM evaluation systems
    '''
    def __init__(self, attack_args, model):
        self.attack_args = attack_args
        self.model = model
        self.tokenizer = self.model.tokenizer
        self.prompt_template = load_prompt_template()

        self.adv_phrase = self._load_phrase(self.attack_args.attack_phrase)
    
    @abstractmethod
    def get_adv_phrase(self, **kwargs):
        '''
            Return the adversarial phrase
        '''
        raise NotImplementedError
    

    def _load_phrase(self, phrase_name):
        if phrase_name == 'semicolon':
            return ';' * self.attack_args.num_adv_tkns
        elif phrase_name == 'bland':
            return "Sterling's wife receives gifts after suing; endangered gray whales may be even fewer in number."
        elif phrase_name == 'bland2':
            return "A young man named Michael was driving home from work when he saw a group of people gathered around."
        elif phrase_name == 'greedy-comparative-flant5base':
            phrase = "resuggest concatenation relation ending relationally"
            return ' '.join(phrase.split()[:self.attack_args.num_greedy_phrase_words]) + '.'
        elif phrase_name == 'greedy-comparative-flant5xl':
            phrase = "uncontradictory"
            return ' '.join(phrase.split()[:self.attack_args.num_greedy_phrase_words]) + '.'
        else:
            print('No specific phrase loaded')
            return ''





class BaseComparativeAttacker(BaseAttacker):
    '''
    Base class for adversarial attacks on comparative assessment system
    '''
    def __init__(self, attack_args, model):
        BaseAttacker.__init__(self, attack_args, model)

    def get_adv_phrase(self, **kwargs):
        return self.adv_phrase


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


    def prep_input(self, context, summary_A, summary_B):
        input_text = self.prompt_template.format(context=context, summary_A=summary_A, summary_B=summary_B)
        tok_input = self.tokenizer(input_text, return_tensors='pt').to(self.model.device)
        input_ids = tok_input['input_ids'][0]
        return input_ids
