import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import torch.nn as nn
import random
from random import randint
import json
import os

from .attack import BaseComparativeAttacker, BaseAbsoluteAttacker, BaseAbsoluteEnsAttacker
from src.tools.saving import next_dir

class BaseGreedyAttacker:
    def next_word_score(self, data, curr_adv_phrase, cache_path, array_job_id=-1):
        '''
            curr_adv_phrase: current universal adversarial phrase
            Returns the assessment score for each word in word list as next uni adv word
        '''

        # check for cache
        pos = len(curr_adv_phrase.split(' '))+1 if curr_adv_phrase != '' else 1
        path = next_dir(cache_path, f'pos{pos}')
        if array_job_id != -1:
            path = next_dir(path, f'array_job{array_job_id}')

        fpath_prev = f'{path}/prev.txt'
        fpath_scores = f'{path}/scores.txt'
        if os.path.isfile(fpath_prev):
            with open(fpath_prev, 'r') as f:
                prev = json.load(f)
            with open(fpath_scores, 'r') as f:
                word_2_score = json.load(f)

            return prev, word_2_score

        # set the summary ids for evaluation for next greedy word
        # summary_ids = random.sample(range(self.attack_args.num_systems_seen),2)
        summary_ids = self.attack_args.seen_systems

        score_no_attack = self.sample_evaluate_uni_attack_seen(data, curr_adv_phrase, summary_ids=summary_ids)
        word_2_score = {}
        for word in tqdm(self.word_list):
            if curr_adv_phrase == '':
                adv_phrase = word + '.'
            else:
                adv_phrase = curr_adv_phrase + ' ' + word + '.'
            score = self.sample_evaluate_uni_attack_seen(data, adv_phrase, summary_ids=summary_ids)
            word_2_score[word] = score
        
        # cache
        with open(fpath_prev, 'w') as f:
            prev = {'prev-adv-phrase': curr_adv_phrase, 'score':score_no_attack}
            json.dump(prev, f)
        with open(fpath_scores, 'w') as f:
            json.dump(word_2_score, f)
        
        return prev, word_2_score
    
    @staticmethod
    def next_best_word(base_path, pos=1):
        '''
            base_path: directory with scores.txt and prev.txt (or array_job files)
            Give the next best word from output saved files
        '''

        def best_from_dict(word_2_score, pos=1):
            prev = [None, 0]
            best = [None, 0]

            for k,v in word_2_score.items():
                if v>best[1]:
                    prev[0] = best[0]
                    prev[1] = best[1]
                    best[0]=k
                    best[1]=v
                elif v>prev[1]:
                    prev[0]=k
                    prev[1]=v
            if pos==1:
                return best[0], best[1]
            elif pos==2:
                return prev[0], prev[1]
            else:
                print("Not supported pos")

        if os.path.isfile(f'{base_path}/scores.txt'):
            with open(f'{base_path}/scores.txt', 'r') as f:
                word_2_score = json.load(f)
            return best_from_dict(word_2_score, pos=pos)
        
        elif os.path.isdir(f'{base_path}/array_job1'):
            combined = {}
            for i in range(200):
                try:
                    with open(f'{base_path}/array_job{i}/scores.txt', 'r') as f:
                        word_2_score = json.load(f)
                except:
                    continue
                combined = {**combined, **word_2_score}
            
            return best_from_dict(combined, pos=pos)

        else:
            raise ValueError("No cached scores") 

class GreedyComparativeAttacker(BaseComparativeAttacker, BaseGreedyAttacker):
    def __init__(self, attack_args, model, word_list=None):
        BaseComparativeAttacker.__init__(self, attack_args, model)
        self.word_list = word_list
    
    def sample_evaluate_uni_attack_seen(self, data, adv_phrase='', summary_ids=None):
        '''
            List: [dict]
                Keys: 'prompt', 'prediction', 'adv_target', 'adv_prompt', 'adv_predicton'
        
            Returns the average probability attacked system i is better than system j
            Randomly samples the pair of summary systems i and j for each sample (context)
             -> don't sample if summary_ids given
            Only consider the seen summarization systems
        '''

        result = 0
        for sample in data:
            context = sample.context
            # if summary_ids == None:
            #     summi, summj  = random.sample(sample.responses[:self.attack_args.num_systems_seen], 2)
            # else:
            summi = sample.responses[summary_ids[0]]
            summj = sample.responses[summary_ids[1]]
            if adv_phrase != '':
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


class GreedyAbsoluteAttacker(BaseAbsoluteAttacker, BaseGreedyAttacker):
    def __init__(self, attack_args, model, word_list=None, template=1):
        BaseAbsoluteAttacker.__init__(self, attack_args, model, template=template)
        self.word_list = word_list
    
    def sample_evaluate_uni_attack_seen(self, data, adv_phrase='', summary_ids=None):
        '''
            Returns the average (across contexts) score of a summary
            Randomly samples summary system i for each sample (context) if no summary_ids
            Only consider the seen summarization systems
        '''
        result = 0
        for sample in data:
            context = sample.context
            # if summary_ids == None:
            #     summ  = random.sample(sample.responses[:self.attack_args.num_systems_seen], 1)
            # else:
            summ = sample.responses[summary_ids[0]]
            if adv_phrase != '':
                summ = summ + ' ' + adv_phrase
                
            input_ids = self.prep_input(context, summ)
            with torch.no_grad():
                output = self.model.g_eval_score(input_ids.unsqueeze(dim=0))
                score = output.score
            result += score

        return result/len(data)




class GreedyAbsoluteEnsAttacker(BaseAbsoluteEnsAttacker, BaseGreedyAttacker):
    def __init__(self, attack_args, model, word_list=None):
        BaseAbsoluteEnsAttacker.__init__(self, attack_args, model)
        self.word_list = word_list
    
    def sample_evaluate_uni_attack_seen(self, data, adv_phrase='', summary_ids=None):
        '''
            Returns the average (across contexts) score of a summary
            Randomly samples summary system i for each sample (context) if no summary_ids
            Only consider the seen summarization systems
        '''
        result = 0
        for sample in data:
            context = sample.context
            # if summary_ids == None:
            #     summ  = random.sample(sample.responses[:self.attack_args.num_systems_seen], 1)
            # else:
            summ = sample.responses[summary_ids[0]]
            if adv_phrase != '':
                summ = summ + ' ' + adv_phrase
                
            # prompt template 1
            input_ids = self.prep_input(context, summ, self.prompt_template1)
            with torch.no_grad():
                output = self.model.g_eval_score(input_ids.unsqueeze(dim=0))
                score1 = output.score

            # prompt template 2
            input_ids = self.prep_input(context, summ, self.prompt_template2)
            with torch.no_grad():
                output = self.model.g_eval_score(input_ids.unsqueeze(dim=0))
                score2 = output.score
            
            score = 0.5*(score1+score2)
            result += score

        return result/len(data)


