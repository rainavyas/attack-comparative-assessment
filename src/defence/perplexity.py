'''
    Run this script from the directory outside src
    Plots a PR curve for perplexity as a defence
'''

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn as nn
from tqdm import tqdm
import os
import numpy as np


class HFBase():
    '''
        Load HF model
    '''
    def __init__(self, model_url, device):
        self.tokenizer = AutoTokenizer.from_pretrained(model_url)
        self.model = AutoModelForCausalLM.from_pretrained(model_url)
        self.model.to(device)
        self.device = device


def perplexity(model, sentence, device):
    '''
        Compute perplexity of a sentence
        Perplexity is the reciprocal of the normalized probability of the sentence
        (1/N(log(p)))^-1
    '''

    sf = nn.Softmax(dim=0)
    ids = model.tokenizer(sentence, return_tensors='pt')['input_ids'][0].to(device)
    logsum = 0
    for i in range(1,len(ids)):
        next_tkn_logits = model.model.forward(input_ids = ids[:i].unsqueeze(dim=0))['logits'][0,-1,:].detach().cpu()
        probs = sf(next_tkn_logits)
        curr_id = ids[i].detach().cpu().item()
        prob_token = probs[curr_id]
        logsum += torch.log(prob_token)
    perplexity = (logsum/len(ids))**(-1)
    return perplexity

def compute_perplexities(model, sentences, device, cache_path, force_run=False):
    # check for cache
    if os.path.isfile(cache_path) and not force_run:
        with open(cache_path, 'rb') as f:
            perplexities = np.load(f)
    
    else:
        perplexities = []
        for text in tqdm(sentences):
            perplexities.append(perplexity(model, text, device))

        perplexities = np.array(perplexities)
        with open(cache_path, 'wb') as f:
            np.save(f, perplexities)

    return perplexities

def get_best_f_score(precisions, recalls, beta=1.0):
    f_scores = (1+beta**2)*((precisions*recalls)/((precisions*(beta**2))+recalls))
    f_scores = np.nan_to_num(f_scores)
    ind = np.argmax(f_scores)
    return precisions[ind], recalls[ind], f_scores[ind]






