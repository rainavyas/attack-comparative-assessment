'''
    Designed for gcg attack
'''

import sys
import os
import torch
import numpy as np

from src.tools.tools import get_default_device, set_seeds
from src.tools.args import core_args, attack_args
from src.tools.saving import base_path_creator, attack_base_path_creator
from src.data.load_data import load_data
from src.models import load_model
from src.attacker.gcg import GCGAttacker

def get_fpaths(core_args, attack_args, attack_base_path):
    fpaths = ['eval_no_attack', 'eval_attack']
    if core_args.eval_train:
        fpaths = [f'{c}_trn' for c in fpaths]
    if attack_args.eval_init:
        fpaths = [f'{c}_init' for c in fpaths]
    fpaths = [f'{attack_base_path}/{p}.npy' for p in fpaths]
    return fpaths

if __name__ == "__main__":

    # get command line arguments
    core_args, c = core_args()
    attack_args, a = attack_args()

    print(core_args)
    print(attack_args)
    
    set_seeds(core_args.seed)
    base_path = base_path_creator(core_args)
    attack_base_path = attack_base_path_creator(attack_args, base_path)

    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/attack.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')

    # Get the device
    if core_args.force_cpu:
        device = torch.device('cpu')
    else:
        device = get_default_device(core_args.gpu_id)
    
    print(device)
    # Load the data
    train_data, test_data = load_data(core_args)
    if core_args.eval_train:
        test_data = train_data

    # Load the model, tokenizer
    model = load_model(model_name=core_args.model_name, device=device)

    # universal attack (and cache)
    attacker = GCGAttacker(attack_args, model)
    if attack_args.eval_init:
        adv_phrase = attacker.init_phrase
    else:
        adv_phrase = attacker.universal_attack(train_data, cache_path=attack_base_path)

    # evaluate on test data - separately for seen and unseen summary generation systems
    fpaths = get_fpaths(core_args, attack_args, attack_base_path)

    # 1) No attack
    fpath = fpaths[0]
    if os.path.isfile(fpath) and not attack_args.force_run:
        with open(fpath, 'rb') as f:
            result = np.load(f)
    else:
        result = attacker.evaluate_uni_attack(test_data)
        with open(fpath, 'wb') as f:
            np.save(f, result)
    print('No attack')
    print(result)
    print()

    # 2) Attack i 
    fpath = fpaths[1]
    if os.path.isfile(fpath) and not attack_args.force_run:
        with open(fpath, 'rb') as f:
            result = np.load(f)
    else:
        result = attacker.evaluate_uni_attack(test_data, adv_phrase, attack_type='A')
        with open(fpath, 'wb') as f:
            np.save(f, result)
    print('Attack i')
    print(result)
    print()

    

