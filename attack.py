import sys
import os
import torch

from src.tools.tools import get_default_device, set_seeds
from src.tools.args import core_args, attack_args
from src.tools.saving import base_path_creator
from src.data.data_utils import load_data
from src.models.load_model import load_model
from src.attacker.gcg import GCGAttacker


if __name__ == "__main__":

    # get command line arguments
    core_args, c = core_args()
    attack_args, a = attack_args()

    set_seeds(core_args.seed)
    base_path = base_path_creator(core_args)

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
    
    # Load the data
    data = load_data(core_args.data_name)

    # Load the model, tokenizer
    model, tokenizer = load_model(core_args, device=device)

    # attack (and cache)
    attacker = GCGAttacker(attack_args, model, tokenizer, device)
    adv_data = attacker.attack_all_samples(data, cache_path=base_path)

    # evaluate
    no_attack_success_rate, attack_success_rate = attacker.evaluate_attack(adv_data)
    print(f'Success before attack:\t{no_attack_success_rate*100}%')
    print(f'Success after attack:\t{attack_success_rate*100}%')