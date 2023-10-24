import sys
import os
import torch

from src.tools.tools import get_default_device, set_seeds
from src.tools.args import core_args, attack_args
from src.tools.saving import base_path_creator
from src.data.load_data import load_data
from src.models import load_model
from src.attacker.gcg import GCGAttacker


if __name__ == "__main__":

    # get command line arguments
    core_args, c = core_args()
    attack_args, a = attack_args()

    print(core_args)
    print(attack_args)
    
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
    
    print(device)
    # Load the data
    train_data, test_data = load_data(core_args)

    # Load the model, tokenizer
    model = load_model(model_name=core_args.model_name, device=device)

    # universal attack (and cache)
    attacker = GCGAttacker(attack_args, model)
    adv_data = attacker.universal_attack(train_data, cache_path=base_path)

    # evaluate on test data - separately for seen and unseen summary generation systems
