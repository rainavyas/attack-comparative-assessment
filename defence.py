import os
import sys
import numpy as np


from sklearn.metrics import precision_recall_curve

from src.defence.perplexity import HFBase, compute_perplexities, get_best_f_score
from src.attacker.attack import BaseAbsoluteAttacker
from src.tools.tools import get_default_device, set_seeds
from src.tools.args import core_args, attack_args
from src.tools.saving import base_path_creator, attack_base_path_creator_eval
from src.data.load_data import load_data


if __name__ == "__main__":

    # get command line arguments
    core_args, c = core_args()
    attack_args, a = attack_args()

    # set seeds
    set_seeds(core_args.seed)
    base_path = base_path_creator(core_args)
    attack_base_path = attack_base_path_creator_eval(attack_args, base_path)

    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/defence.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')

    # Get the device
    if core_args.force_cpu:
        device = torch.device('cpu')
    else:
        device = get_default_device(core_args.gpu_id)
    print(device)

    # use Mistral non instruction tuned model to compute perplexity of each sentence
    model = HFBase('mistralai/Mistral-7B-v0.1', device)


    # load attacker for getting eval phrase
    attacker = BaseAbsoluteAttacker(attack_args, model)
    attack_phrase = attacker.adv_phrase

    # load data
    _, data = load_data(core_args)

    clean_texts = [r for d in data for r in d.responses]
    adv_texts = [c + ' ' + attack_phrase for c in clean_texts]

    # compute perplexity
    print("Computing perplexity for clean texts")
    cache_path = f'{base_path}/perplexity.npy'
    clean_perplexities = compute_perplexities(model, clean_texts, device, cache_path, force_run=attack_args.force_run)
   
    print("Computing perplexity for adv texts")
    cache_path = f'{attack_base_path}/perplexity.npy'
    adv_perplexities = compute_perplexities(model, adv_texts, device, cache_path, force_run=attack_args.force_run)

    # Generate PR curves
    # save_path = f'{attack_base_path}/pr_perplexity.png'

    labels = [0]*len(clean_texts) + [1]*len(adv_texts)
    perplexities = np.concatenate((clean_perplexities, adv_perplexities))
    precision, recall, _ = precision_recall_curve(labels, perplexities)
    precision = precision[:-1]
    recall = recall[:-1]
    best_precision, best_recall, best_f1 =  get_best_f_score(precision, recall)

    fpath = f'{attack_base_path}/perplexity.npz'
    np.savez(fpath, precision=np.asarray(precision), recall=np.asarray(recall))