from .attack import BaseComparativeAttacker
from .greedy import GreedyComparativeAttacker
from .gcg import GCGComparativeAttacker

def select_eval_attacker(attack_args, core_args, model):
    if core_args.assessment == 'comparative':
        return BaseComparativeAttacker(attack_args, model)

def select_train_attacker(attack_args, core_args, model, **kwargs):
    if core_args.assessment == 'comparative':
        if attack_args.attack_method == 'greedy':
            return GreedyComparativeAttacker(attack_args, model, **kwargs)
        elif attack_args.attack_method == 'gcg':
            GCGComparativeAttacker(attack_args, model)
        