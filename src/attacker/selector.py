from .attack import BaseComparativeAttacker, BaseAbsoluteAttacker, BaseAbsoluteEnsAttacker
from .greedy import GreedyComparativeAttacker, GreedyAbsoluteAttacker, GreedyAbsoluteEnsAttacker
from .gcg import GCGComparativeAttacker

def select_eval_attacker(attack_args, core_args, model):
    if core_args.assessment == 'comparative':
        return BaseComparativeAttacker(attack_args, model)
    elif core_args.assessment == 'absolute-ens':
        return BaseAbsoluteEnsAttacker(attack_args, model)
    elif core_args.assessment == 'absolute':
        return BaseAbsoluteAttacker(attack_args, model, template=1)
    elif core_args.assessment == 'absolute2':
        return BaseAbsoluteAttacker(attack_args, model, template=2)

def select_train_attacker(attack_args, core_args, model, **kwargs):
    if core_args.assessment == 'comparative':
        if attack_args.attack_method == 'greedy' or attack_args.attack_method == 'greedy2':
            return GreedyComparativeAttacker(attack_args, model, **kwargs)
        elif attack_args.attack_method == 'gcg':
            return GCGComparativeAttacker(attack_args, model)

    elif core_args.assessment == 'absolute-ens':
        if attack_args.attack_method == 'greedy' or attack_args.attack_method == 'greedy2':
            return GreedyAbsoluteEnsAttacker(attack_args, model)
    elif core_args.assessment == 'absolute':
        if attack_args.attack_method == 'greedy' or attack_args.attack_method == 'greedy2':
            return GreedyAbsoluteAttacker(attack_args, model, template=1, **kwargs)
    elif core_args.assessment == 'absolute2':
        if attack_args.attack_method == 'greedy' or attack_args.attack_method == 'greedy2':
            return GreedyAbsoluteAttacker(attack_args, model, template=2, **kwargs)

        