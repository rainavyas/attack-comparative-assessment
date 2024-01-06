from .attack import BaseComparativeAttacker, BaseAbsoluteAttacker
from .greedy import GreedyComparativeAttacker, GreedyAbsoluteAttacker
from .gcg import GCGComparativeAttacker

def select_eval_attacker(attack_args, core_args, model):
    if core_args.assessment == 'comparative':
        return BaseComparativeAttacker(attack_args, model)
    elif 'comparative-asym' in core_args.assessment:
        if core_args.assessment == 'comparative-asym':
            return BaseComparativeAttacker(attack_args, model, symmetric='asymA')
        elif core_args.assessment == 'comparative-asymB':
            return BaseComparativeAttacker(attack_args, model, symmetric='asymB')
    # elif core_args.assessment == 'absolute-ens':
    #     return BaseAbsoluteEnsAttacker(attack_args, model)
    elif core_args.assessment == 'absolute':
        if 'unieval' in core_args.model_name:
            return BaseAbsoluteAttacker(attack_args, model, type_ass='unieval')
        else:
            return BaseAbsoluteAttacker(attack_args, model, template=1)
    elif core_args.assessment == 'absolute2':
        return BaseAbsoluteAttacker(attack_args, model, template=2)
    elif core_args.assessment == 'absolute3':
        return BaseAbsoluteAttacker(attack_args, model, template=3)
    elif core_args.assessment == 'absolute-cot':
        return BaseAbsoluteAttacker(attack_args, model, template='cot')

def select_train_attacker(attack_args, core_args, model, **kwargs):
    if core_args.assessment == 'comparative':
        if attack_args.attack_method == 'greedy' or attack_args.attack_method == 'greedy2':
            return GreedyComparativeAttacker(attack_args, model, **kwargs)
        elif attack_args.attack_method == 'gcg':
            return GCGComparativeAttacker(attack_args, model)
    if 'comparative-asym' in core_args.assessment:
        if core_args.assessment == 'comparative-asym':
            symmetric = 'asymA'
        elif core_args.assessment == 'comparative-asymB':
            symmetric = 'asymB'
        if attack_args.attack_method == 'greedy' or attack_args.attack_method == 'greedy2':
            return GreedyComparativeAttacker(attack_args, model, symmetric=symmetric, **kwargs)

    # elif core_args.assessment == 'absolute-ens':
    #     if attack_args.attack_method == 'greedy' or attack_args.attack_method == 'greedy2':
    #         return GreedyAbsoluteEnsAttacker(attack_args, model)
    elif core_args.assessment == 'absolute':
        if attack_args.attack_method == 'greedy' or attack_args.attack_method == 'greedy2':
            if 'unieval' in core_args.model_name:
                return GreedyAbsoluteAttacker(attack_args, model, type_ass='unieval', **kwargs)
            else:
                return GreedyAbsoluteAttacker(attack_args, model, template=1, **kwargs)
    elif core_args.assessment == 'absolute2':
        if attack_args.attack_method == 'greedy' or attack_args.attack_method == 'greedy2':
            return GreedyAbsoluteAttacker(attack_args, model, template=2, **kwargs)

        