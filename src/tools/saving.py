import os

def base_path_creator(core_args, create=True):
    path = '.'
    path = next_dir(path, 'experiments', create=create)
    path = next_dir(path, core_args.data_name, create=create)
    path = next_dir(path, core_args.model_name, create=create)
    return path

def attack_base_path_creator(attack_args, path='.', create=True):
    path = next_dir(path, attack_args.attack_method, create=create)
    path = next_dir(path, f'init-{attack_args.init_phrase}', create=create)
    path = next_dir(path, f'topk-{attack_args.topk}', create=create)
    path = next_dir(path, f'num_adv_tkns-{attack_args.num_adv_tkns}', create=create)
    path = next_dir(path, f'inner_steps-{attack_args.inner_steps}', create=create)
    path = next_dir(path, f'outer_steps-{attack_args.outer_steps}', create=create)
    return path

def next_dir(path, dir_name, create=True):
    if not os.path.isdir(f'{path}/{dir_name}'):
        if create:
            os.mkdir(f'{path}/{dir_name}')
        else:
            raise ValueError ("provided args do not give a valid model path")
    path += f'/{dir_name}'
    return path