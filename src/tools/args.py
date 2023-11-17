import argparse

def core_args():
    commandLineParser = argparse.ArgumentParser(allow_abbrev=False)
    commandLineParser.add_argument('--model_name', type=str, default='flant5-base', help='comparative assessment system')
    commandLineParser.add_argument('--gpu_id', type=int, default=0, help='select specific gpu')
    commandLineParser.add_argument('--data_name', type=str, default='summeval', help='dataset for exps')
    commandLineParser.add_argument('--train_frac', type=float, default=0.2, help='fraction of samples for learning attack')
    commandLineParser.add_argument('--seed', type=int, default=1, help='select seed')
    commandLineParser.add_argument('--force_cpu', action='store_true', help='force cpu use')
    commandLineParser.add_argument('--eval_train', action='store_true', help='Evaluate attack on the train split')
    return commandLineParser.parse_known_args()

def attack_args():
    commandLineParser = argparse.ArgumentParser(allow_abbrev=False)
    commandLineParser.add_argument('--attack_method', type=str, default='gcg-uni', choices=['gcg-uni'], help='Adversarial attack approach')
    commandLineParser.add_argument('--topk', type=int, default=256, help='topk candidates for gcg')
    commandLineParser.add_argument('--num_adv_tkns', type=int, default=20, help='number of concatenated tokens in uni adv phras')
    commandLineParser.add_argument('--inner_steps', type=int, default=5, help='inner iter steps per batch for uni-gcg alg')
    commandLineParser.add_argument('--outer_steps', type=int, default=100, help='outer iter steps for uni-gcg alg')
    commandLineParser.add_argument('--num_systems_seen', type=int, default=8, help='number of summarization systems adversary has access to')
    commandLineParser.add_argument('--init_phrase', default='semicolon', type=str, choices=['semicolon', 'bland', 'greedy'], help='select optimizer initialization phrase for gcg')
    commandLineParser.add_argument('--eval_init', action='store_true', help='Evaluate attack with init phrase')
    return commandLineParser.parse_known_args()

def process_args():
    commandLineParser = argparse.ArgumentParser(allow_abbrev=False)
    commandLineParser.add_argument('--grid_latex', action='store_true', help='print latex table format for 16 by 16 tables')
    commandLineParser.add_argument('--grid_latex_summ', action='store_true', help='print summarized latex table format for 16 by 16 tables')
    return commandLineParser.parse_known_args()

def initialization_args():
    commandLineParser = argparse.ArgumentParser(allow_abbrev=False)
    commandLineParser.add_argument('--bland', action='store_true', help='learn bland phrase')
    return commandLineParser.parse_known_args()


