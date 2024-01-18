import argparse

def core_args():
    commandLineParser = argparse.ArgumentParser(allow_abbrev=False)
    commandLineParser.add_argument('--model_name', type=str, default='flant5-base', help='assessment system')
    commandLineParser.add_argument('--assessment', type=str, choices=['comparative', 'absolute', 'absolute-ens', 'absolute2', 'absolute3', 'absolute-cot', 'comparative-asym', 'comparative-asymB', 'comparative-coherence', 'absolute-coherence', 'comparative-naturalness', 'absolute-naturalness'], default='comparative', help='assessment system')
    commandLineParser.add_argument('--gpu_id', type=int, default=0, help='select specific gpu')
    commandLineParser.add_argument('--data_name', type=str, default='summeval', help='dataset for exps')
    commandLineParser.add_argument('--train_frac', type=float, default=0.2, help='fraction of samples for learning attack')
    commandLineParser.add_argument('--seed', type=int, default=1, help='select seed')
    commandLineParser.add_argument('--force_cpu', action='store_true', help='force cpu use')
    commandLineParser.add_argument('--eval_train', action='store_true', help='Evaluate attack on the train split')
    return commandLineParser.parse_known_args()

def attack_args():
    commandLineParser = argparse.ArgumentParser(allow_abbrev=False)
    # train attack args
    commandLineParser.add_argument('--attack_method', type=str, default='gcg', choices=['bland', 'bland2', 'gcg', 'greedy', 'greedy2'], help='Adversarial attack approach for training')
    commandLineParser.add_argument('--topk', type=int, default=256, help='topk candidates for gcg')
    commandLineParser.add_argument('--num_adv_tkns', type=int, default=20, help='number of concatenated tokens in uni adv phrase')
    commandLineParser.add_argument('--inner_steps', type=int, default=5, help='inner iter steps per batch for uni-gcg alg')
    commandLineParser.add_argument('--outer_steps', type=int, default=100, help='outer iter steps for uni-gcg alg')
    # commandLineParser.add_argument('--num_systems_seen', type=int, default=8, help='number of summarization systems adversary has access to')
    commandLineParser.add_argument('--seen_systems', type=int, default=[2,4], nargs='+', help='summarization systems attacker has access to during training')
    commandLineParser.add_argument('--init_phrase', default='greedy-comparative-flant5base', type=str, help='select initialization phrase for gcg')
    commandLineParser.add_argument('--prev_phrase', default='', type=str, help='previously learnt adv phrase for greedy approach')
    commandLineParser.add_argument('--array_job_id', type=int, default=-1, help='-1 means not to run as an array job')
    commandLineParser.add_argument('--array_word_size', type=int, default=400, help='number of words to test for each array job in greedy attack')

    # eval attack args
    commandLineParser.add_argument('--attack_phrase', type=str, default='greedy-comparative-flant5base', help='Specifc adversarial attack phrase to evaluate')
    commandLineParser.add_argument('--num_greedy_phrase_words', type=int, default=-1, help='for greedy phrase select only first k words')
    commandLineParser.add_argument('--force_run', action='store_true', help='Do not load from cache')
    commandLineParser.add_argument('--not_none', action='store_true', help='Do not evaluate the none attack')
    return commandLineParser.parse_known_args()

def process_args():
    commandLineParser = argparse.ArgumentParser(allow_abbrev=False)
    commandLineParser.add_argument('--grid_latex', action='store_true', help='print latex table format for 16 by 16 tables')
    commandLineParser.add_argument('--grid_latex_summ', action='store_true', help='print summarized latex table format')
    commandLineParser.add_argument('--avg_rank', action='store_true', help='Give the average rank of of the attacked summaries out of 16')
    return commandLineParser.parse_known_args()

def initialization_args():
    commandLineParser = argparse.ArgumentParser(allow_abbrev=False)

    return commandLineParser.parse_known_args()


