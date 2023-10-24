import argparse

def core_args():
    commandLineParser = argparse.ArgumentParser(allow_abbrev=False)
    commandLineParser.add_argument('--model_name', type=str, default='T5', help='comparative assessment system')
    commandLineParser.add_argument('--gpu_id', type=int, default=0, help='select specific gpu')
    commandLineParser.add_argument('--data_name', type=str, default='summeval', help='dataset for exps')
    commandLineParser.add_argument('--train_frac', type=float, default=0.1, help='fraction of samples for learning attack')
    commandLineParser.add_argument('--seed', type=int, default=1, help='select seed')
    commandLineParser.add_argument('--force_cpu', action='store_true', help='force cpu use')
    return commandLineParser.parse_known_args()

def attack_args():
    commandLineParser = argparse.ArgumentParser(allow_abbrev=False)
    commandLineParser.add_argument('--attack_method', type=str, default='gcg-uni', choices=['gcg-uni'], help='Adversarial attack approach')
    commandLineParser.add_argument('--topk', type=int, default=256, help='topk candidates for gcg')
    commandLineParser.add_argument('--batch_size', type=int, default=8, help='batchsize for gcg alg')
    commandLineParser.add_argument('--init_phrase', type=str, default="! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !", help='initialisation string for gcg')
    commandLineParser.add_argument('--inner_steps', type=int, default=10, help='inner iter steps per batch for uni-gcg alg')
    commandLineParser.add_argument('--outer_steps', type=int, default=10, help='outer iter steps for uni-gcg alg')
    commandLineParser.add_argument('--adv_special_tkn', type=str, default="<adv-tkn>", help='initialisation string for gcg')
    commandLineParser.add_argument('--num_systems_seen', type=int, default=8, help='number of summarization systems adversary has access to')
    return commandLineParser.parse_known_args()

