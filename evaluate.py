'''
Generic functions to process saved outputs for presenting results
'''

import numpy as np
import os
import sys

from src.tools.args import core_args, attack_args, process_args
from src.tools.saving import base_path_creator
from attack import get_fpaths


if __name__ == "__main__":

    # get command line arguments
    core_args, c = core_args()
    attack_args, a = attack_args()
    process_args, p = process_args()

    print(core_args)
    print(attack_args)

    base_path = base_path_creator(core_args)