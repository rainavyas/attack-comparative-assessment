'''
Generic functions to process saved outputs for presenting results
'''

import numpy as np
import os
import sys

from src.tools.args import core_args, attack_args, process_args
from src.tools.saving import base_path_creator, attack_base_path_creator_eval
from src.attacker.evaluations import comparative_evals, absolute_evals



if __name__ == "__main__":

    # get command line arguments
    core_args, c = core_args()
    attack_args, a = attack_args()
    process_args, p = process_args()

    print(core_args)
    print(attack_args)

    base_path = base_path_creator(core_args)
    attack_base_path = attack_base_path_creator_eval(attack_args, base_path)

    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/process.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')

    if core_args.data_name == 'summeval':
        num_systems=16
    elif core_args.data_name == 'topicalchat':
        num_systems=6

    if 'comparative' in core_args.assessment:

        # no attack
        fpath = f'{base_path}/all_comparisons.npy'
        with open(fpath, 'rb') as f:
            arr_none = np.load(f)

        # attack i
        fpath = f'{attack_base_path}/all_comparisons.npy'
        with open(fpath, 'rb') as f:
            arr_attack = np.load(f)

        if process_args.grid_latex:
            # print tables in latex
        
            def latex_print(arr):
                content = ''
                for i in range(num_systems):
                    content += f'&{i+1}'
                    for j in range(num_systems):
                        content += f'&{arr[i][j]*100:.2f}'
                    content += f'\\\\'
                    if i==7:
                        content += f'\\cmidrule{{2-18}}'

                top_row =  ''.join(['&'+str(i) for i in range(1,num_systems+1)])
                out = f''' 
                            \\begin{{tabular}}{{cc|cccccccc|cccccccc}}
                            & &\\multicolumn{{num_systems}}{{c}}{{$S_j$}} \\\\
                                & {top_row}\\\\ \\midrule
                            \\multirow{{num_systems}}{{*}}{{$S_i$}}
                            {content}
                            \\end{{tabular}}
                        '''
        
                return out

            print('No attack')
            print(latex_print(comparative_evals(arr_none)))
            print()

            print('attack i')
            print(latex_print(comparative_evals(arr_attack)))
            print()

        
        if process_args.grid_latex_summ:
            # summary table of latex grid

            def latex_print(arr_none=None, arr_attack=None):

                out = f'''
                    \\begin{{tabular}}{{lcccc}}
                    \\toprule
                    Attack & seen-seen &  seen-unseen & unseen-seen & unseen-unseen\\\\ \\midrule
                '''

                def latex_row(name, arr):
                    content = name
                    ss = np.asarray(attack_args.seen_systems)
                    not_ss = np.asarray([i for i in range(len(arr)) if i not in attack_args.seen_systems])

                    content += f'&{np.mean(arr[np.ix_(ss, ss)])*100:.2f}'
                    content += f'&{np.mean(arr[np.ix_(ss, not_ss)])*100:.2f}'
                    content += f'&{np.mean(arr[np.ix_(not_ss, ss)])*100:.2f}'
                    content += f'&{np.mean(arr[np.ix_(not_ss, not_ss)])*100:.2f}'
                    content += f'&{np.mean(arr)*100:.2f}'
                    return content + f'\\\\'

                if arr_none is not None:
                    out += latex_row('None', arr_none)
                out += latex_row(f'Attack $i$, $S_i(\\mathbf{{c}}_n)\\oplus\hat{{\\bm{{\\delta}}}}$', arr_attack)

                out +=f'\\bottomrule \\end{{tabular}}'
                return out
        
            print(latex_print(comparative_evals(arr_none), comparative_evals(arr_attack)))

        if process_args.avg_rank:
            if not attack_args.not_none:
                print("No attack avg. rank", comparative_evals(arr_none, type='avg_rank', no_attack_comparisons=arr_none))
            print("Attack i avg. rank", comparative_evals(arr_attack, type='avg_rank', no_attack_comparisons=arr_none))



    if 'absolute' in core_args.assessment:

        # no attack
        fpath = f'{base_path}/all_scores.npy'
        with open(fpath, 'rb') as f:
            arr_none = np.load(f)

        # attack i
        fpath = f'{attack_base_path}/all_scores.npy'
        with open(fpath, 'rb') as f:
            arr_attack = np.load(f)

        if process_args.grid_latex_summ:
            # latex format of all 16 avg (across test) contexts and avg for 16

            def latex_print(arr):
                content = ''
                for i in range(len(arr)):
                    content += f'&{arr[i]:.2f}'
                content += f'&{np.mean(arr):.2f}'
                print(content)

            if not attack_args.not_none:
                latex_print(absolute_evals(arr_none))
            latex_print(absolute_evals(arr_attack))
        
        if process_args.avg_rank:
            if not attack_args.not_none:
                print("No attack avg. rank", absolute_evals(arr_none, type='avg_rank', no_attack_scores=arr_none))
            print("Attack i avg. rank", absolute_evals(arr_attack, type='avg_rank', no_attack_scores=arr_none))




