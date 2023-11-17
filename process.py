'''
Generic functions to process saved outputs for presenting results
'''

import numpy as np

from src.tools.args import core_args, attack_args, process_args
from src.tools.saving import base_path_creator, attack_base_path_creator
from attack import get_fpaths



if __name__ == "__main__":

    # get command line arguments
    core_args, c = core_args()
    attack_args, a = attack_args()
    process_args, p = process_args()

    print(core_args)
    print(attack_args)

    base_path = base_path_creator(core_args)
    attack_base_path = attack_base_path_creator(attack_args, base_path)

    fpaths = get_fpaths(core_args, attack_args, attack_base_path)

    if process_args.grid_latex:
        # print tables in latex
    
        def latex_print(arr):
            content = ''
            for i in range(16):
                content += f'&{i+1}'
                for j in range(16):
                    content += f'&{arr[i][j]*100:.2f}'
                content += f'\\\\'
                if i==7:
                    content += f'\\cmidrule{{2-18}}'

            out = f''' 
                        \\begin{{tabular}}{{cc|cccccccc|cccccccc}}
                        & &\\multicolumn{{16}}{{c}}{{$S_j$}} \\\\
                            & & 1 & 2 & 3 & 4 & 5 & 6 & 7 & 8 & 9 & 10 & 11 & 12 & 13 & 14 & 15 & 16\\\\ \\midrule
                        \\multirow{{16}}{{*}}{{$S_i$}}
                        {content}
                        \\end{{tabular}}
                    '''
    
            return out

        print('No attack')
        fpath = fpaths[0]
        with open(fpath, 'rb') as f:
            arr = np.load(f)
        print(latex_print(arr))
        print()

        print('attack i')
        fpath = fpaths[1]
        with open(fpath, 'rb') as f:
            arr = np.load(f)
        print(latex_print(arr))
        print()

    
    if process_args.grid_latex_summ:
        # summary table of latex grid

        def latex_print(arr_none, arr_attack):

            out = f'''
                \\begin{{tabular}}{{lcccc}}
                \\toprule
                Attack & seen-seen &  seen-unseen & unseen-seen & unseen-unseen\\\\ \\midrule
            '''

            def latex_row(name, arr):
                content = name
                content += f'&{np.mean(arr[:8,:8])*100:.2f}'
                content += f'&{np.mean(arr[:8,8:])*100:.2f}'
                content += f'&{np.mean(arr[8:,:8])*100:.2f}'
                content += f'&{np.mean(arr[8:,8:])*100:.2f}'
                return content + f'\\\\'

            out += latex_row('None', arr_none)
            out += latex_row(f'Attack $i$, $S_i(\\mathbf{{c}}_n)\\oplus\hat{{\\bm{{\\delta}}}}$', arr_attack)

            out +=f'\\bottomrule \\end{{tabular}}'
            return out
    
        fpath = fpaths[0]
        with open(fpath, 'rb') as f:
            arr_none = np.load(f)

        fpath = fpaths[1]
        with open(fpath, 'rb') as f:
            arr_attack = np.load(f)
        
        print()
        print(latex_print(arr_none, arr_attack))


