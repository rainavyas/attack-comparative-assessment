'''
Plot cached PR curves
'''
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.defence.perplexity import get_best_f_score

sns.set_style('whitegrid')

if __name__ == '__main__':


    # TopicalChat - cont - 4
    fpath = 'experiments/topicalchat/flant5-base/comparative/attack_eval/topic-greedy-absolute-cont-flant5xl/num_words-4/perplexity.npz'
    npzfile = np.load(fpath)
    precision = npzfile['precision']
    recall = npzfile['recall']
    best_precision, best_recall, best_f1 =  get_best_f_score(precision, recall)

    
    plt.plot(recall, precision, color='blue', linestyle='solid', label='CNT-4')
    print('Topic-cont-4')
    print('best precision', best_precision)
    print('best recall', best_recall)
    print('best F1', best_f1)
    print()

    
    # TopicalChat - cont - 2
    fpath = 'experiments/topicalchat/flant5-base/comparative/attack_eval/topic-greedy-absolute-cont-flant5xl/num_words-2/perplexity.npz'
    npzfile = np.load(fpath)
    precision = npzfile['precision']
    recall = npzfile['recall']
    best_precision, best_recall, best_f1 =  get_best_f_score(precision, recall)

    
    plt.plot(recall, precision, color='blue', linestyle='dotted', label='CNT-2')
    print('Topic-cont-2')
    print('best precision', best_precision)
    print('best recall', best_recall)
    print('best F1', best_f1)
    print()




    # TopicalChat - overall - 4
    fpath = 'experiments/topicalchat/flant5-base/comparative/attack_eval/topic-greedy-absolute-flant5xl/num_words-4/perplexity.npz'
    npzfile = np.load(fpath)
    precision = npzfile['precision']
    recall = npzfile['recall']
    best_precision, best_recall, best_f1 =  get_best_f_score(precision, recall)

    
    plt.plot(recall, precision, color='red', linestyle='solid', label='OVE-4')
    print('Topic-ove-4')
    print('best precision', best_precision)
    print('best recall', best_recall)
    print('best F1', best_f1)
    print()



    # TopicalChat - overall - 2
    fpath = 'experiments/topicalchat/flant5-base/comparative/attack_eval/topic-greedy-absolute-flant5xl/num_words-2/perplexity.npz'
    npzfile = np.load(fpath)
    precision = npzfile['precision']
    recall = npzfile['recall']
    best_precision, best_recall, best_f1 =  get_best_f_score(precision, recall)

    
    plt.plot(recall, precision, color='red', linestyle='dotted', label='OVE-2')
    print('Topic-ove-2')
    print('best precision', best_precision)
    print('best recall', best_recall)
    print('best F1', best_f1)
    print()


    plt.legend()
    save_path = 'experiments/topic_defence_pr.png'
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.savefig(save_path, bbox_inches='tight')


    plt.clf()


    # Summ - cons - 4
    fpath = 'experiments/summeval/flant5-base/comparative/attack_eval/greedy-absolute-cons-flant5xl/num_words-4/perplexity.npz'
    npzfile = np.load(fpath)
    precision = npzfile['precision']
    recall = npzfile['recall']
    best_precision, best_recall, best_f1 =  get_best_f_score(precision, recall)

    
    plt.plot(recall, precision, color='blue', linestyle='solid', label='CNT-4')
    print('Summ-con-4')
    print('best precision', best_precision)
    print('best recall', best_recall)
    print('best F1', best_f1)
    print()

    
    # Summ - cons - 2
    fpath = 'experiments/summeval/flant5-base/comparative/attack_eval/greedy-absolute-cons-flant5xl/num_words-2/perplexity.npz'
    npzfile = np.load(fpath)
    precision = npzfile['precision']
    recall = npzfile['recall']
    best_precision, best_recall, best_f1 =  get_best_f_score(precision, recall)

    
    plt.plot(recall, precision, color='blue', linestyle='dotted', label='CNT-2')
    print('Summ-con-2')
    print('best precision', best_precision)
    print('best recall', best_recall)
    print('best F1', best_f1)
    print()




    # Summ - overall - 4
    fpath = 'experiments/summeval/flant5-base/comparative/attack_eval/greedy-absolute-flant5xl/num_words-4/perplexity.npz'
    npzfile = np.load(fpath)
    precision = npzfile['precision']
    recall = npzfile['recall']
    best_precision, best_recall, best_f1 =  get_best_f_score(precision, recall)

    
    plt.plot(recall, precision, color='red', linestyle='solid', label='OVE-4')
    print('Summ-ove-4')
    print('best precision', best_precision)
    print('best recall', best_recall)
    print('best F1', best_f1)
    print()



    # Summ - overall - 2
    fpath = 'experiments/summeval/flant5-base/comparative/attack_eval/greedy-absolute-flant5xl/num_words-2/perplexity.npz'
    npzfile = np.load(fpath)
    precision = npzfile['precision']
    recall = npzfile['recall']
    best_precision, best_recall, best_f1 =  get_best_f_score(precision, recall)

    
    plt.plot(recall, precision, color='red', linestyle='dotted', label='OVE-2')
    print('Summ-ove-2')
    print('best precision', best_precision)
    print('best recall', best_recall)
    print('best F1', best_f1)
    print()


    plt.legend()
    save_path = 'experiments/summ_defence_pr.png'
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.savefig(save_path, bbox_inches='tight')




    