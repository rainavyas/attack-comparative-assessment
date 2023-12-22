import numpy as np
import scipy.stats as ss

def comparative_evals(all_comparisons, type='prob', no_attack_comparisons=None):
    if type == 'prob':
        return np.mean(all_comparisons, axis=0)
    
    elif type == 'avg_rank':
        # get rank when only system i is attacked. Avg rank across contexts and attacking each system i in turn
        ranks = []
        for i in range(len(all_comparisons[1])):
            context_ranks = []
            not_i_inds = [k for k in range(len(all_comparisons[1])) if k!=i]
            for c in range(len(all_comparisons)):
                attack_score = np.sum(all_comparisons[c,i,:])
                ref_scores = np.sum(no_attack_comparisons[np.ix_([c],not_i_inds)][:].squeeze(), axis=-1)
                scores = np.concatenate((np.array([attack_score]), ref_scores))
                context_ranks.append(len(all_comparisons[1]) - ss.rankdata(scores)[0] + 1)
            ranks.append(np.mean(np.array(context_ranks)))
        return np.mean(np.array(ranks))

def absolute_evals(all_scores, type='score', no_attack_scores=None):
    if type == 'score':
        return np.mean(all_scores, axis=0)
    
    elif type == 'avg_rank':
        # get rank when only system i is attacked. Avg rank across contexts and attacking each system i in turn
        ranks = []
        for i in range(len(all_scores[1])):
            context_ranks = []
            not_i_inds = [k for k in range(len(all_scores[1])) if k!=i]
            for c in range(len(all_scores)):
                attack_score = all_scores[c,i]
                ref_scores = no_attack_scores[np.ix_([c],not_i_inds)].squeeze()
                scores = np.concatenate((np.array([attack_score]), ref_scores))
                context_ranks.append(len(all_scores[1]) - ss.rankdata(scores)[0] + 1)
            ranks.append(np.mean(np.array(context_ranks)))
        return np.mean(np.array(ranks))
