import numpy as np

from datasets import load_dataset
from types import SimpleNamespace
from typing import List


def load_data(core_args):
    if core_args.data_name == 'summeval':
        train, test = load_summeval(train_frac=core_args.train_frac)
    
    return train, test
    
def load_summeval(train_frac=0.1)->List[SimpleNamespace]:
    output = []
    summ_eval = load_dataset('mteb/summeval')['test']
    for k, row in enumerate(summ_eval):
        ex = SimpleNamespace(
            context_id=str(k),
            context=row['text'],
            responses=row['machine_summaries'],
            reference=row['human_summaries'][0],
            scores={
                'coherence':row['coherence'],
                'fluency':row['fluency'],
                'consistency':row['consistency'],
                'relevance':row['relevance'],
                # 'overall':np.sum([row['coherence'], row['fluency'], row['consistency'],row['relevance']], axis=0)
                'overall':np.sum([row['coherence'], row['fluency'], row['consistency']], axis=0)
            }
        )
        output.append(ex)
    train_samples = int(train_frac*len(output))
    return output[:train_samples],  output[train_samples:]
