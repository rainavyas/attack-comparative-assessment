from datasets import load_dataset
from types import SimpleNamespace
from typing import List


def load_data(data_name):
    if data_name == 'summeval':
        train, test = load_summeval()
    
    return train, test
    
def load_summeval()->List[SimpleNamespace]:
    output = []
    summ_eval = load_dataset('mteb/summeval')['test']
    for k, row in enumerate(summ_eval):
        ex = SimpleNamespace(
            context_id=str(k),
            context=row['text'],
            responses=row['machine_summaries'],
            reference=row['human_summaries'][0],
            scores={
                'coherency':row['coherence'],
                'fluency':row['fluency'],
                'consistency':row['consistency'],
                'relevance':row['relevance']
            }
        )
        output.append(ex)
    return output[:10],  output[90:]
