import numpy as np
import json

from datasets import load_dataset
from types import SimpleNamespace
from typing import List


def load_data(core_args):
    if core_args.data_name == 'summeval':
        train, test = load_summeval(train_frac=core_args.train_frac)
    elif core_args.data_name == 'topicalchat':
        train, test = load_topicalchat(train_frac=core_args.train_frac)
    return train, test
    
def load_summeval(train_frac=0.2)->List[SimpleNamespace]:
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

def load_topicalchat(train_frac=0.2) -> List[SimpleNamespace]:
        data_path = "/rds/project/rds-8YSp2LXTlkY/data/nlg_evaluation/topicalchat_usr/tc_usr_data.json"
        with open(data_path, "r") as f:
            x = f.read()
        data = json.loads(x)

        output = []
        for k, row in enumerate(data):
            responses = row['responses']
            coh = [np.mean(x['Understandable']) for x in responses]
            nat = [np.mean(x['Natural']) for x in responses]
            con = [np.mean(x['Maintains Context']) for x in responses]
            eng = [np.mean(x['Engaging']) for x in responses]
            gr = [np.mean(x['Uses Knowledge']) for x in responses]

            ex = SimpleNamespace(
                context_id=str(k),
                context=row['context'],
                responses=[x['response'] for x in responses],
                fact=row['fact'],
                scores={
                    'coherence': coh,
                    'naturalness': nat,
                    'continuity': con,
                    'engagingness': eng,
                    'groundedness': gr,
                    'overall':np.sum([coh, nat, con, eng, gr], axis=0)
                }
            )
            output.append(ex)

        train_samples = int(train_frac*len(output))
        return output[:train_samples],  output[train_samples:]
