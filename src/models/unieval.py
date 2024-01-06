from UniEval.utils import convert_to_json
from UniEval.metric.evaluator import get_evaluator

class AbsoluteUniEval:
    def __init__(self, member='all'):
        self.evaluator = get_evaluator('summarization')
        self.member = member

    def eval_score(self, src, output):
        # uni eval scores
        data = convert_to_json(output_list=[output], 
                       src_list=[src], ref_list=[''])
        if self.member == 'all':
            # return ensemble output
            scores = self.evaluator.evaluate(data, dims=['coherence', 'consistency', 'fluency'], 
                                 overall=False, print_result=False)[0]

            score = (scores['coherence']+scores['consistency']+scores['fluency'])/3
        else:
            # return member output
            score = self.evaluator.evaluate(data, dims=[self.member], 
                                 overall=False, print_result=False)[0][self.member]
        return score