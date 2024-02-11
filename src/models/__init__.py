from .flant5 import ComparativeFlanT5, AbsoluteFlanT5, AbsoluteCoTFlanT5
from .llama import ComparativeLlama, AbsoluteLlama, AbsoluteCoTLlama
from .unieval import AbsoluteUniEval
from .openai import AbsoluteOpenAIModel

def load_model(args, device='cpu', assessment='comparative'):
    model_name = args.model_name

    # for comparative assessment need a decoder prefix
    if 'llama' in args.model_name or 'mistral' in args.model_name:
        if args.data_name == 'summeval':
            decoder_prefix = '\n\nAnswer: Summary'
        elif args.data_name == 'topicalchat':
            decoder_prefix = '\n\nAnswer: Response'

    else:
        if args.data_name == 'summeval':
            decoder_prefix = 'Summary'
        elif args.data_name == 'topicalchat':
            decoder_prefix = 'Response'
        

    if 'flant5' in args.model_name:
        if 'comparative' in assessment:
            model = ComparativeFlanT5(model_name, device=device, decoder_prefix=decoder_prefix)
        elif 'absolute' in assessment:
            if 'cot' in assessment:
                model = AbsoluteCoTFlanT5(model_name, device=device)
            else:
                model = AbsoluteFlanT5(model_name, device=device)

    elif 'llama' in model_name:
        if 'comparative' in assessment:
            model = ComparativeLlama(model_name, device=device, decoder_prefix=decoder_prefix)
        elif 'absolute' in assessment:
            if 'cot' in assessment:
                model = AbsoluteCoTLlama(model_name, device=device)
            else:
                model = AbsoluteLlama(model_name, device=device)

    elif 'unieval' in model_name:
        if model_name == 'unieval':
            model = AbsoluteUniEval(member='all')
        else:
            # single member of ensemble evaluated
            member = model_name.split('-')[-1]
            if member not in ['coherence', 'consistency', 'fluency']:
                raise ValueError('Invalid member type for UniEval')
            model = AbsoluteUniEval(member=member)

    elif 'gpt' in model_name:
        if 'comparative' in assessment:
            pass
        elif 'absolute' in assessment:
            model = AbsoluteOpenAIModel(model_name)

    else:
        raise Exception('invalid model name')
    return model