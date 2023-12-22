from .flant5 import ComparativeFlanT5, AbsoluteFlanT5, AbsoluteCoTFlanT5
from .llama import ComparativeLlama, AbsoluteLlama, AbsoluteCoTLlama

def load_model(model_name, device='cpu', assessment='comparative'):
    if 'flant5' in model_name:
        if 'comparative' in assessment:
            model = ComparativeFlanT5(model_name, device=device)
        elif 'absolute' in assessment:
            if 'cot' in assessment:
                model = AbsoluteCoTFlanT5(model_name, device=device)
            else:
                model = AbsoluteFlanT5(model_name, device=device)
    elif 'llama' in model_name:
        if 'comparative' in assessment:
            model = ComparativeLlama(model_name, device=device)
        elif 'absolute' in assessment:
            if 'cot' in assessment:
                model = AbsoluteCoTLlama(model_name, device=device)
            else:
                model = AbsoluteLlama(model_name, device=device)
    else:
        raise Exception('invalid model name')
    return model