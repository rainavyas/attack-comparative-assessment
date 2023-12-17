from .flant5 import ComparativeFlanT5, AbsoluteFlanT5
from .llama import ComparativeLlama, AbsoluteLlama

def load_model(model_name, device='cpu', assessment='comparative'):
    if 'flant5' in model_name:
        if assessment == 'comparative':
            model = ComparativeFlanT5(model_name, device=device)
        elif assessment == 'absolute':
            model = AbsoluteFlanT5(model_name, device=device)
    elif 'llama' in model_name:
        if assessment == 'comparative':
            model = ComparativeLlama(model_name, device=device)
        elif assessment == 'absolute':
            model = AbsoluteLlama(model_name, device=device)
    else:
        raise Exception('invalid model name')
    return model