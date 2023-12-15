from .flant5 import ComparativeFlanT5, AbsoluteFlanT5

def load_model(model_name, device='cpu', assessment='comparative'):
    if 'flant5' in model_name:
        if assessment == 'comparative':
            model = ComparativeFlanT5(model_name, device=device)
        elif assessment == 'absolute':
            model = AbsoluteFlanT5(model_name, device=device)
    else:
        raise Exception('invalid model name')
    return model