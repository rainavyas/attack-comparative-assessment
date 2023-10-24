from .flant5 import ComparativeFlanT5

def load_model(model_name, device='cpu'):
    if 'flant5' in model_name:
        model = ComparativeFlanT5(model_name, device=device)
    else:
        raise Exception('invalid model name')
    return model