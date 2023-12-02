import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


MODEL_URLS = {
    'llama-2-7b-chat-hf':'meta-llama/Llama-2-7b-chat-hf',
}

class LlamaBase(torch.nn.Module):
    def __init__(self, model_name, device=None):
        # load model and tokenizer
        system_url = MODEL_URLS[model_name]
        self.tokenizer = AutoTokenizer.from_pretrained(system_url)
        self.model = AutoModelForCausalLM.from_pretrained(system_url, return_dict=True)
        
        # set device 
        if not device:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.to(device)
