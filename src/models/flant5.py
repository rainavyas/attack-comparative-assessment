import torch
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from types import SimpleNamespace
from functools import lru_cache
from typing import List

MODEL_URLS = {
    'flant5-base':'google/flan-t5-base',
    'flant5-large':'google/flan-t5-large',
    'flant5-xl':'google/flan-t5-xl',
    'flant5-xxl':'google/flan-t5-xxl',
}

class ComparativeFlanT5:
    def __init__(self, model_name, decoder_prefix='Summary', label_words=['A', 'B'], bsz=1, device=None):
        # load model and tokenizer
        system_url = MODEL_URLS[model_name]
        self.tokenizer = AutoTokenizer.from_pretrained(system_url)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(system_url, return_dict=True)
        
        # set device 
        if not device:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.to(device)
      
        # set up prompt-based compatative classifier
        self.decoder_input_ids = self.setup_decoder_ids(decoder_prefix, bsz=bsz)
        self.label_ids = self.setup_label_words()
        
    #== Setup methods =======================================================================================#
    def setup_decoder_ids(self, decoder_prefix, bsz=1):
        # set up decoder prefix
        if decoder_prefix:
            decoder_input_ids = self.tokenizer(
                [decoder_prefix for _ in range(bsz)],
                return_tensors="pt",
            ).input_ids

            # add start token
            decoder_input_ids = self.model._shift_right(decoder_input_ids)
        else:
            # set input to start of sentence token
            decoder_input_ids = self.model.config.decoder_start_token_id * torch.ones(bsz, 1, dtype=torch.long)
        
        return decoder_input_ids
    
    def setup_label_words(self):
        label_words = [' A', ' B']
        label_ids = [int(self.tokenizer(word, add_special_tokens=False).input_ids[0]) for word in label_words]
        return label_ids
    
    def to(self, device):
        self.device = device
        self.model.to(self.device)

    #== Main forward method =================================================================================#
    def forward(self, decoder_input_ids=None, **kwargs):
        output = self.model(
            decoder_input_ids=self.decoder_input_ids,
            **kwargs
        )

        vocab_logits = output.logits[:,-1]
        #self.debug_output_logits(input_ids, vocab_logits)

        class_logits = vocab_logits[:, tuple(self.label_ids)]
        #raw_class_probs = F.softmax(vocab_logits, dim=-1)[:, tuple(self.label_ids)]
        
        preds = torch.argmax(class_logits, dim=-1)
        
        return SimpleNamespace(
            logits=class_logits,
            preds=preds
        )

    #== Model util methods ==================================================================================#
    def get_embedding_matrix(self):
        return self.model.shared.weight
    
    def get_embeddings(self, input_ids):
        return self.model.shared(input_ids)
    
    #== Debug method ========================================================================================#
    def debug_output_logits(self, input_ids, logits):
        # Debug function to see what outputs would be
        indices = logits.topk(k=5).indices[0]
        print(tokenizer.decode(input_ids[0]))
        print('\n')
        print(self.label_ids)
        print(indices)
        print(self.tokenizer.decode(indices))
        print('\n\n')
        import time; time.sleep(1)
