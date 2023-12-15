import torch
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from types import SimpleNamespace
from functools import lru_cache
from typing import List
import numpy as np

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
        
        return decoder_input_ids.to(self.device)
    
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





class AbsoluteFlanT5:
    def __init__(self, system_name:str, scores=[1,2,3,4,5], device=None):

        system_url = MODEL_URLS[system_name]
        self.tokenizer = AutoTokenizer.from_pretrained(system_url)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(system_url, return_dict=True)
        if not device:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.to(device)

        # set up args for prompt template
        self.decoder_ids = False
        self.probs_setup = False

        self.set_up_prompt_classifier(scores)

    def to(self, device):
        self.device = device
        self.model.to(self.device)

    #== Output generation methods =================================================================#
    def text_response(self, input_text, top_k:int=10, max_new_tokens:int=None, **kwargs):
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)

        #print(input_text)
        #import time; time.sleep(2)
        with torch.no_grad():
            output = self.model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                top_k=top_k,
                # do_sample=do_sample,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.eos_token_id,
                no_repeat_ngram_size=3
            )

        output_tokens = output[0]
        output_text = self.tokenizer.decode(output_tokens, skip_special_tokens=True).strip()
        return SimpleNamespace(output_text=output_text)

    def set_up_prompt_classifier(self, scores=[1, 2, 3, 4, 5], decoder_prefix=''):
        self.scores = scores
        # Set up label words
        label_words = [str(i) for i in scores]
        label_ids = [self.tokenizer(word, add_special_tokens=False).input_ids for word in label_words]
        self.label_ids = [int(self.tokenizer(word, add_special_tokens=False).input_ids[-1]) for word in label_words]

        # Set up decoder prefix
        self.decoder_prefix = decoder_prefix
        self.decoder_input_ids = self._get_decoder_ids()
        # print(f"decoder prefix is {self.decoder_prefix}")
   
    def _get_decoder_ids(self, bsz=1) -> List[int]:
        if self.decoder_prefix:
            # repeat template bsz times
            decoder_input_ids = self.tokenizer(
                [self.decoder_prefix for _ in range(bsz)],
                return_tensors="pt",
            ).input_ids

            # add start token
            decoder_input_ids = self.model._shift_right(decoder_input_ids)
        else:
            # set input to start of sentence token
            decoder_input_ids = self.model.config.decoder_start_token_id * torch.ones(bsz, 1, dtype=torch.long)

        decoder_input_ids = decoder_input_ids.to(self.device)
        return decoder_input_ids
    

    def prompt_classifier_response(self, input_text):
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)

        with torch.no_grad():
            output = self.model(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                decoder_input_ids=self.decoder_input_ids
            )

        vocab_logits = output.logits[:,-1]

        class_logits = vocab_logits[0, tuple(self.label_ids)]
        raw_class_probs = F.softmax(vocab_logits, dim=-1)[0, tuple(self.label_ids)]
        pred = int(torch.argmax(class_logits[0]))+1
        probs = F.softmax(class_logits, dim=-1).cpu().numpy()
        
        #self.debug_output_logits(inputs, vocab_logits)
        
        return SimpleNamespace(
            logits=[float(i) for i in class_logits],
            probs=probs,
            pred=pred, 
            raw_probs=[float(i) for i in raw_class_probs]
        )

    def prompt_classifier_response_score(self, input_text):
        '''
            Returns the expected score
        '''
        output = self.prompt_classifier_response(
            input_text=input_text
        )
        return np.sum(output.probs*np.array(self.scores))

    def debug_output_logits(self, input_text, logits):
        # Debug function to see what outputs would be
        indices = logits.topk(k=5).indices[0]
        print(input_text)
        print('\n')
        print(self.label_ids)
        print(indices)
        print(self.tokenizer.decode(indices))
        print('\n\n')
        import time; time.sleep(1)
