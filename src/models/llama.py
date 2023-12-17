import torch
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModelForCausalLM
from types import SimpleNamespace


MODEL_URLS = {
    'llama-2-7b-chat-hf':'meta-llama/Llama-2-7b-chat-hf',
}

class ComparativeLlama:
    def __init__(self, model_name, label_words=['A', 'B'], device=None):
        # load model and tokenizer
        system_url = MODEL_URLS[model_name]
        self.tokenizer = AutoTokenizer.from_pretrained(system_url)
        self.model = AutoModelForCausalLM.from_pretrained(system_url, return_dict=True)
        
        # set device 
        if not device:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
        self.to(device)
      
        if device == 'cuda' and 'llama' in model_name:
            self.model = self.model.half()

        # set up prompt-based compatative classifier
        self.label_ids = self.setup_label_words()
        
    def setup_label_words(self):
        label_words = ['A', 'B']
        label_ids = [int(self.tokenizer(word, add_special_tokens=False).input_ids[0]) for word in label_words]
        return label_ids
    
    def to(self, device):
        self.device = device
        self.model.to(self.device)

    #== Main forward method =================================================================================#
    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        vocab_logits = output.logits[:,-1]
        class_logits = vocab_logits[:, tuple(self.label_ids)]
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
        print(self.tokenizer.decode(input_ids[0]))
        print('\n')
        print(self.label_ids)
        print(indices)
        print(self.tokenizer.decode(indices))
        print('\n\n')
        import time; time.sleep(1)

class GevalLlama(ComparativeLlama):
    def __init__(self, system_name:str, scores=[1, 2, 3, 4 , 5], device=None):
        self.scores = torch.LongTesnor([int(i) for i in scores])
        label_words = [str(i) for i in scores]

        super().__init__(system_name, label_words, device)
        self.scores.to(device)

    def g_eval_score(self, input_ids, attention_mask):
        output = self.forward(input_ids=input_ids, attention_mask=attention_mask)

        probs = F.softmax(output.class_logits, dim=-1)
        score = torch.sum(probs*self.scores)
        
        output

        return output