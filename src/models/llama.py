import torch
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModelForCausalLM
from types import SimpleNamespace


MODEL_URLS = {
    'llama2-7b':'meta-llama/Llama-2-7b-chat-hf',
    'mistral-7b':'mistralai/Mistral-7B-Instruct-v0.2'
}

class ComparativeLlama:
    def __init__(self, model_name, decoder_prefix='Summary', label_words=['A', 'B'], device=None):
        # load model and tokenizer
        system_url = MODEL_URLS[model_name]
        self.tokenizer = AutoTokenizer.from_pretrained(system_url)
        self.model = AutoModelForCausalLM.from_pretrained(system_url, return_dict=True)
        
        # set up prompt-based compatative classifier
        self.label_ids = self.setup_label_words(label_words)

        self.decoder_prefix = decoder_prefix
        decoder_prefix_tokens = self.tokenizer(decoder_prefix, add_special_tokens=False)
        self.decoder_prefix_ids = torch.LongTensor(decoder_prefix_tokens['input_ids'])
        self.decoder_prefix_mask = torch.LongTensor(decoder_prefix_tokens['attention_mask'])

        # set device 
        if not device:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.to(device)
      
        if device == 'cuda' and 'llama' in model_name:
            self.model = self.model.half()

    def setup_label_words(self, label_words):
        label_ids = [int(self.tokenizer(word, add_special_tokens=False).input_ids[-1]) for word in label_words]
        return label_ids

    def add_decoder_prefix(self, input_ids, attention_mask):
        bsz = input_ids.shape[0]
        input_ids = torch.cat([input_ids, self.decoder_prefix_ids.repeat(bsz, 1)], dim=-1)
        if attention_mask:
            assert torch.all(attention_mask[:, -1] == 1)
            attention_mask = torch.cat([attention_mask, self.decoder_prefix_mask.repeat(bsz, 1)], dim=-1)

        return input_ids, attention_mask

    def to(self, device):
        self.device = device
        self.model.to(self.device)
        self.decoder_prefix_ids = self.decoder_prefix_ids.to(device) 
        self.decoder_prefix_mask = self.decoder_prefix_mask.to(device)

    #== Main forward method =================================================================================#
    def forward(self, input_ids, attention_mask=None):
        if self.decoder_prefix:
            input_ids, attention_mask = self.add_decoder_prefix(input_ids, attention_mask)

        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        vocab_logits = output.logits[:,-1]
        class_logits = vocab_logits[:, tuple(self.label_ids)]
        preds = torch.argmax(class_logits, dim=-1)
        
        # self.debug_output_logits(vocab_logits) #VYAS for debug
        # print(F.softmax(vocab_logits, dim=-1)[:, tuple(self.label_ids)]) #VYAS for debug

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
    def debug_output_logits(self, logits):
        # Debug function to see what outputs would be
        indices = logits.topk(k=5).indices[0]
        print(self.label_ids)
        print(indices)
        print(self.tokenizer.decode(indices))
        print('\n\n')
        import time; time.sleep(1)

class AbsoluteLlama(ComparativeLlama):
    def __init__(self, model_name, scores=[1, 2, 3, 4 , 5], device=None):
        self.scores = torch.LongTensor([int(i) for i in scores])
        label_words = [str(i) for i in scores]

        super().__init__(model_name, label_words=label_words, device=device)
        self.scores = self.scores.to(device)

    def eval_score(self, input_ids, attention_mask=None):
        output = self.forward(input_ids=input_ids, attention_mask=attention_mask)
        probs = F.softmax(output.logits, dim=-1)
        
        score = torch.sum(probs*self.scores)
        output.score = score

        return output


class AbsoluteCoTLlama:
    def __init__(self, model_name, bsz=1, device=None):
        # load model and tokenizer
        system_url = MODEL_URLS[model_name]
        self.tokenizer = AutoTokenizer.from_pretrained(system_url)
        self.model = AutoModelForCausalLM.from_pretrained(system_url, return_dict=True)

        # set device 
        if not device:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.to(device)

    def eval_score(self, input_ids, attention_mask=None):
        with torch.no_grad():
            output = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                do_sample=False,
                max_new_tokens=50)
        
        # remove input ids (llama returns prompts with output)
        output = output[:,input_ids.shape[-1]:]
        out_text = self.tokenizer.decode(output.squeeze(dim=0))

        # extract score -- assume it is in '[[score]]'
        pos1 = out_text.find("[[")
        pos2 = out_text.find("]]")
        if pos1 == -1 or pos2 == -1:
            score = 5
        else:
            score = int(out_text[pos1+2 : pos2].strip())

        # print(out_text, score)
        return SimpleNamespace(
            text=out_text,
            score=torch.tensor(score)
        )


        
    def to(self, device):
        self.device = device
        self.model.to(self.device)
