import torch
from .attack import Attacker
import random



class GCGAttacker(Attacker):
    def __init__(self, attack_args, model, tokenizer, device):
        Attacker.__init__(self, attack_args, model, tokenizer, device)
        self.special_tkns_txt = self.attack_args.adv_special_tkn * self.num_adv_tkns
    
    def attack_batch(self, batch, adv_phrase):
        adv_ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(adv_phrase))
    
        # get gradient per adv token one-hot-vector (over the batch)
        adv_grads_batch = []
        for sample in batch:
            context = sample['context']
            summaries = random.sample(sample['responses'][:self.attack_args.num_systems_seen])
            summA = summaries[0]
            summB = summaries[1]

            attackA_txt = self.prompt_template(context, f'{summA} {self.special_tkns_txt}', summB)
            attackB_txt = self.prompt_template(context, summB, f'{summB} {self.special_tkns_txt}')

            attackA_ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(attackA_txt))
            attackB_ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(attackB_txt))

            with torch.no_grad():
                adv_grads = self.token_gradients(attackA_ids, adv_ids) - self.token_gradients(attackB_ids, adv_ids)
            adv_grads_batch.append(adv_grads)

        with torch.no_grad():
            adv_grads_batch = torch.stack(adv_grads_batch, dim=0)
            adv_grads = torch.mean(adv_grads_batch) # [N x V] N:num adv tokens; V: vocab size
            top_indices = torch.topk(adv_grads, self.attack_args.topk, dim=1)
        
        # randomly sample an adv token to substitute with one of the top-k inds; repeat
        for _ in range(self.attack_args.inner_steps):
            tgt_tkn = random.randint(0, self.num_adv_tkns-1)
            substitute = random.randint(0, self.attack_args.topk - 1)
            substitute_id = top_indices[tgt_tkn][substitute]
            adv_ids[tgt_tkn] = substitute_id
        
        adv_phrase = ' '.join(self.tokenizer.convert_ids_to_tokens(adv_ids))
        return adv_phrase

        


    def token_gradients(self, input_ids, attack_ids):
        '''
        input_ids must include the self.attack_args.adv_special_tkn where attack_ids are to be placed
        Returns the tensor of gradients for the attack_ids one-hot-encoded vectors
        '''
        #TODO adian
