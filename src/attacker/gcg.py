import torch
import torch.nn.functional as F

import numpy as np
import torch.nn as nn
import random

from .attack import Attacker

class GCGAttacker(Attacker):
    def __init__(self, attack_args, model):
        Attacker.__init__(self, attack_args, model)
        
        # tokenzier stuff
        self.tokenizer.add_tokens([f"<attack_tok>"])
        self.adv_special_tkn_id = len(self.tokenizer) - 1
        
        self.special_tkns_txt = ''.join(["<attack_tok>" for _ in range(self.num_adv_tkns)])
        
    def attack_batch(self, batch, adv_phrase):
        '''
            Update universal adversarial phrase on batch of samples
        '''
        adv_ids = self.tokenizer(adv_phrase, add_special_tokens=False, return_tensors='pt')['input_ids'].squeeze().to(self.model.device)
        if len(adv_ids) == self.num_adv_tkns + 1:
            adv_ids = adv_ids[1:]

        # get gradient per adv token one-hot-vector (over the batch)
        adv_grads_batch = []
        logging = []
        for sample in batch:
            context = sample.context
            summary_A, summary_B = random.sample(sample.responses[:self.attack_args.num_systems_seen], 2)
             
            attacked_summary_A = summary_A + f' {self.special_tkns_txt}'
            attacked_summary_B = summary_B + f' {self.special_tkns_txt}'
            
            attack_A_ids = self.prep_input(context, attacked_summary_A, summary_B)
            attack_B_ids = self.prep_input(context, summary_A, attacked_summary_B)

            adv_grads_A, output_A = self.token_gradients(attack_A_ids, adv_ids, torch.LongTensor([0]))
            adv_grads_B, output_B = self.token_gradients(attack_B_ids, adv_ids, torch.LongTensor([1]))

            adv_grads_batch.append(adv_grads_A + adv_grads_B)
            
            #for logging
            prob_A = F.softmax(output_A.logits)
            prob_B = F.softmax(output_B.logits)

            logging.append(prob_A[0][0].cpu().item())
            logging.append(prob_B[0][1].cpu().item())

        print(np.mean(logging))
        print(np.mean([i > 0.5 for i in logging]))

        with torch.no_grad():
            adv_grads_batch = torch.stack(adv_grads_batch, dim=0)
            adv_grads = torch.mean(adv_grads_batch, dim=0) # [N x V] N:num adv tokens; V: vocab size
            top_values, top_indices = torch.topk(-1*adv_grads, self.attack_args.topk, dim=1)

        # randomly sample an adv token to substitute with one of the top-k inds; repeat
        for _ in range(self.attack_args.inner_steps):
            tgt_tkn = random.randint(0, self.num_adv_tkns-1)
            substitute = random.randint(0, self.attack_args.topk - 1)
            substitute_id = top_indices[tgt_tkn][substitute]
            adv_ids[tgt_tkn] = substitute_id
        
        adv_phrase = self.tokenizer.decode(adv_ids)

        print(adv_phrase)
        return adv_phrase
    
    def prep_input(self, context, summary_A, summary_B):
        input_text = self.prompt_template.format(context=context, summary_A=summary_A, summary_B=summary_B)
        tok_input = self.tokenizer(input_text, return_tensors='pt').to(self.model.device)
        input_ids = tok_input['input_ids'][0]
        return input_ids

    def token_gradients(self, input_ids, adv_ids, target):
        """
        input_ids must include the self.attack_args.adv_special_tkn where attack_ids are to be placed
        Returns the tensor of gradients for the attack_ids one-hot-encoded vectors
        Gradient is wrt to the loss as per the target label (0 for summA and 1 for summB)

        https://github.com/llm-attacks/llm-attacks/blob/main/llm_attacks/gcg/gcg_attack.py
        """
        assert len(input_ids.shape) == 1, "input must be a 1D torch tensor"

        # fill input with fill_ids
        input_ids = input_ids.clone()
        attack_toks = (input_ids == self.adv_special_tkn_id)
        input_ids[attack_toks] = adv_ids

        # find slice of start and end position of 
        start, stop = np.where(attack_toks.cpu())[0][[0, -1]]
        input_slice = slice(start, stop+1)

        # embed input_ids into one hot encoded inputs
        embed_weights = self.model.get_embedding_matrix()
        one_hot = torch.zeros(
            input_ids[input_slice].shape[0],
            embed_weights.shape[0],
            device=self.model.device,
            dtype=embed_weights.dtype
        )
        
        input_ids = input_ids.to(self.model.device)
        one_hot.scatter_(
            1, 
            input_ids[input_slice].unsqueeze(1),
            torch.ones(one_hot.shape[0], 1, device=self.model.device, dtype=embed_weights.dtype)
        )
        
        one_hot.requires_grad_()
        input_embeds = (one_hot @ embed_weights).unsqueeze(0)

        # now stitch it together with the rest of the embeddings
        embeds = self.model.get_embeddings(input_ids.unsqueeze(0)).detach()

        full_embeds = torch.cat(
            [
                embeds[:,:input_slice.start,:], 
                input_embeds, 
                embeds[:,input_slice.stop:,:]
            ], 
            dim=1)

        output = self.model.forward(inputs_embeds=full_embeds)
        #loss = nn.CrossEntropyLoss()(output.logits, target.to(self.model.device))
        loss = F.cross_entropy(output.logits, target.to(self.model.device))
        loss.backward()

        return one_hot.grad.clone(), output

