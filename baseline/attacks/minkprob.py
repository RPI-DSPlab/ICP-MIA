import numpy as np
import torch
import torch.nn.functional as F
from attacks import AbstractAttack
from datasets import Dataset
from transformers import PreTrainedModel


def conditional_min_k_prob(model: PreTrainedModel, 
                           token_ids: torch.Tensor, 
                           attention_mask: torch.Tensor, 
                           output_start_indices: torch.Tensor,  # Indices where output begins
                           k: int = 20):
    with torch.no_grad():
        labels = token_ids.clone()
        outputs = model(token_ids, attention_mask=attention_mask)

        # Get logits and targets
        shift_logits = outputs.logits[..., :-1, :].contiguous()
        shift_attention_mask = attention_mask[..., :-1]
        shift_targets = labels[..., 1:].contiguous()

        # Create a mask to only select tokens from the output part
        batch_size, seq_len = shift_targets.shape
        output_mask = torch.zeros_like(shift_attention_mask)
        
        for i, start_idx in enumerate(output_start_indices):
            # The output_start_indices refers to the start position of output
            # We need to subtract 1 since we're shifted for next token prediction
            output_start = max(0, start_idx - 1)
            # Only select tokens in the output part (consider attention_mask)
            output_mask[i, output_start:] = shift_attention_mask[i, output_start:]
        
        # Apply mask to set non-output tokens to -100 (ignored in loss)
        masked_targets = shift_targets.clone()
        masked_targets[output_mask == 0] = -100
        
        # Calculate token-level log probabilities
        token_logp = -F.cross_entropy(
            shift_logits.view(-1, model.config.vocab_size),
            masked_targets.view(-1), 
            reduction="none", 
            ignore_index=-100
        )
        token_logp = token_logp.view(batch_size, -1)
        
        # Filter to only include log probabilities from the output part
        valid_logp = []
        for i in range(batch_size):
            # Get probabilities from output portion only
            sample_output_logp = token_logp[i][output_mask[i] == 1].detach().cpu().numpy()
            if len(sample_output_logp) > 0:
                valid_logp.append(sample_output_logp)
            else:
                valid_logp.append(np.array([0.0]))  # Prevent empty arrays
        
        # Calculate the average of k% lowest probabilities for each sample
        k_min_probas = []
        for probas in valid_logp:
            sorted_probas = np.sort(probas)
            k_count = max(1, int(k / 100 * len(probas)))
            k_min_probas.append(np.mean(sorted_probas[:k_count]))
            
    return np.array(k_min_probas)


class ConditionalMinKProbAttack(AbstractAttack):
    def __init__(self, name, model, tokenizer, config):
        super().__init__(name, model, tokenizer, config)
    
    def run(self, dataset: Dataset) -> Dataset:
        dataset = dataset.map(
            lambda x: self.score(x),
            batched=True,
            batch_size=self.config['batch_size']
        )
        return dataset
    
    def score(self, batch):
        # Directly use the instruction, input, and output fields from the batch
        instructions = batch.get('instruction', [''] * len(batch['text']))
        inputs = batch.get('input', [''] * len(batch['text']))
        outputs = batch.get('output', [''] * len(batch['text']))
        
        # Combine the fields to form complete texts
        texts = []
        for instr, inp, out in zip(instructions, inputs, outputs):
            # Ensure we're not adding None values
            instr = instr if instr is not None else ""
            inp = inp if inp is not None else ""
            out = out if out is not None else ""
            texts.append(instr + inp + out)
        
        # Determine output start indices
        output_start_indices = []
        for i, (instr, inp, _) in enumerate(zip(instructions, inputs, outputs)):
            # Encode instruction and input to find their token length
            instr = instr if instr is not None else ""
            inp = inp if inp is not None else ""
            
            # Tokenize the combined instruction and input to get exact token count
            instruction_input_text = instr + inp
            instruction_input_tokens = self.tokenizer.encode(instruction_input_text)
            
            # The output starts right after the instruction and input
            output_start_idx = len(instruction_input_tokens)
            output_start_indices.append(output_start_idx)
        
        # Tokenize the complete texts for model input
        tokenized = self.tokenizer.batch_encode_plus(texts, return_tensors='pt', padding="longest")
        token_ids = tokenized['input_ids'].to(self.device)
        attention_mask = tokenized['attention_mask'].to(self.device)
        
        # Calculate conditional MinKProb
        output_start_indices_tensor = torch.tensor(output_start_indices).to(self.device)
        k_min_probas = conditional_min_k_prob(
            self.model, 
            token_ids, 
            attention_mask, 
            output_start_indices_tensor,
            k=self.config['k']
        )
        
        return {self.name: k_min_probas}
