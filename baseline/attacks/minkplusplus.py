import numpy as np
import torch
import torch.nn.functional as F
from attacks import AbstractAttack
from datasets import Dataset
from transformers import PreTrainedModel


def conditional_min_k_plusplus(model: PreTrainedModel, 
                              token_ids: torch.Tensor, 
                              attention_mask: torch.Tensor, 
                              output_start_indices: torch.Tensor,  # Indices where output begins
                              k: int = 20):
    with torch.no_grad():
        labels = token_ids.clone()
        outputs = model(token_ids, attention_mask=attention_mask)

        shift_logits = outputs.logits[..., :-1, :]
        shift_attention_mask = attention_mask[..., :-1]
        shift_targets = labels[..., 1:]

        # Create a mask to only select tokens from the output part
        batch_size, seq_len = shift_targets.shape
        output_mask = torch.zeros_like(shift_attention_mask)
        
        for i, start_idx in enumerate(output_start_indices):
            # Subtract 1 since we're shifted for next token prediction
            output_start = max(0, start_idx - 1)
            # Only consider tokens in the output part (and respect attention_mask)
            output_mask[i, output_start:] = shift_attention_mask[i, output_start:]

        # Set non-output tokens to be ignored in loss calculation
        masked_targets = shift_targets.clone()
        masked_targets[output_mask == 0] = -100

        # Calculate token-level log probabilities
        token_logp = -F.cross_entropy(shift_logits.contiguous().view(-1, model.config.vocab_size),
                                     masked_targets.contiguous().view(-1), reduction="none")
        token_logp = token_logp.view_as(shift_targets)

        # Calculate mean and std of log softmax only for tokens in the output part
        log_softmax = F.log_softmax(shift_logits, dim=2)
        
        # Calculate the normalized scores
        mu = log_softmax.mean(dim=2)
        sigma = log_softmax.std(dim=2)

        # Compute token scores: (logp - mean) / std
        token_score = (token_logp - mu) / sigma
        
        # Set non-output tokens to a large value so they're ignored in sorting
        token_score[output_mask == 0] = 100
        token_score = token_score.detach().cpu().numpy()

        # Get the k% lowest scores for each sample, considering only output part
        k_min_scores = []
        for i, scores in enumerate(token_score):
            # Filter to only include scores from output part
            output_scores = scores[output_mask[i].cpu().numpy() == 1]
            if len(output_scores) > 0:
                sorted_scores = np.sort(output_scores)
                k_count = max(1, int(k / 100 * len(output_scores)))
                k_min_scores.append(np.mean(sorted_scores[:k_count]))
            else:
                # Fallback in case there are no output tokens
                k_min_scores.append(0.0)
            
    return np.array(k_min_scores)


class ConditionalMinKplusplusAttack(AbstractAttack):
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
        
        # Calculate conditional MinK++
        output_start_indices_tensor = torch.tensor(output_start_indices).to(self.device)
        k_min_scores = conditional_min_k_plusplus(
            self.model, 
            token_ids, 
            attention_mask, 
            output_start_indices_tensor,
            k=self.config['k']
        )
        
        return {self.name: k_min_scores}