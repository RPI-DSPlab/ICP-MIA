# Adapted from https://github.com/mireshghallah/neighborhood-curvature-mia/

import logging
from heapq import nlargest

import numpy as np
import torch
import torch.nn.functional as F
from attacks import AbstractAttack
from datasets import Dataset
from transformers import AutoModelForMaskedLM, AutoTokenizer

class ConditionalNeighborhoodAttack(AbstractAttack):
    def __init__(self, name, model, tokenizer, config):
        super().__init__(name, model, tokenizer, config)
        self.mlm_device = config['device']
        self.mlm_model = AutoModelForMaskedLM.from_pretrained(
            config['mlm_model'], torch_dtype=torch.float16).to(self.mlm_device)
        self.mlm_tokenizer = AutoTokenizer.from_pretrained(config['mlm_model'])
        self.n_neighbors = config['n_neighbors']
        self.top_k = config['top_k']
        self.is_scale_embeds = config['is_scale_embeds']
        # Minimum output length to consider for valid computation
        self.min_output_length = config.get("min_output_length", 5)
        # Default score to use when calculation would be invalid
        self.default_score = config.get("default_score", 0.0)

    def run(self, dataset: Dataset) -> Dataset:
        dataset = dataset.map(
            self._compute_conditional_neighborhood_score,
            batched=True,
            batch_size=self.config.get("batch_size", 8),
            new_fingerprint=f"{self.signature(dataset)}_v1",
        )
        return dataset

    def _compute_conditional_neighborhood_score(self, batch):
        """Calculate conditional neighborhood score on output part only"""
        scores = []
        
        for i in range(len(batch["instruction"])):
            try:
                # Extract components
                instruction = batch["instruction"][i] if batch["instruction"][i] is not None else ""
                input_text = batch.get("input", [""] * len(batch["instruction"]))[i]
                if input_text is None:
                    input_text = ""
                output = batch["output"][i] if batch["output"][i] is not None else ""
                
                # Simply concatenate instruction and input to form prompt
                prompt = instruction
                if input_text:
                    prompt += "\n" + input_text
                
                # Handle empty or very short outputs
                if not output or len(output) < self.min_output_length:
                    scores.append(self.default_score)
                    continue
                
                # Calculate neighborhood score
                score = self.conditional_neighborhood_score(prompt, output)
                
                # Check for NaN or infinite values
                if np.isnan(score) or np.isinf(score):
                    print(f"Warning: NaN or Inf detected for sample {i}. Using default score.")
                    scores.append(self.default_score)
                else:
                    scores.append(float(score))
                
            except Exception as e:
                # Catch any exceptions during calculation
                print(f"Error calculating neighborhood score for sample {i}: {e}")
                scores.append(self.default_score)
        
        return {self.name: scores}

    def conditional_neighborhood_score(self, prompt: str, output: str):
        """Calculate neighborhood score for output part only, given prompt"""
        with torch.no_grad():
            # Calculate original score for the output part only
            original_score = self.get_conditional_logprob(prompt, output)
            
            # Generate neighbors for the output part only
            neighbors = [x[0] for x in self.generate_neighbors_for_output(output)]
            
            if not neighbors:
                # If no neighbors could be generated, return the original score
                return original_score
            
            # Calculate scores for neighbors (all conditioned on the same prompt)
            neighbor_scores = []
            for neighbor_output in neighbors:
                neighbor_score = self.get_conditional_logprob(prompt, neighbor_output)
                neighbor_scores.append(neighbor_score)
            
            final_score = original_score - np.mean(neighbor_scores)
            return final_score

    def get_conditional_logprob(self, prompt: str, output: str):
        """Calculate log probability of output given prompt"""
        # Concatenate prompt and output
        full_text = prompt + output
        
        # Tokenize
        full_tokens = self.tokenizer(full_text, return_tensors='pt')
        prompt_tokens = self.tokenizer(prompt, return_tensors='pt')
        
        # Get token IDs
        input_ids = full_tokens.input_ids.to(self.device)
        prompt_length = prompt_tokens.input_ids.size(1)
        
        # Calculate loss only on output part
        ce_loss = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=self.tokenizer.pad_token_id)
        logits = self.model(input_ids, labels=input_ids).logits[:, :-1, :].transpose(1, 2)
        manual_logprob = -ce_loss(logits, input_ids[:, 1:])
        
        # Only consider tokens from the output part
        output_logprob = manual_logprob[:, prompt_length-1:]  # -1 because of shifting
        mask = output_logprob != 0
        
        if mask.sum() == 0:
            return 0.0  # No valid output tokens
        
        manual_logprob_mean = (output_logprob * mask).sum() / mask.sum()
        return manual_logprob_mean.item()

    def generate_neighbors_for_output(self, output: str):
        """Generate neighbors for the output text only"""
        max_len = self.mlm_tokenizer.model_max_length
        tokenized = self.mlm_tokenizer(output, return_tensors='pt', truncation=False).input_ids[0]

        # Check if the sequence exceeds the max length
        if len(tokenized) > max_len:
            logging.warning(
                f"Warning: output is longer than max input length for MLM ({max_len}). Using first chunk.")
            # Truncate the tokenized input to the max length
            truncated_tokenized = tokenized[:max_len]
        else:
            truncated_tokenized = tokenized
        
        # Generate replacements for the output
        texts_with_replacements = self._generate_neighbors_for_chunk(truncated_tokenized)
        return texts_with_replacements

    def _generate_neighbors_for_chunk(self, tokenized_chunk):
        """Generate neighbors for a tokenized chunk"""
        # Skip if the chunk is too short to meaningfully perturb
        if len(tokenized_chunk) <= 3:  # Start, content, end tokens at minimum
            return []
            
        text_tokenized = tokenized_chunk.unsqueeze(0).to(self.mlm_device)
        replacements = dict()

        for target_token_index in range(1, len(text_tokenized[0])-1):
            target_token = text_tokenized[0, target_token_index]

            embeds = self.mlm_model.roberta.embeddings(text_tokenized)
            embeds = torch.cat((embeds[:, :target_token_index, :],
                                F.dropout(embeds[:, target_token_index, :], p=0.7,
                                          training=self.is_scale_embeds).unsqueeze(dim=0),
                                embeds[:, target_token_index+1:, :]), dim=1)

            token_probs = torch.softmax(self.mlm_model(inputs_embeds=embeds).logits, dim=2)

            token_probs[:, :, self.mlm_tokenizer.bos_token_id] = 0
            token_probs[:, :, self.mlm_tokenizer.eos_token_id] = 0

            original_prob = token_probs[0, target_token_index, target_token]

            top_probabilities, top_candidates = torch.topk(token_probs[:, target_token_index, :], self.top_k, dim=1)

            for cand, prob in zip(top_candidates[0], top_probabilities[0]):
                if not cand == target_token:
                    if original_prob.item() == 1:
                        replacements[(target_token_index, cand)] = prob.item()/(1-0.9)
                    else:
                        replacements[(target_token_index, cand)] = prob.item()/(1-original_prob.item())

        # If no replacements found, return empty list
        if not replacements:
            return []

        highest_scored = nlargest(min(self.n_neighbors, len(replacements)), replacements, key=replacements.get)

        texts = []
        for single in highest_scored:
            try:
                alt = text_tokenized.to("cpu")
                target_token_index, cand = single
                alt = torch.cat((alt[:, 1:target_token_index], torch.LongTensor([cand]).unsqueeze(0),
                                alt[:, target_token_index+1:-1]), dim=1)
                alt_text = self.mlm_tokenizer.batch_decode(alt)[0]
                texts.append((alt_text, replacements[single]))
            except Exception as e:
                logging.warning(f"Error generating neighbor: {e}")
                continue

        return texts