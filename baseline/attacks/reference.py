
import numpy as np
import torch
from attacks import AbstractAttack
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


class ReferenceAttack(AbstractAttack):
    def __init__(self, name, model, tokenizer, config):
        super().__init__(name, model, tokenizer, config)
        self.ref_device = config['device']
        self.ref_model = AutoModelForCausalLM.from_pretrained(config['ref_model'], torch_dtype=torch.float16).to(self.ref_device)
        self.ref_tokenizer = AutoTokenizer.from_pretrained(config['ref_model'])
        print('reference model: '+ config['ref_model'])
        # from loss.py
        self.min_output_length = config.get("min_output_length", 2)
        self.default_score = config.get("default_score", 0.0)

    def run(self, dataset: Dataset) -> Dataset:
        return dataset.map(
            self._compute_reference_score,
            batched=True,
            batch_size=self.config.get("batch_size", 8)
        )

    def _compute_reference_score(self, batch):
        """Calculate reference-based score on output part only, for instruct datasets."""
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
                    
                # Calculate loss on output part for both models
                target_ll = -self._conditional_loss(
                    prompt=prompt, 
                    output=output, 
                    model=self.model, 
                    tokenizer=self.tokenizer,
                    device=self.device
                )
                
                ref_ll = -self._conditional_loss(
                    prompt=prompt,
                    output=output,
                    model=self.ref_model,
                    tokenizer=self.ref_tokenizer,
                    device=self.ref_device
                )
                
                # Score is log_prob_target - log_prob_ref = -loss_target - (-loss_ref) = ref_loss - target_loss
                score = target_ll - ref_ll  
                
                if np.isnan(score) or np.isinf(score):
                    print(f"Warning: NaN or Inf detected for sample {i}. Using default score.")
                    scores.append(self.default_score)
                else:
                    scores.append(float(score))
                
            except Exception as e:
                print(f"Error calculating score for sample {i}: {e}")
                scores.append(self.default_score)
            
        return {self.name: scores}

    def _conditional_loss(self, prompt, output, model, tokenizer, device):
        """Calculate loss on output part only"""
        # Concatenate prompt and output
        full_text = prompt + output
        
        # Tokenize
        full_tokens = tokenizer(full_text, return_tensors='pt')
        prompt_tokens = tokenizer(prompt, return_tensors='pt')
        
        # Get token IDs and attention mask
        input_ids = full_tokens.input_ids.to(device)
        attention_mask = full_tokens.attention_mask.to(device)
        
        # Create labels with prompt part set to -100
        labels = input_ids.clone()
        prompt_length = prompt_tokens.input_ids.size(1)
        if prompt_length > 0:
            labels[:, :prompt_length] = -100
        
        # Check if there are any output tokens to calculate loss on
        output_tokens = (labels != -100).sum().item()
        if output_tokens == 0:
            print(f"Warning: No output tokens to calculate loss on. Output: '{output}'")
            return 0.0  # No output tokens to calculate loss
        
        # Calculate loss
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids, 
                attention_mask=attention_mask, 
                labels=labels
            )
            loss = outputs.loss.item()
        
        return loss
    
