import torch
import numpy as np
from attacks import AbstractAttack
from datasets import Dataset


class ConditionalLossAttack(AbstractAttack):
    def __init__(self, name, model, tokenizer, config):
        super().__init__(name, model, tokenizer, config)
        self.device = model.device
        # Minimum output length to consider for valid computation
        self.min_output_length = config.get("min_output_length", 2)
        # Default score to use when calculation would be invalid
        self.default_score = config.get("default_score", 0.0)

    def run(self, dataset: Dataset) -> Dataset:
        # Always recalculate the loss focusing only on the output part
        return dataset.map(
            self._compute_conditional_loss,
            batched=True,
            batch_size=self.config.get("batch_size", 8)
        )

    def _compute_conditional_loss(self, batch):
        """Calculate conditional loss on output part only"""
        losses = []
        
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
                    losses.append(self.default_score)
                    continue
                    
                # Calculate loss on output part only
                loss = self._conditional_loss(
                    prompt=prompt, 
                    output=output, 
                    model=self.model, 
                    tokenizer=self.tokenizer
                )
                
                # Check for NaN or infinite values
                neg_loss = -loss
                if np.isnan(neg_loss) or np.isinf(neg_loss):
                    print(f"Warning: NaN or Inf detected for sample {i}. Using default score.")
                    losses.append(self.default_score)
                else:
                    losses.append(float(neg_loss))  # Ensure it's a regular float
                
            except Exception as e:
                # Catch any exceptions during calculation
                print(f"Error calculating loss for sample {i}: {e}")
                losses.append(self.default_score)
            
        return {self.name: losses}

    def _conditional_loss(self, prompt, output, model, tokenizer):
        """Calculate loss on output part only"""
        # Concatenate prompt and output
        full_text = prompt + output
        
        # Tokenize
        full_tokens = tokenizer(full_text, return_tensors='pt')
        prompt_tokens = tokenizer(prompt, return_tensors='pt')
        
        # Get token IDs and attention mask
        input_ids = full_tokens.input_ids.to(self.device)
        attention_mask = full_tokens.attention_mask.to(self.device)
        
        # Create labels with prompt part set to -100
        labels = input_ids.clone()
        prompt_length = prompt_tokens.input_ids.size(1)
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