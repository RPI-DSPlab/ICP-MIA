import zlib
import torch
from attacks import AbstractAttack
from datasets import Dataset


class ConditionalZlibAttack(AbstractAttack):
    def __init__(self, name, model, tokenizer, config):
        super().__init__(name, model, tokenizer, config)
        self.device = model.device

    def run(self, dataset: Dataset) -> Dataset:
        return dataset.map(
            self._compute_conditional_zlib_score,
            batched=True,
            batch_size=self.config.get("batch_size", 8)
        )

    def _compute_conditional_zlib_score(self, batch):
        """Calculate conditional Zlib score on output part only"""
        scores = []
        
        for i in range(len(batch["instruction"])):
            # Extract components
            instruction = batch["instruction"][i] if batch["instruction"][i] is not None else ""
            input_text = batch.get("input", [""] * len(batch["instruction"]))[i]
            if input_text is None:
                input_text = ""
            output = batch["output"][i] if batch["output"][i] is not None else ""
            
            # Handle empty outputs
            if not output or len(output) == 0:
                scores.append(0.0)
                continue
                
            # Calculate loss on output part only
            loss = self._conditional_loss(
                prompt=instruction + ("\n" + input_text if input_text else ""), 
                output=output, 
                model=self.model, 
                tokenizer=self.tokenizer
            )
            
            # Calculate zlib entropy only on the output part
            zlib_entropy = len(zlib.compress(output.encode())) / len(output)
            
            # Calculate zlib score
            zlib_score = -loss / zlib_entropy
            scores.append(zlib_score)
            
        return {self.name: scores}

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
        
        # Calculate loss
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids, 
                attention_mask=attention_mask, 
                labels=labels
            )
            loss = outputs.loss.item()
        
        return loss