import numpy as np
import torch
import random
import re
from attacks import AbstractAttack
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

class ConditionalSPVAttack(AbstractAttack):
    def __init__(self, name: str, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, config):
        super().__init__(name, model, tokenizer, config)
        self.device = model.device
        self.mask_model_name = config["mask_model"]
        self.pattern = re.compile(r"<extra_id_\d+>")
        self.sample_number = config.get("sample_number", 5)
        self.idx_rate = config.get("idx_rate", 0.3)
        self.batch_size = config.get("batch_size", 4)
        self.reference_model_path = config.get("reference_model", None)
        self.calibration = config.get("calibration", False)
        self.config = config
    
        self.mask_model = AutoModelForSeq2SeqLM.from_pretrained(self.mask_model_name).to(self.config["mask_device"])
        self.mask_tokenizer = AutoTokenizer.from_pretrained(self.mask_model_name)
        self.reference_model = AutoModelForCausalLM.from_pretrained(self.reference_model_path).to(self.config["reference_device"]) if self.calibration else None
        self.reference_tokenizer = AutoTokenizer.from_pretrained(self.reference_model_path) if self.calibration else None
        
    def run(self, dataset: Dataset) -> Dataset:
        dataset = dataset.map(
            self._compute_variation_score,
            batched=True,
            batch_size=self.batch_size,
            new_fingerprint=f"{self.signature(dataset)}_v1"
        )
        return dataset

    def _compute_variation_score(self, batch):
        """Compute variation score for response part only"""
        variation_scores = []
        
        for i in range(len(batch["instruction"])):
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
                
            # Handle empty outputs
            if not output:
                variation_scores.append(0.0)
                continue
                
            # Calculate original loss on output part only
            original_loss = self._conditional_loss(
                prompt=prompt, 
                output=output, 
                model=self.model, 
                tokenizer=self.tokenizer
            )
            
            # Generate perturbations and calculate losses
            # Only perturb the output part
            all_perturbed_losses = []
            ref_all_perturbed_losses = [] if self.calibration else None
            
            for _ in range(self.sample_number):
                # Only perturb the response part
                perturbed_output = self.sentence_perturbation([output], self.idx_rate)[0]
                
                # Calculate perturbed loss
                perturbed_loss = self._conditional_loss(
                    prompt=prompt, 
                    output=perturbed_output, 
                    model=self.model, 
                    tokenizer=self.tokenizer
                )
                all_perturbed_losses.append(perturbed_loss)
                
                if self.calibration:
                    ref_perturbed_loss = self._conditional_loss(
                        prompt=prompt, 
                        output=perturbed_output, 
                        model=self.reference_model, 
                        tokenizer=self.reference_tokenizer
                    )
                    ref_all_perturbed_losses.append(ref_perturbed_loss)
            
            # Calculate variation score
            perturbed_mean_loss = np.mean(all_perturbed_losses)
            variation_score = perturbed_mean_loss - original_loss
            
            if self.calibration:
                ref_original_loss = self._conditional_loss(
                    prompt=prompt, 
                    output=output, 
                    model=self.reference_model, 
                    tokenizer=self.reference_tokenizer
                )
                ref_perturbed_mean_loss = np.mean(ref_all_perturbed_losses)
                ref_variation_score = ref_perturbed_mean_loss - ref_original_loss
                variation_score = variation_score - ref_variation_score
                
            variation_scores.append(variation_score)
            
        return {self.name: variation_scores}

    def _conditional_loss(self, prompt, output, model, tokenizer):
        """Calculate loss on output part only"""
        # Concatenate prompt and output
        full_text = prompt + output
        
        # Tokenize
        full_tokens = tokenizer(full_text, return_tensors='pt')
        prompt_tokens = tokenizer(prompt, return_tensors='pt')
        
        # Get token IDs and attention mask
        input_ids = full_tokens.input_ids.to(model.device)
        attention_mask = full_tokens.attention_mask.to(model.device)
        
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

    def tokenize_and_mask(self, text, span_length, pct, idx_rate, ceil_pct=False):
        tokens = text.split(' ')
        mask_string = '<<<mask>>>'
        n_spans = pct * len(tokens) / (span_length + self.config["buffer_size"] * 2)
        if ceil_pct:
            n_spans = np.ceil(n_spans)
        n_spans = int(n_spans)
        n_masks = 0
        while n_masks < n_spans:
            start = random.randint(0, len(tokens) - span_length)
            end = start + span_length
            search_start = max(0, start - self.config["buffer_size"])
            search_end = min(len(tokens), end + self.config["buffer_size"])
            if mask_string not in tokens[search_start:search_end]:
                tokens[start:end] = [mask_string]
                n_masks += 1
        num_filled = 0
        for idx, token in enumerate(tokens):
            if token == mask_string:
                tokens[idx] = f'<extra_id_{num_filled}>'
                num_filled += 1
        assert num_filled == n_masks, f"num_filled {num_filled} != n_masks {n_masks}"
        return ' '.join(tokens)

    @staticmethod
    def count_masks(texts):
        return [len([x for x in text.split() if x.startswith("<extra_id_")]) for text in texts]

    def replace_masks(self, texts):
        n_expected = self.count_masks(texts)
        if len(n_expected) == 0:
            stop_id = self.mask_tokenizer.eos_token_id
        else:
            stop_id = self.mask_tokenizer.encode(f"<extra_id_{max(n_expected)}>")[0]  # Note: may need +1

        tokenized = self.mask_tokenizer(texts, return_tensors="pt", padding=True).to(self.device)
        outputs = self.mask_model.generate(
            **tokenized,
            max_length=512,       
            do_sample=True,
            top_p=self.config["mask_top_p"],
            num_return_sequences=1,
            eos_token_id=stop_id
        )
        return self.mask_tokenizer.batch_decode(outputs, skip_special_tokens=False)

    def extract_fills(self, texts):
        texts = [x.replace("<pad>", "").replace("</s>", "").strip() for x in texts]
        extracted_fills = [self.pattern.split(x)[1:-1] for x in texts]
        extracted_fills = [[y.strip() for y in fills] for fills in extracted_fills]
        return extracted_fills

    def apply_extracted_fills(self, masked_texts, extracted_fills):
        tokens_list = [x.split(' ') for x in masked_texts]
        n_expected = self.count_masks(masked_texts)
        for idx, (tokens, fills, n) in enumerate(zip(tokens_list, extracted_fills, n_expected)):
            if len(fills) < n:
                tokens_list[idx] = []
            else:
                for fill_idx in range(n):
                    try:
                        mask_index = tokens.index(f"<extra_id_{fill_idx}>")
                        tokens[mask_index] = fills[fill_idx]
                    except ValueError:
                        continue
        return [" ".join(tokens) for tokens in tokens_list]

    def sentence_perturbation(self, texts, idx_rate):
        cfg = self.config
        masked_texts = [self.tokenize_and_mask(x, cfg["span_length"], cfg["pct"], idx_rate, cfg["ceil_pct"]) for x in texts]
        raw_fills = self.replace_masks(masked_texts)
        extracted_fills = self.extract_fills(raw_fills)
        perturbed_texts = self.apply_extracted_fills(masked_texts, extracted_fills)
        attempts = 1
        while '' in perturbed_texts:
            idxs = [idx for idx, x in enumerate(perturbed_texts) if x == '']
            print(f'WARNING: {len(idxs)} texts have no fills. Retrying [attempt {attempts}]...')
            masked_texts_retry = [self.tokenize_and_mask(texts[idx], cfg["span_length"], cfg["pct"], idx_rate, cfg["ceil_pct"]) for idx in idxs]
            raw_fills_retry = self.replace_masks(masked_texts_retry)
            extracted_fills_retry = self.extract_fills(raw_fills_retry)
            new_perturbed_texts = self.apply_extracted_fills(masked_texts_retry, extracted_fills_retry)
            for i, idx in enumerate(idxs):
                perturbed_texts[idx] = new_perturbed_texts[i]
            attempts += 1
        return perturbed_texts