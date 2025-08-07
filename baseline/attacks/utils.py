from typing import Optional

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel

import numpy as np


def compute_nlloss(
    model: PreTrainedModel,
    token_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    ignore_prefix: Optional[int] = None,
):
    with torch.no_grad():
        labels = token_ids.clone()

        outputs = model(token_ids, attention_mask=attention_mask)

        shift_logits = outputs.logits[..., :-1, :].contiguous().view(-1, model.config.vocab_size)
        shift_attention_mask = attention_mask[..., :-1]
        shift_targets = labels[..., 1:]

        shift_targets[shift_attention_mask == 0] = -100

        loss = F.cross_entropy(shift_logits, shift_targets.contiguous().view(-1), reduction="none")
        loss = loss.view(token_ids.shape[0], -1)

        if ignore_prefix:
            loss = loss[:, ignore_prefix:]
            shift_attention_mask = shift_attention_mask[:, ignore_prefix:]

        loss = loss.sum(axis=1) / shift_attention_mask.sum(axis=1)

        return loss.detach().cpu().numpy()


def batch_nlloss(batch, model, tokenizer, device, key='nlloss'):
    
    tokenized = tokenizer.batch_encode_plus(batch['text'], return_tensors='pt', padding="longest")
    token_ids = tokenized['input_ids'].to(device)
    attention_mask = tokenized['attention_mask'].to(device)
    losses = compute_nlloss(model, token_ids, attention_mask)
    return {key: losses}


def get_ll(sentence, model, tokenizer, device):
    input_ids = torch.tensor(tokenizer.encode(sentence)).unsqueeze(0)
    input_ids = input_ids.to(device)
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
    loss, logits = outputs[:2]
    return get_all_prob(input_ids, loss, logits)

def get_conditional_ll(input_text, target_text, model, tokenizer, device):
    input_encodings = tokenizer(input_text, return_tensors="pt")
    target_encodings = tokenizer(target_text, return_tensors="pt")
    concat_ids = torch.cat((input_encodings.input_ids.to(device), target_encodings.input_ids.to(device)), dim=1).long()
    labels = concat_ids.clone()
    labels[:, :input_encodings.input_ids.size(1)] = -100 
    with torch.no_grad():
        outputs = model(concat_ids, labels=labels)
    loss, logits = outputs[:2]
    return get_all_prob(labels, loss, logits)

def get_all_prob(input_ids, loss, logits):
    probabilities = torch.nn.functional.log_softmax(logits, dim=-1)
    all_prob = []
    input_ids_processed = input_ids[0][1:]
    for i, token_id in enumerate(input_ids_processed):
        probability = probabilities[0, i, token_id].item()
        all_prob.append(probability)
    ll = -loss.item()  # log-likelihood
    ppl = torch.exp(loss).item()
    prob = torch.exp(-loss).item()
    return prob, ll , ppl, all_prob, loss.item()


def conditional_min_k_prob(input_text, target_text, model, tokenizer, device, k_percent=20):

    # Tokenize input and target
    input_encodings = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)
    target_encodings = tokenizer(target_text, return_tensors="pt", truncation=True, max_length=512)

    # Concatenate input and target as context + generation
    concat_ids = torch.cat((input_encodings.input_ids.to(device), target_encodings.input_ids.to(device)), dim=1)
    attention_mask = torch.cat((input_encodings.attention_mask.to(device), target_encodings.attention_mask.to(device)), dim=1)

    with torch.no_grad():
        outputs = model(concat_ids, attention_mask=attention_mask)
        logits = outputs.logits

    # Shift logits and labels to calculate log-prob
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = concat_ids[:, 1:].contiguous()
    shift_attention_mask = attention_mask[:, 1:].contiguous()

    # Ignore padding tokens in the loss calculation
    shift_labels[shift_attention_mask == 0] = -100

    # Calculate log-probabilities for each token
    token_logp = -torch.nn.functional.cross_entropy(
        shift_logits.view(-1, model.config.vocab_size),
        shift_labels.view(-1),
        reduction="none"
    )
    token_logp = token_logp.view(concat_ids.size(0), -1)
    
    target_start_idx = input_encodings.input_ids.size(1)
    target_logp = token_logp[:, target_start_idx:]

    target_logp = target_logp.detach().cpu().numpy()
    target_attention = shift_attention_mask[:, target_start_idx:].detach().cpu().numpy()

    # For each sequence, sort and compute lowest k%
    min_k_logp = []
    for logp, mask in zip(target_logp, target_attention):
        logp = logp[mask == 1]  # Only consider valid tokens (mask == 1)
        sorted_logp = np.sort(logp)  # Sort log-probs (ascending order)
        k_count = max(1, int(len(sorted_logp) * (k_percent / 100)))  # Number of tokens to select
        min_k_logp.append(np.mean(sorted_logp[:k_count]))  # Compute mean of lowest k%

    return np.array(min_k_logp), target_logp

