import importlib
import random
from typing import Any, Dict

import numpy as np
import torch
import yaml
from attacks import AbstractAttack
from datasets import Dataset, load_dataset
from transformers import PreTrainedModel, PreTrainedTokenizer
import json
from collections import defaultdict

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, 'r') as config_file:
        return yaml.safe_load(config_file)


def format_demo(self, instruction: str, output: str, input: str = ""):
    if input.strip():
        user_msg = f"{instruction.strip()}\n{input.strip()}"
    else:
        user_msg = instruction.strip()
    return (
        f"<|start_header_id|>user<|end_header_id|>\n\n{user_msg}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n{output.strip()}<|eot_id|>"
    )

def build_query_prompt(self, instruction: str, input: str = ""):
    if input.strip():
        user_msg = f"{instruction.strip()}\n{input.strip()}"
    else:
        user_msg = instruction.strip()
    return (
        f"<|start_header_id|>user<|end_header_id|>\n\n{user_msg}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n"
    )

def load_attack(
    attack_name: str,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    config: Dict[str, Any]
) -> AbstractAttack:
    try:
        module = importlib.import_module(f"attacks.{config['module']}")

        ret = None
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if isinstance(attr, type) and issubclass(attr, AbstractAttack) and attr is not AbstractAttack:
                if ret is None:
                    ret = attr(
                        name=attack_name,
                        model=model,
                        tokenizer=tokenizer,
                        config=config
                    )
                else:
                    raise ValueError(f"Multiple classes implementing AlgorithmInterface found in {attack_name}")

        if ret is not None:
            return ret
        else:
            raise ValueError(f"No class implementing AlgorithmInterface found in {attack_name}")
    except ImportError as e:
        raise ValueError(f"Failed to import algorithm '{attack_name}': {str(e)}")


def get_available_attacks(config) -> list:
    return set(config.keys()) - {"global"}


def load_mimir_dataset(name: str, split: str) -> Dataset:

    dataset = load_dataset("iamgroot42/mimir", name, split=split)

    assert 'member' in dataset.column_names
    assert 'nonmember' in dataset.column_names

    all_texts = [dataset['member'][k] for k in range(len(dataset))]
    all_labels = [1] * len(dataset)
    all_texts += [dataset['nonmember'][k] for k in range(len(dataset))]
    all_labels += [0] * len(dataset)
    
    new_dataset = Dataset.from_dict({"text": all_texts, "label": all_labels})
    
    return new_dataset

def split_texts_by_group(texts, group_size, ratio, seed=None):

    groups = [(texts[i:i + group_size], list(range(i, min(i + group_size, len(texts))))) 
              for i in range(0, len(texts), group_size)]

    if seed is not None:
        random.seed(seed)

    random.shuffle(groups)

    split_point = int(len(groups) * ratio)
    part1_groups = groups[:split_point]
    part2_groups = groups[split_point:]

    part1 = [text for group, _ in part1_groups for text in group]
    part2 = [text for group, _ in part2_groups for text in group]
    part1_indices = [idx for _, indices in part1_groups for idx in indices]
    part2_indices = [idx for _, indices in part2_groups for idx in indices]

    return part1, part2, part1_indices, part2_indices


from datasets import Dataset, concatenate_datasets, load_dataset

def load_heathcaremagic_dataset(name: str, split: str) -> Dataset:
    """
    Load and prepare the HealthCareMagic dataset
    
    Args:
        name: Dataset name
        split: Split name (not used but kept for API consistency)
    
    Returns:
        Dataset with randomly selected samples (500 members and 500 non-members)
    """
    new_dataset = load_dataset(name)
    member = new_dataset['train']
    nonmember = new_dataset['test']
    samples_per_class = 500
    member_size = min(samples_per_class, len(member))
    nonmember_size = min(samples_per_class, len(nonmember))
    member = member.select(range(member_size))
    nonmember = nonmember.select(range(nonmember_size))
    
    member_texts = [format_demo(instruction, output, input) for instruction, input, output in zip(member['instruction'], member["input"], member['output'])]
    nonmember_texts = [format_demo(instruction, output, input) for instruction, input, output in zip(nonmember['instruction'], nonmember["input"], nonmember['output'])]
    
    member_labels = [1] * member_size
    nonmember_labels = [0] * nonmember_size
    
    # Combine features
    all_features = {}
    
    # Add features from both datasets
    for key in member.features:
        if key in nonmember.features:
            all_features[key] = member[key] + nonmember[key]
    
    # Add text and label columns
    all_features['text'] = member_texts + nonmember_texts
    all_features['label'] = member_labels + nonmember_labels
    
    # Create the combined dataset
    combined_dataset = Dataset.from_dict(all_features)
    return combined_dataset


def load_MedInstruct_dataset(name: str, split: str) -> Dataset:
    new_dataset = load_dataset(name)
    # # shuffle the dataset
    new_dataset = new_dataset.shuffle()
    
    member = new_dataset['train']
    nonmember = new_dataset['test']
    samples_per_class = 500
    member_size = min(samples_per_class, len(member))
    nonmember_size = min(samples_per_class, len(nonmember))
    member = member.select(range(member_size))
    nonmember = nonmember.select(range(nonmember_size))
    member_texts = [format_demo(instruction, output, input) for instruction, input, output in zip(member['instruction'], member["input"], member['output'])]
    nonmember_texts = [format_demo(instruction, output, input) for instruction, input, output in zip(nonmember['instruction'], nonmember["input"], nonmember['output'])]
    member_labels = [1] * member_size
    nonmember_labels = [0] * nonmember_size
    # Combine features
    all_features = {}
    # Add features from both datasets
    for key in member.features:
        if key in nonmember.features:
            all_features[key] = member[key] + nonmember[key]
    # Add text and label columns
    all_features['text'] = member_texts + nonmember_texts
    all_features['label'] = member_labels + nonmember_labels
    # Create the combined dataset
    combined_dataset = Dataset.from_dict(all_features)
    return combined_dataset


    
def load_CNNDM_dataset(name:str, split:str) -> Dataset:
    
    new_dataset = load_dataset(name)
    # shuffle the dataset
    member = new_dataset['train']
    nonmember = new_dataset['test']
    samples_per_class = 500
    member_size = min(samples_per_class, len(member))
    nonmember_size = min(samples_per_class, len(nonmember))
    member = member.select(range(member_size))
    nonmember = nonmember.select(range(nonmember_size))
    member_texts = [format_demo(instruction, output, input) for instruction, input, output in zip(member['instruction'], member["input"], member['output'])]
    nonmember_texts = [format_demo(instruction, output, input) for instruction, input, output in zip(nonmember['instruction'], nonmember["input"], nonmember['output'])]
    member_labels = [1] * member_size
    nonmember_labels = [0] * nonmember_size
    # Combine features
    all_features = {}
    # Add features from both datasets
    for key in member.features:
        if key in nonmember.features:
            all_features[key] = member[key] + nonmember[key]
    # Add text and label columns
    all_features['text'] = member_texts + nonmember_texts
    all_features['label'] = member_labels + nonmember_labels
    # Create the combined dataset
    combined_dataset = Dataset.from_dict(all_features)
    return combined_dataset



    

