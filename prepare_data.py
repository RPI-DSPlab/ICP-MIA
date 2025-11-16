import json
import os
import random
from datasets import load_dataset
from sklearn.model_selection import train_test_split


def download_and_split_dataset(
    dataset_name="lavita/ChatDoctor-HealthCareMagic-100k",
    output_dir="./data/healthcaremagic",
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
    attack_samples=1000,
    seed=42
):

    print(f"Downloading dataset: {dataset_name}")
    
    # Download dataset from HuggingFace
    dataset = load_dataset(dataset_name)
    
    # Get the train split (assuming the dataset has a 'train' split)
    if 'train' in dataset:
        data = dataset['train']
    else:
        # If no train split, use the first available split
        data = dataset[list(dataset.keys())[0]]
    
    print(f"Total samples: {len(data)}")
    
    # Convert to list of dictionaries
    all_data = []
    for item in data:
        # Keep the standard instruction format for LLaMA-Factory
        sample = {
            "instruction": item.get("instruction", ""),
            "input": item.get("input", ""),
            "output": item.get("output", "")
        }
        all_data.append(sample)
    
    # First split: separate train from (val + test)
    train_data, temp_data = train_test_split(
        all_data,
        test_size=(val_ratio + test_ratio),
        random_state=seed
    )
    
    # Second split: separate val from test
    val_data, test_data = train_test_split(
        temp_data,
        test_size=test_ratio / (val_ratio + test_ratio),
        random_state=seed
    )
    
    print(f"Train samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")
    print(f"Test samples: {len(test_data)}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save datasets to JSON files
    train_path = os.path.join(output_dir, "healthcaremagic_train.json")
    val_path = os.path.join(output_dir, "healthcaremagic_val.json")
    test_path = os.path.join(output_dir, "healthcaremagic_test.json")
    
    print(f"\nSaving datasets to {output_dir}")
    
    with open(train_path, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    print(f"  - Saved train set to: {train_path}")
    
    with open(val_path, 'w', encoding='utf-8') as f:
        json.dump(val_data, f, ensure_ascii=False, indent=2)
    print(f"  - Saved validation set to: {val_path}")
    
    with open(test_path, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)
    print(f"  - Saved test set to: {test_path}")
    
    # Create attack dataset with labels
    print(f"\n{'='*60}")
    print("Creating attack evaluation dataset...")
    print(f"{'='*60}")
    
    random.seed(seed)
    
    # Sample from train and test
    if len(train_data) < attack_samples:
        print(f"Warning: Only {len(train_data)} training samples available, using all")
        train_samples = train_data.copy()
    else:
        train_samples = random.sample(train_data, attack_samples)
    
    if len(test_data) < attack_samples:
        print(f"Warning: Only {len(test_data)} test samples available, using all")
        test_samples = test_data.copy()
    else:
        test_samples = random.sample(test_data, attack_samples)
    
    # Add labels
    attack_data = []
    for sample in train_samples:
        sample_copy = sample.copy()
        sample_copy['label'] = 1  # member
        attack_data.append(sample_copy)
    
    for sample in test_samples:
        sample_copy = sample.copy()
        sample_copy['label'] = 0  # non-member
        attack_data.append(sample_copy)
    
    # Shuffle the combined data
    random.shuffle(attack_data)
    
    print(f"  - Members (label=1): {len(train_samples)}")
    print(f"  - Non-members (label=0): {len(test_samples)}")
    print(f"  - Total: {len(attack_data)}")
    
    # Save attack dataset
    attack_path = os.path.join(output_dir, "healthcaremagic_attack.json")
    with open(attack_path, 'w', encoding='utf-8') as f:
        json.dump(attack_data, f, ensure_ascii=False, indent=2)
    print(f"  - Saved attack dataset to: {attack_path}")
    
    print("\nAll data preparation completed successfully!")
    
    return train_data, val_data, test_data, attack_data


if __name__ == "__main__":
    # Download and split the dataset
    download_and_split_dataset(
        dataset_name="lavita/ChatDoctor-HealthCareMagic-100k",
        output_dir="./data/healthcaremagic",
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        attack_samples=1000,
        seed=42
    )

