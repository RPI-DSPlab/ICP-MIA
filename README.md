# ICP-MIA: In-Context Prefix Membership Inference Attack

This repository implements the In-Context Prefix (ICP) Membership Inference Attack, a novel approach for detecting whether specific data samples were used to train Large Language Models (LLMs). The ICP-MIA leverages in-context learning capabilities to perform membership inference without requiring model modifications or gradient access.

## Overview

ICP-MIA is a black-box membership inference attack that uses carefully crafted in-context prefixes to probe whether target examples were present in the model's training data. The attack works by:

1. **Similarity-based ICP**: Finding semantically similar examples from a reference dataset to use as in-context prefixes
2. **Self-perturbation ICP**: Using perturbed versions of the target example itself as prefixes

## Features

- **Two Attack Variants**: Similarity-based and self-perturbation ICP approaches
- **Flexible Configuration**: YAML-based configuration system for easy experimentation
- **Multiple Data Formats**: Support for both instruction-tuning and pre-training data formats
- **Comprehensive Evaluation**: Built-in metrics including AUC and TPR@FPR calculations
- **Order Effect Analysis**: Optional analysis of training order effects on membership inference
- **Caching System**: Efficient caching for embeddings and intermediate results

## Installation

### Requirements

```bash
pip install torch transformers datasets sentence-transformers faiss-cpu numpy scikit-learn matplotlib pyyaml tqdm
```

For GPU support with FAISS:
```bash
pip install faiss-gpu
```

### Optional Dependencies
- `pandas`: For additional data processing
- `seaborn`: For enhanced plotting capabilities

## Quick Start

### 1. Configuration

Create a configuration file based on the provided templates:

```bash
# For similarity-based ICP
cp config/config_icp_ref.yaml your_config.yaml

# For self-perturbation ICP  
cp config/config_icp_sp.yaml your_config.yaml
```

Edit the configuration file with your model and data paths:

```yaml
model:
  target_model_path: "/path/to/your/model"
  device: "cuda:0"

data:
  train_data_path: "/path/to/your/train_data.json"
  test_data_path: "/path/to/your/test_data.json"
  data_format: "instruction"  # or "pretrain"
```

### 2. Running the Attack

```bash
python icp_mia_attack.py --config your_config.yaml
```

## Configuration Options

### Model Configuration
- `target_model_path`: Path to the target model to attack
- `reference_model_path`: Optional reference model for comparison
- `device`: Computing device (cuda:0, cpu, etc.)
- `max_prompt_tokens`: Maximum tokens in the prompt
- `torch_dtype`: Model precision (float16, float32)

### Data Configuration
- `train_data_path`: Path to training data (JSON format)
- `test_data_path`: Path to test data for evaluation
- `data_format`: Format type ("instruction" or "pretrain")
- `test_size`: Number of test samples to evaluate
- `member_detection_strategy`: How to identify members ("auto", "source_file", "content_match")

### Similarity-based ICP Configuration
- `enabled`: Enable/disable similarity-based attack
- `prefix_pool_source`: Source dataset for finding similar prefixes
- `top_k`: Number of top similar prefixes to use
- `max_prefix_candidates`: Maximum candidates to consider
- `aggregation_strategy`: How to aggregate scores ("max", "min", "mean", "median")
- `embedding_model`: Sentence transformer model for embeddings

### Self-perturbation ICP Configuration
- `enabled`: Enable/disable self-perturbation attack
- `perturbation_file_path`: Path to file containing perturbations
- `perturbation_key`: JSON key for perturbation data
- `top_k`: Number of perturbations to use
- `aggregation_strategy`: Score aggregation method

## Data Formats

### Instruction Format
For instruction-tuning data:
```json
[
  {
    "instruction": "Translate the following text to French:",
    "input": "Hello, how are you?",
    "output": "Bonjour, comment allez-vous?"
  }
]
```

### Pretrain Format
For pre-training data:
```json
[
  {
    "text": "The quick brown fox jumps over the lazy dog."
  }
]
```

## Output

The attack generates detailed results including:
- **AUC Score**: Area under the ROC curve
- **TPR@FPR**: True Positive Rate at specified False Positive Rates
- **Detailed Results**: Individual sample scores and predictions
- **Visualizations**: ROC curves and score distributions

Results are saved in the specified output directory with timestamps.

## Order Effect Analysis

Enable order effect analysis to study how the position of data in training affects membership inference:

```yaml
experiment:
  order_effect_enabled: true
  order_effect_partitions: 10
  order_effect_sample_size: 500
```

This analyzes whether examples seen earlier or later during training are more susceptible to membership inference.

## Advanced Usage

### Custom Prefix Pools
You can provide custom prefix pools for similarity-based attacks:

```yaml
similarity_based_icp:
  prefix_pool_source: "/path/to/custom/prefix_pool.json"
```

## Contributing

This is research code accompanying an anonymous submission. Contributions and feedback are welcome. 