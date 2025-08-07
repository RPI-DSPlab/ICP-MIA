# Baseline Membership Inference Attacks

This directory contains implementations of various baseline membership inference attacks (MIA) for Large Language Models (LLMs). These attacks serve as baselines for comparison with the novel ICP-MIA approach implemented in the parent directory.

## Overview

The baseline attacks implement several well-established membership inference techniques adapted for LLMs. Each attack is designed to determine whether specific data samples were used during the training of a target language model.

## Available Attacks

### 1. Loss-based Attack (`loss`)
**Module**: `attacks.loss.ConditionalLossAttack`

The fundamental MIA approach that uses the conditional loss of the target model on the output portion of examples. Members typically have lower loss values than non-members.

**Key Features**:
- Calculates conditional loss on output text only
- Handles variable-length outputs
- Configurable minimum output length threshold

### 2. Reference Model Attack (`reference`) 
**Module**: `attacks.reference.ReferenceAttack`

Compares the target model's loss against a reference model's loss. The ratio or difference between losses can indicate membership.

**Key Features**:
- Uses an independently trained reference model
- Computes loss ratio between target and reference models
- Effective when reference model is trained on different data

### 3. Zlib Compression Attack (`zlib`)
**Module**: `attacks.zlib.ConditionalZlibAttack`

Combines model loss with text compressibility using zlib compression. Based on the intuition that memorized text should have lower perplexity relative to its compressibility.

**Key Features**:
- Computes zlib compression ratio for outputs
- Combines loss and compression metrics
- Effective for detecting overfitted examples

### 4. Neighborhood Attack (`neighborhood`)
**Module**: `attacks.neighborhood.NeighborhoodAttack`

Perturbs input texts and analyzes the model's confidence on perturbed versions. Members typically show more consistent predictions across perturbations.

**Key Features**:
- Uses masked language model for perturbations
- Configurable number of neighbors and perturbation strategies
- Analyzes prediction consistency

### 5. Min-K% Prob Attack (`minkprob`)
**Module**: `attacks.minkprob.MinkProbAttack`

Analyzes the minimum k% of token probabilities in the model's output. Based on the observation that memorized sequences have fewer low-probability tokens.

**Key Features**:
- Configurable k percentage
- Focuses on tail probability distribution
- Effective for detecting exact memorization

### 6. Min-K%++ Attack (`minkplusplus`)
**Module**: `attacks.minkplusplus.MinkPlusPlusAttack`

An enhanced version of Min-K% that incorporates additional statistical measures and normalization techniques.

**Key Features**:
- Improved statistical analysis over Min-K%
- Better handling of length variations
- Enhanced normalization strategies

### 7. Smaller Perturbation Vector (SPV) Attack (`spv`)
**Module**: `attacks.spv.SPVAttack`

Generates perturbations using a smaller model and analyzes how the target model responds to these perturbations compared to the original text.

**Key Features**:
- Uses T5-based perturbation generation
- Configurable perturbation parameters
- Supports calibration with reference models

### 8. Bag of Words Attack (`bag_of_words`)
**Module**: `attacks.bag_of_words.BagOfWordsAttack`

A simple baseline that uses traditional machine learning on bag-of-words features to distinguish between members and non-members.

**Key Features**:
- Random Forest classifier
- TF-IDF feature extraction
- Fast and interpretable baseline

## Installation

### Requirements

Install the required dependencies:

```bash
pip install torch transformers datasets scikit-learn numpy matplotlib seaborn tabulate pyyaml tqdm
```

Additional requirements for specific attacks:
- **SPV Attack**: `pip install torch transformers[t5]`
- **Neighborhood Attack**: `pip install torch transformers[roberta]`

## Quick Start

### 1. Configuration

Create a configuration file based on the template:

```bash
cp config/config_healthcaremagic.yaml your_config.yaml
```

Edit the configuration with your model and dataset paths:

```yaml
global:
  target_model: "/path/to/your/model"
  datasets:
    - your_dataset: "/path/to/your/dataset"
      split: "train"
  batch_size: 16
  device: "cuda:0"

# Enable desired attacks
loss:
  module: loss

reference:
  module: reference
  ref_model: "/path/to/reference/model"
```

### 2. Running Attacks

```bash
python main.py --config your_config.yaml
```

## Configuration Format

### Global Configuration

```yaml
global:
  target_model: "/path/to/target/model"           # Target model to attack
  datasets:                                       # List of datasets to evaluate
    - dataset_name: "/path/to/dataset"
      split: "train"                              # Dataset split to use
  batch_size: 16                                  # Batch size for inference
  device: "cuda:0"                               # Computing device
  fpr_thresholds: [0.01, 0.05, 0.1]             # FPR thresholds for evaluation
  n_bootstrap_samples: 100                        # Bootstrap samples for confidence intervals
```

### Attack-Specific Configuration

Each attack can be configured with specific parameters:

```yaml
# Loss-based attack
loss:
  module: loss
  min_output_length: 2          # Minimum output length to consider
  default_score: 0.0           # Default score for invalid samples

# Reference model attack
reference:
  module: reference
  ref_model: "/path/to/ref/model"
  device: "cuda:1"
  batch_size: 16

# SPV attack with perturbations
spv:
  module: spv
  mask_model: 'google-t5/t5-base'
  span_length: 1
  pct: 0.2                     # Percentage of tokens to perturb
  sample_number: 10            # Number of perturbations per example
  calibration: true            # Enable reference model calibration
  reference_model: "meta-llama/Llama-3.2-3B-Instruct"
```

## Supported Datasets

The framework supports multiple dataset formats and sources:

### Hugging Face Datasets
- **MIMIR datasets**: Use `mimir_name` in configuration
- **Standard datasets**: Direct dataset loading

### Custom Datasets
- **JSON format**: Instruction-tuning data with `instruction`, `input`, `output` fields
- **Text format**: Pre-training data with `text` field

### Healthcare/Medical Datasets
- **HealthcareMagic**: Medical Q&A dataset
- **MedInstruct**: Medical instruction dataset
- **ChatDoctor**: Medical conversation dataset

## Output and Evaluation

### Metrics
The framework computes standard MIA evaluation metrics:
- **AUC**: Area Under the ROC Curve
- **TPR@FPR**: True Positive Rate at specified False Positive Rates
- **Bootstrap Confidence Intervals**: Statistical significance testing

### Results Format
Results are saved in tabular format showing:
- Attack method performance
- Statistical significance indicators
- Detailed per-sample scores (optional)

### Visualization
- ROC curves for each attack
- Score distribution histograms
- Comparative performance plots

## Implementation Details

### Abstract Attack Interface
All attacks inherit from `AbstractAttack` and implement:
- `__init__()`: Initialize attack with model and configuration
- `run()`: Execute attack on dataset and return scored results
- `signature()`: Generate unique identifier for caching

### Caching System
- Automatic result caching based on attack configuration
- Configurable cache directory
- Hash-based cache invalidation

### Error Handling
- Graceful handling of malformed inputs
- Robust tokenization and encoding
- Memory management for large datasets

## Extending the Framework

### Adding New Attacks

1. Create a new file in `attacks/` directory
2. Inherit from `AbstractAttack`
3. Implement required methods
4. Add configuration entry

Example:
```python
from attacks import AbstractAttack
from datasets import Dataset

class MyCustomAttack(AbstractAttack):
    def __init__(self, name, model, tokenizer, config):
        super().__init__(name, model, tokenizer, config)
        # Custom initialization
    
    def run(self, dataset: Dataset) -> Dataset:
        # Implement attack logic
        return dataset.map(self._compute_score, batched=True)
    
    def _compute_score(self, batch):
        # Score computation logic
        return {"score": scores}
```

### Custom Dataset Loaders

Add dataset loading functions to `utils.py`:

```python
def load_custom_dataset(dataset_path: str) -> Dataset:
    # Custom loading logic
    return Dataset.from_dict({"text": texts, "label": labels})
```

## Performance Optimization

### GPU Memory Management
- Use gradient checkpointing for large models
- Configure appropriate batch sizes
- Enable mixed precision training

### Parallel Processing
- Multi-GPU support for model inference
- Parallel dataset processing
- Asynchronous result computation

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce batch size or use gradient checkpointing
2. **Slow Performance**: Check device configuration and batch sizes
3. **Import Errors**: Ensure all dependencies are installed
4. **Dataset Loading**: Verify file paths and formats

### Debug Mode
Enable detailed logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Citation

Citation information will be provided upon acceptance of the paper.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

This is research code accompanying an anonymous submission. Contributions and feedback are welcome.

## Related Work

- [MIMIR: MIA benchmark suite](https://github.com/iamgroot42/mimir)
- [Min-K% Prob Attack](https://arxiv.org/abs/2306.05000)
- [Loss-based MIA](https://arxiv.org/abs/1610.05820) 