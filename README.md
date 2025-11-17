# ICP-MIA

This repository implements ICP-MIA (In-Context Probing Membership Inference Attack), a novel black-box attack framework for detecting whether specific data samples were used during fine-tuning of LLMs. Our method introduces the Optimization Gap—the disparity in remaining loss-reduction potential between member and non-member samples—as a fundamental membership signal. At convergence, member samples exhibit minimal remaining optimization potential, while non-member samples retain significant room for further improvement.

## Installation

### Install LLama-Factory
```
conda create -n LLamaFactory
conda activate LLamaFactory

git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git

cd LLaMA-Factory

pip install -e ".[torch,metrics]" --no-build-isolation
```

### Prepare Dataset

#### Step 1: Download and Split Data

Run the following command to download the HealthCareMagic-100k dataset and split it:

```bash
python prepare_data.py
```

This will create the following files in `./data/healthcaremagic/`:
- `healthcaremagic_train.json` (80% of data) - for model training
- `healthcaremagic_val.json` (10% of data) - for validation
- `healthcaremagic_test.json` (10% of data) - for testing
- `healthcaremagic_attack.json` (1000 members + 1000 non-members with labels) - for attack evaluation 

#### Step 2: Configure LLaMA-Factory Dataset

Copy the data files to LLaMA-Factory's data directory:

```bash
cp ./data/healthcaremagic/*.json ./LLaMA-Factory/data/
```

Then add the following entries to `./LLaMA-Factory/data/dataset_info.json`:

```json
  "healthcaremagic_train": {
    "file_name": "healthcaremagic_train.json"
  },
  "healthcaremagic_val": {
    "file_name": "healthcaremagic_val.json"
  },
  "healthcaremagic_test": {
    "file_name": "healthcaremagic_test.json"
  }
```

### Prepare Target Models

Train your target model using LLaMA-Factory:

```bash
cd LLaMA-Factory

CUDA_VISIBLE_DEVICES=0 llamafactory-cli train ../config/config_training.yaml
```

### Prepare Attack Dataset

Generate perturbations for the attack dataset created by `prepare_data.py`:

```bash
python generate_perturbations.py convert \
  --input ./data/healthcaremagic/healthcaremagic_attack.json \
  --output ./data/healthcaremagic/healthcaremagic_attack_perturbed.json \
  --mask_rate 0.7 \
  --num_perturbations 20
```

The output will be in `target_example` format with `mask_perturbations` and `label` fields.

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
  data_format: "instruction"  # or "text"
```

### 2. Running the Attack

```bash
python icp_mia_attack.py --config your_config.yaml
```

## Configuration Options

### Data Configuration
- `train_data_path`: Path to training data (JSON format as shown before)
- `test_data_path`: Path to test data for evaluation
- `data_format`: Format type ("instruction" or "pretrain")
- `test_size`: Number of test samples to evaluate

### Similarity-based ICP Configuration
- `enabled`: Enable/disable similarity-based attack
- `prefix_pool_source`: Source dataset for finding similar prefixes
- `top_k`: Number of top similar prefixes to use
- `max_prefix_candidates`: Maximum candidates to consider
- `aggregation_strategy`: How to aggregate scores ("max", "min", "mean", "median")
- `embedding_model`: Sentence transformer model for calculating similarity

### Self-perturbation ICP Configuration
- `enabled`: Enable/disable self-perturbation attack
- `perturbation_file_path`: Path to file containing perturbations
- `top_k`: Number of perturbations to use
- `aggregation_strategy`: Score aggregation method

## Output

The attack generates detailed results including:
- **AUC Score**: Area under the ROC curve
- **TPR@FPR**: True Positive Rate at specified False Positive Rates

Results are saved in the specified output directory with timestamps.

### Custom Prefix Pools
You can provide custom prefix pools for similarity-based attacks:

```yaml
similarity_based_icp:
  prefix_pool_source: "/path/to/custom/prefix_pool.json"
```
### Other datasets HuggingFace Path

lavita/ChatDoctor-iCliniq
 