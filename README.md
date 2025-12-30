# In-Context Probing for Membership Inference in Fine-Tuned Language Models

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17906756.svg)](https://doi.org/10.5281/zenodo.17906756) [![arXiv](https://img.shields.io/badge/arXiv-2512.16292-b31b1b.svg)](https://arxiv.org/abs/2512.16292)


 This is a repository for the paper "In-Context Probing for Membership Inference in Fine-Tuned Language Models", accepted by NDSS 2026. This is a cleaned-up version of our [ICP-MIA framework repository](https://github.com/RPI-DSPlab/ICP-MIA) to contain only essential scripts for reproducing results in this paper. Our paper is available at [here](./paper.pdf).

 ## Abstract

 > Membership inference attacks (MIAs) pose a critical privacy threat to fine-tuned large language models (LLMs), especially when models are adapted to domain-specific tasks using sensitive data. While prior black-box MIA techniques rely on confidence scores or token likelihoods, these signals are often entangled with a sample’s intrinsic properties—such as content difficulty or rarity—leading to poor generalization and low signal-to-noise ratios. In this paper, we propose ICP-MIA, a novel MIA framework grounded in the theory of training dynamics, particularly the phenomenon of diminishing returns during optimization. We introduce the Optimization Gap as a fundamental signal of membership: at convergence, member samples exhibit minimal remaining loss-reduction potential, while non-members retain significant potential for further optimization. To estimate this gap in a black-box setting, we propose In-Context Probing (ICP)—a training-free method that simulates fine-tuning-like behavior via strategically constructed input contexts. We propose two probing strategies: reference-data-based (using semantically similar public samples) and self-perturbation (via masking or generation). Experiments on three tasks and multiple LLMs show that ICP-MIA significantly outperforms prior black-box MIAs, particularly at low false positive rates. We further analyze how reference data alignment, model type, PEFT configurations, and training schedules affect attack effectiveness. Our findings establish ICP-MIA as a practical and theoretically grounded framework for auditing privacy risks in deployed LLMs.

## Installation

### Install LLama-Factory
```
conda create -n LLamaFactory python=3.10
conda activate LLamaFactory

git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git

cd LLaMA-Factory

pip install -e ".[torch,metrics]" --no-build-isolation
```

### Prepare Dataset

#### Step 1: Download and Split Data

Run the following command to download the dataset and split it:

#### Healthcaremagic:
```bash
python prepare_data.py --dataset lavita/ChatDoctor-HealthCareMagic-100k --output_dir ./data/healthcaremagic
```
#### MedInstruct:
```bash
python prepare_data.py --dataset lavita/AlpaCare-MedInstruct-52k --output_dir ./data/MedInstruct
```

This will create the following files in `./data/healthcaremagic/` 
- `healthcaremagic_train.json` (80% of data) - for model training
- `healthcaremagic_val.json` (10% of data) - for validation
- `healthcaremagic_test.json` (10% of data) - for testing
- `healthcaremagic_attack.json` (1000 members + 1000 non-members with labels) - for attack evaluation 


#### Step 2: Configure LLaMA-Factory Dataset (Example for HealthcareMagic)
Copy the data files to LLaMA-Factory's data directory:

```bash
cp ./data/healthcaremagic/*.json ./LLaMA-Factory/data/
cp ./data/MedInstruct/*.json ./LLaMA-Factory/data/
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

```json
  "MedInstruct_train": {
    "file_name": "MedInstruct_train.json"
  },
  "MedInstruct_val": {
    "file_name": "MedInstruct_train.json"
  },
  "MedInstruct_test": {
    "file_name": "MedInstruct_test.json"
  }
```


### Prepare Target Models

Train your target model using LLaMA-Factory on one GPU:

```bash
cd LLaMA-Factory

CUDA_VISIBLE_DEVICES=0 llamafactory-cli train ../config/config_training.yaml
```



Train your target model using LLaMa-Factory on multi-GPUs:

First, uncomment the deepspeed in `config_training.yaml`
Then:
```bash
cd LLaMA-Factory

pip install deepspeed

llamafactory-cli train ../config/config_training.yaml
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

### Prepaer Attack environment
We separated the attack environment from the training environment, so we need to create another attack environment.
```
conda deactivate 

conda create -n ICPMIA python=3.10

pip install -r requirements.txt
```

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

Please use the second saved checkpoint as target model

```yaml
model:
  target_model_path: "/path/to/your/model"
  device: "cuda:0"

data:
  train_data_path: "/path/to/your/train_data.json"
  test_data_path: "/path/to/your/test_data.json"
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


## Optional Datasets

- **iCliniq**: `lavita/ChatDoctor-iCliniq`
- **AlpaCare-Med-52k**: `lavita/AlpaCare-MedInstruct-52k`
- **TOFU**: `locuslab/TOFU`
