import os
import json
import yaml
import torch
import faiss
import numpy as np
import argparse
from tqdm import tqdm
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

import pandas as pd
from datasets import load_dataset, Dataset
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Configuration for target and reference models"""
    target_model_path: str
    reference_model_path: Optional[str] = None
    device: str = "cuda:0"
    max_prompt_tokens: int = 8192
    torch_dtype: str = "float16"

@dataclass
class DataConfig:
    """Configuration for datasets"""
    train_data_path: str
    test_data_path: str
    data_format: str = "instruction"  
    test_size: int = 500
    random_seed: int = 42
    enable_shuffle: bool = True  
    # Member detection strategy: "source_file", "content_match", "relaxed_match", "auto"
    member_detection_strategy: str = "auto"
    # Fallback when no members found
    force_balanced_split: bool = False

@dataclass
class SimilarityBasedICPConfig:
    """Configuration for similarity-based ICP-MIA"""
    enabled: bool = True
    prefix_pool_source: str = "dolly"  # "dolly" or local path
    top_k: int = 1
    max_prefix_candidates: int = 20
    aggregation_strategy: str = "max"  # "max", "min", "mean", "median"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

@dataclass
class SelfPerturbationICPConfig:
    """Configuration for self-perturbation ICP-MIA"""
    enabled: bool = False
    perturbation_file_path: Optional[str] = None
    perturbation_key: str = "perturbations"  # Key to use for perturbations
    top_k: int = 5
    aggregation_strategy: str = "mean"  # "max", "min", "mean", "median"

@dataclass
class ExperimentConfig:
    """Configuration for experiment settings"""
    output_dir: str = "./results"
    cache_dir: str = "./cache"
    experiment_name: Optional[str] = None
    save_detailed_results: bool = True
    fpr_thresholds: List[float] = None
    # --- Order effect analysis settings ---
    order_effect_enabled: bool = False  # Enable training-order effect study
    order_effect_partitions: int = 10   # Number of contiguous splits of the training set
    order_effect_sample_size: int = 500 # Samples drawn (with fixed seed) from each partition
    order_effect_plot_file: str = "order_effect_icp_auc.png"

    def __post_init__(self):
        if self.fpr_thresholds is None:
            self.fpr_thresholds = [0.01, 0.05, 0.1]
        if self.experiment_name is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.experiment_name = f"mia_experiment_{timestamp}"

@dataclass
class MIAConfig:
    """Main configuration class for MIA experiments"""
    model: ModelConfig
    data: DataConfig
    similarity_based_icp: SimilarityBasedICPConfig
    self_perturbation_icp: SelfPerturbationICPConfig
    experiment: ExperimentConfig

    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'MIAConfig':
        """Load configuration from YAML file"""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return cls(
            model=ModelConfig(**config_dict['model']),
            data=DataConfig(**config_dict['data']),
            similarity_based_icp=SimilarityBasedICPConfig(**config_dict.get('similarity_based_icp', {})),
            self_perturbation_icp=SelfPerturbationICPConfig(**config_dict.get('self_perturbation_icp', {})),
            experiment=ExperimentConfig(**config_dict.get('experiment', {}))
        )

class MIAEvaluator:
    """Evaluator for computing MIA metrics"""
    
    @staticmethod
    def compute_metrics(labels: List[int], scores: List[float], 
                       fpr_thresholds: List[float] = None) -> Dict[str, float]:
        """Compute evaluation metrics for MIA attack"""
        if fpr_thresholds is None:
            fpr_thresholds = [0.01, 0.05, 0.1]
        
        # Compute AUC
        auc = roc_auc_score(labels, scores)
        
        # Compute ROC curve
        fpr, tpr, thresholds = roc_curve(labels, scores)
        
        # Compute TPR at specified FPR thresholds
        results = {"auc": auc}
        
        for target_fpr in fpr_thresholds:
            # Find the index of the threshold that gives FPR closest to target_fpr
            idx = np.argmin(np.abs(fpr - target_fpr))
            target_tpr = tpr[idx]
            actual_fpr = fpr[idx]
            
            logger.info(f"Target FPR: {target_fpr}, Actual FPR: {actual_fpr:.4f}, TPR: {target_tpr:.4f}")
            results[f"tpr@{target_fpr}"] = target_tpr
        
        return results

class DataLoader:
    """Data loader for MIA experiments"""
    
    @staticmethod
    def load_json_data(file_path: str) -> List[Dict[str, Any]]:
        """Load data from JSON file"""
        with open(file_path, 'r') as f:
            return json.load(f)
    
    @staticmethod
    def load_hf_dataset(dataset_name: str, split: str = "train") -> Dataset:
        """Load dataset from Hugging Face"""
        return load_dataset(dataset_name, split=split)
    
    @staticmethod
    def prepare_target_examples(data: List[Dict[str, Any]], test_size: int, 
                               random_seed: int, data_format: str, 
                               enable_shuffle: bool = True) -> List[Dict[str, str]]:
        """Prepare member and non-member examples for evaluation"""
        import random
        
        if enable_shuffle:
            random.seed(random_seed)
            # Shuffle data
            random.shuffle(data)
        
        # Take the first test_size examples
        examples = data[:test_size]
        
        # Convert to standard format
        standardized = []
        for ex in examples:
            if data_format == "instruction":
                standardized.append({
                    "instruction": ex.get("instruction", ""),
                    "context": ex.get("input", ex.get("context", "")),
                    "response": ex.get("output", ex.get("response", ""))
                })
            elif data_format == "pretrain":
                if "text" in ex:
                    standardized.append({"text": ex["text"]})
            else:
                raise ValueError(f"Unsupported data_format: {data_format}")
        
        return standardized

class SimilarityBasedICP:
    """Similarity-based ICP-MIA implementation"""
    
    def __init__(self, config: SimilarityBasedICPConfig, cache_dir: str, data_format: str, device: str):
        self.config = config
        self.cache_dir = cache_dir
        self.data_format = data_format
        self.device = device
        self.encoder = None
        self.index = None
        self.prefix_pool = None
        
    def setup(self):
        """Setup encoder and index for similarity search"""
        logger.info("Setting up similarity-based ICP...")
        
        # Load sentence encoder
        self.encoder = SentenceTransformer(self.config.embedding_model, device=self.device)
        
        # Load prefix pool data
        if os.path.exists(self.config.prefix_pool_source):
            logger.info(f"Loading prefix pool from local file: {self.config.prefix_pool_source}...")
            data = DataLoader.load_json_data(self.config.prefix_pool_source)
            
            if self.data_format == "instruction":
                self.prefix_pool = Dataset.from_list([
                    {"instruction": ex.get("instruction", ""), 
                     "response": ex.get("output", ex.get("response", "")),
                     "context": ex.get("input", ex.get("context", ""))}
                    for ex in data
                ])
            elif self.data_format == "pretrain":
                self.prefix_pool = Dataset.from_list([
                    {"text": ex["text"]} for ex in data if "text" in ex
                ])
        else:
            logger.info(f"Loading prefix pool from Hugging Face: {self.config.prefix_pool_source}...")
            # Assume it's a Hugging Face dataset path
            hf_path = "databricks/databricks-dolly-15k" if self.config.prefix_pool_source == "dolly" else self.config.prefix_pool_source
            dolly_ds = load_dataset(hf_path, split="train")

            if self.data_format == "instruction":
                self.prefix_pool = Dataset.from_list([
                    {"instruction": ex["instruction"], "response": ex["response"], 
                     "context": ex.get("context", "")}
                    for ex in dolly_ds
                ])
            elif self.data_format == "pretrain":
                # Convert instruction format to pretrain format for dolly dataset
                self.prefix_pool = Dataset.from_list([
                    {"text": ex.get("instruction", "") + " " + ex.get("context", "") + " " + ex.get("response", "")}
                    for ex in dolly_ds
                ])

        # Create embeddings and FAISS index
        self._create_embeddings_and_index()
    
    def _create_embeddings_and_index(self):
        """Create embeddings and FAISS index for similarity search"""
        cache_name = f"{self.config.prefix_pool_source.replace('/', '_')}_embeddings"
        emb_file = os.path.join(self.cache_dir, f"{cache_name}.npy")
        index_file = os.path.join(self.cache_dir, f"{cache_name}.faiss")
        
        if os.path.exists(emb_file) and os.path.exists(index_file):
            logger.info("Loading cached embeddings and index...")
            self.embeddings = np.load(emb_file)
            self.index = faiss.read_index(index_file)
        else:
            logger.info("Creating new embeddings and index...")
            
            if self.data_format == "instruction":
                prefix_texts = [
                    self._format_demo_instruction(
                        ex["instruction"], 
                        "", 
                        ex.get("context", "")
                    )
                    for ex in self.prefix_pool
                ]
            elif self.data_format == "pretrain":
                prefix_texts = [ex["text"] for ex in self.prefix_pool]
            else:
                raise ValueError(f"Unsupported data_format: {self.data_format}")
            
            self.embeddings = self.encoder.encode(
                prefix_texts,
                batch_size=64,
                normalize_embeddings=True,
                show_progress_bar=True
            ).astype("float32")
            
            # Create FAISS index
            self.index = faiss.IndexFlatIP(self.embeddings.shape[1])
            self.index.add(self.embeddings)
            
            # Save to cache
            os.makedirs(self.cache_dir, exist_ok=True)
            np.save(emb_file, self.embeddings)
            faiss.write_index(self.index, index_file)
    
    def _format_demo_instruction(self, instruction: str, output: str, context: str = "") -> str:
        """Format demonstration text for instruction-tuned models"""
        if context.strip():
            user_msg = f"{instruction.strip()}\n{context.strip()}"
        else:
            user_msg = instruction.strip()
        
        return (
            f"<|start_header_id|>user<|end_header_id|>\n\n{user_msg}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n{output.strip()}<|eot_id|>"
        )
    
    def get_prefix_candidates(self, target_example: Dict[str, str]) -> List[str]:
        """Get prefix candidates using similarity search"""
        if self.data_format == "instruction":
            query_text = self._format_demo_instruction(
                target_example["instruction"], "", target_example.get("context", "")
            )
        elif self.data_format == "pretrain":
            query_text = target_example["text"]
        else:
            raise ValueError(f"Unsupported data_format: {self.data_format}")
        
        # Encode query
        query_vec = self.encoder.encode([query_text], normalize_embeddings=True, show_progress_bar=False).astype("float32")
        
        # Search for similar examples
        _, indices = self.index.search(query_vec, self.config.max_prefix_candidates)
        
        # Create prefix demonstrations
        candidates = []
        for idx in indices[0]:
            if self.data_format == "instruction":
                demo = self._format_demo_instruction(
                    self.prefix_pool[int(idx)]["instruction"],
                    self.prefix_pool[int(idx)]["response"],
                    self.prefix_pool[int(idx)].get("context", "")
                )
            elif self.data_format == "pretrain":
                demo = self.prefix_pool[int(idx)]["text"]
            else:
                raise ValueError(f"Unsupported data_format: {self.data_format}")
            candidates.append(demo)
        
        return candidates

class SelfPerturbationICP:
    """Self-perturbation ICP-MIA implementation"""
    
    def __init__(self, config: SelfPerturbationICPConfig, data_format: str):
        self.config = config
        self.data_format = data_format
        self.perturbation_data = None
        
    def setup(self):
        """Setup perturbation data"""
        logger.info("Setting up self-perturbation ICP...")
        
        if self.config.perturbation_file_path:
            with open(self.config.perturbation_file_path, 'r') as f:
                self.perturbation_data = json.load(f)
        else:
            raise ValueError("Perturbation file path must be provided for self-perturbation ICP")
    
    def get_prefix_candidates(self, target_example: Dict[str, str]) -> List[str]:
        """Get prefix candidates from perturbations"""
        # Find matching target example in perturbation data
        for item in self.perturbation_data:
            target = item.get("target_example", {})
            
            # Check for a match based on data format
            is_match = False
            if self.data_format == "instruction":
                if (target.get("instruction", "") == target_example.get("instruction", "") and
                    target.get("input", target.get("context", "")) == target_example.get("context", "")):
                    is_match = True
            elif self.data_format == "pretrain":
                # For pretrain, we expect 'text' field in both target_example and perturbation file
                if "text" in target and "text" in target_example and target["text"] == target_example["text"]:
                    is_match = True
            
            if is_match:
                perturbations = item.get(self.config.perturbation_key, [])
            
                if not perturbations:
                    return []
                
                # Flatten perturbations if they are grouped
                if isinstance(perturbations[0], str):
                    # Already flattened
                    all_perturbations = perturbations
                else:
                    # Need to flatten
                    all_perturbations = []
                    for group in perturbations:
                        if isinstance(group, str):
                            # Split by numbered entries
                            entries = group.split('\n\n')
                            for entry in entries:
                                if entry.strip():
                                    # Remove numbering
                                    cleaned = entry.strip()
                                    if cleaned and cleaned[0].isdigit() and '.' in cleaned[:3]:
                                        cleaned = '.'.join(cleaned.split('.')[1:]).strip()
                                    all_perturbations.append(cleaned)
                        else:
                            all_perturbations.extend(group)
                
                # Limit to top_k and format as demonstrations
                limited_perturbations = all_perturbations[:self.config.top_k]
                
                if self.data_format == "instruction":
                    candidates = []
                    for pert in limited_perturbations:
                        demo = self._format_demo(target_example["instruction"], pert.strip(), 
                                               target_example.get("context", ""))
                        candidates.append(demo)
                    return candidates
                elif self.data_format == "pretrain":
                    # For pretrain, the perturbations are the prefixes themselves
                    return [p.strip() for p in limited_perturbations]

        return []  # No matching example found
    
    def _format_demo(self, instruction: str, output: str, context: str = "") -> str:
        """Format demonstration text"""
        if context.strip():
            user_msg = f"{instruction.strip()}\n{context.strip()}"
        else:
            user_msg = instruction.strip()
        
        return (
            f"<|start_header_id|>user<|end_header_id|>\n\n{user_msg}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n{output.strip()}<|eot_id|>"
        )

class MIAAttacker:
    """Main MIA attacker class"""
    
    def __init__(self, config: MIAConfig):
        self.config = config
        self.tokenizer = None
        self.model = None
        self.ref_tokenizer = None
        self.ref_model = None
        
        # Attack components
        self.similarity_icp = None
        self.perturbation_icp = None
        
        # Data
        self.members = None
        self.nonmembers = None
        
        # Results
        self.results = []
        # Raw training data (for order-effect analysis)
        self.train_data_raw = None
        
    def setup(self):
        """Setup models, data, and attack components"""
        # Create directories
        os.makedirs(self.config.experiment.output_dir, exist_ok=True)
        os.makedirs(self.config.experiment.cache_dir, exist_ok=True)
        
        # Load models
        self._load_models()
        
        # Setup attack components first (needed for self-perturbation data loading)
        if self.config.similarity_based_icp.enabled:
            self.similarity_icp = SimilarityBasedICP(
                self.config.similarity_based_icp, 
                self.config.experiment.cache_dir,
                self.config.data.data_format,
                self.config.model.device
            )
            self.similarity_icp.setup()
        
        if self.config.self_perturbation_icp.enabled:
            self.perturbation_icp = SelfPerturbationICP(
                self.config.self_perturbation_icp, self.config.data.data_format
            )
            self.perturbation_icp.setup()
        
        # Load data (this depends on which attack is enabled)
        self._load_data()
    
    def _load_models(self):
        """Load target and reference models"""
        logger.info(f"Loading target model from {self.config.model.target_model_path}...")
        
        torch_dtype = getattr(torch, self.config.model.torch_dtype)
        
        # Check if this is a PEFT model by looking for adapter_config.json
        adapter_config_path = os.path.join(self.config.model.target_model_path, "adapter_config.json")
        
        if os.path.exists(adapter_config_path):
            # This is a PEFT model
            logger.info("Detected PEFT model, loading with PeftModel...")
            from peft import PeftModel, PeftConfig
            
            # Read the adapter config to get the base model path
            peft_config = PeftConfig.from_pretrained(self.config.model.target_model_path)
            base_model_path = peft_config.base_model_name_or_path
            
            # Load tokenizer from the PEFT model path first, fallback to base model
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.config.model.target_model_path)
            except:
                self.tokenizer = AutoTokenizer.from_pretrained(base_model_path)
            
            # Load base model
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_path,
                torch_dtype=torch_dtype,
                device_map=self.config.model.device
            )
            
            # Load PEFT model
            self.model = PeftModel.from_pretrained(
                base_model, 
                self.config.model.target_model_path,
                device_map=self.config.model.device
            ).eval()
        else:
            # Regular model loading
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model.target_model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model.target_model_path,
                torch_dtype=torch_dtype,
                device_map=self.config.model.device
            ).eval()
        
        if self.config.model.reference_model_path:
            logger.info(f"Loading reference model from {self.config.model.reference_model_path}...")
            self.ref_tokenizer = AutoTokenizer.from_pretrained(self.config.model.reference_model_path)
            self.ref_model = AutoModelForCausalLM.from_pretrained(
                self.config.model.reference_model_path,
                torch_dtype=torch_dtype,
                device_map=self.config.model.device
            ).eval()
    
    def _load_data(self):
        """Load training and test data"""
        logger.info("Loading datasets...")
        
        if self.config.self_perturbation_icp.enabled:
            # For self-perturbation, use the samples from perturbation file
            self._load_perturbation_data()
        else:
            # For similarity-based ICP, use train/test data split
            self._load_similarity_data()
    
    def _load_similarity_data(self):
        """Load data for similarity-based ICP attack from local JSON or Hugging Face."""
        # --- Load train data (members) ---
        train_path = self.config.data.train_data_path
        if train_path.endswith('.json'):
            logger.info(f"Loading training data from local file: {train_path}")
            train_data = DataLoader.load_json_data(train_path)
        else:
            logger.info(f"Loading training data from Hugging Face: {train_path}")
            train_dataset = DataLoader.load_hf_dataset(train_path, split="train")
            train_data = list(train_dataset)
        
        self.train_data_raw = train_data  # preserve full order for order-effect study
        self.members = DataLoader.prepare_target_examples(
            train_data, self.config.data.test_size, self.config.data.random_seed, self.config.data.data_format, self.config.data.enable_shuffle
        )
        
        # --- Load test data (non-members) ---
        test_path = self.config.data.test_data_path
        if test_path.endswith('.json'):
            logger.info(f"Loading test data from local file: {test_path}")
            test_data = DataLoader.load_json_data(test_path)
        else:
            logger.info(f"Loading test data from Hugging Face: {test_path}")
            test_dataset = DataLoader.load_hf_dataset(test_path, split="test")
            test_data = list(test_dataset)

        self.nonmembers = DataLoader.prepare_target_examples(
            test_data, self.config.data.test_size, self.config.data.random_seed, self.config.data.data_format, enable_shuffle=False
        )
        
        logger.info(f"Loaded {len(self.members)} members and {len(self.nonmembers)} non-members")
    
    def _load_perturbation_data(self):
        """Load data for self-perturbation ICP attack from perturbation file"""
        logger.info("Loading perturbation data for self-perturbation ICP...")
        
        if not self.perturbation_icp or not self.perturbation_icp.perturbation_data:
            raise ValueError("Perturbation ICP must be setup before loading perturbation data")
        
        # Extract target examples from perturbation data
        members = []
        nonmembers = []
        all_examples = []  # For force_balanced_split fallback
        has_labels = any("label" in item for item in self.perturbation_icp.perturbation_data)
        
        if has_labels:
            logger.info("Using 'label' field to identify members (label=1) and non-members (label=0)")
        
        for item in self.perturbation_icp.perturbation_data:
            target = item.get("target_example", {})
            
            # Convert to standard format
            example = None
            if self.config.data.data_format == "instruction":
                example = {
                    "instruction": target.get("instruction", ""),
                    "context": target.get("input", target.get("context", "")),
                    "response": target.get("output", target.get("response", ""))
                }
            elif self.config.data.data_format == "pretrain":
                if "text" in target:
                    example = {"text": target["text"]}
            
            if not example:
                continue

            all_examples.append(example)  # Store for potential fallback
            
            # Determine if this is a member or non-member
            # Priority 1: Use label field if available
            if "label" in item:
                is_member = (item["label"] == 1)
            else:
                # Priority 2: Use heuristic detection
                # Set source file information for judgement
                self._current_example_source_file = item.get("source_file", "")
                
                # Check if this example exists in training data
                is_member = self._is_training_example(example)
                
                # Clear the source file information after use
                self._current_example_source_file = None
            
            if is_member:
                members.append(example)
            else:
                nonmembers.append(example)
        
        # Handle the case where no members are found
        force_balanced = getattr(self.config.data, 'force_balanced_split', False)
        if len(members) == 0 and (force_balanced or len(nonmembers) > 0):
            logger.warning(f"No members found using strategy '{getattr(self.config.data, 'member_detection_strategy', 'auto')}'")
            
            if force_balanced:
                logger.info("Applying force_balanced_split: creating balanced member/nonmember split")
                # Shuffle and split the examples into two equal groups
                import random
                random.seed(self.config.data.random_seed)
                random.shuffle(all_examples)
                
                split_point = len(all_examples) // 2
                members = all_examples[:split_point]
                nonmembers = all_examples[split_point:]
                
                logger.info(f"Force balanced split: {len(members)} members, {len(nonmembers)} nonmembers")
            else:
                # Try alternative strategies if auto was used
                current_strategy = getattr(self.config.data, 'member_detection_strategy', 'auto')
                if current_strategy == 'auto':
                    logger.info("Attempting alternative detection strategies...")
                    
                    # Try relaxed matching
                    self.config.data.member_detection_strategy = 'relaxed_match'
                    members_relaxed = []
                    nonmembers_relaxed = []
                    
                    for i, item in enumerate(self.perturbation_icp.perturbation_data):
                        example = all_examples[i] if i < len(all_examples) else None
                        if not example:
                            continue
                        
                        self._current_example_source_file = item.get("source_file", "")
                        is_member = self._is_training_example(example)
                        self._current_example_source_file = None
                        
                        if is_member:
                            members_relaxed.append(example)
                        else:
                            nonmembers_relaxed.append(example)
                    
                    # Restore original strategy
                    self.config.data.member_detection_strategy = current_strategy
                    
                    if len(members_relaxed) > 0:
                        logger.info(f"Relaxed matching found {len(members_relaxed)} members")
                        members = members_relaxed
                        nonmembers = nonmembers_relaxed
                    else:
                        logger.warning("No members found even with relaxed matching. All examples treated as nonmembers.")
        
        self.members = members
        self.nonmembers = nonmembers
        
        logger.info(f"Loaded {len(self.members)} members and {len(self.nonmembers)} non-members from perturbation file")
        
        # Log additional statistics for debugging
        if len(self.members) == 0:
            logger.warning("WARNING: No members found! This may indicate a data matching issue.")
            logger.info(f"Consider setting 'force_balanced_split: true' in config or checking data paths.")
        
        # For perturbation attacks, we don't have train_data_raw for order effect
        self.train_data_raw = []
    
    def _is_training_example(self, example: Dict[str, str]) -> bool:
        """Check if an example exists in the training data with configurable strategy"""
        strategy = getattr(self.config.data, 'member_detection_strategy', 'auto')
        
        # Strategy 1: Source file-based detection (fastest, most reliable when available)
        if strategy in ["source_file", "auto"] and hasattr(self, '_current_example_source_file') and self._current_example_source_file:
            train_path = self.config.data.train_data_path
            # Normalize paths for comparison
            import os
            train_path_normalized = os.path.normpath(train_path)
            source_path_normalized = os.path.normpath(self._current_example_source_file)
            
            # Multiple path matching strategies
            is_train_file = (
                train_path_normalized == source_path_normalized or 
                'train' in source_path_normalized.lower() or
                os.path.basename(train_path_normalized) == os.path.basename(source_path_normalized) or
                train_path_normalized.endswith(source_path_normalized) or
                source_path_normalized.endswith(train_path_normalized)
            )
            
            logger.debug(f"Source file check: {source_path_normalized} vs {train_path_normalized} = {is_train_file}")
            if strategy == "source_file":
                return is_train_file
            elif strategy == "auto" and is_train_file:
                return True
        
        # Strategy 2 & 3: Content-based matching (fallback or explicit choice)
        if strategy in ["content_match", "relaxed_match", "auto"]:
            # Load training data if not already loaded
            if not hasattr(self, '_train_data_for_lookup'):
                train_path = self.config.data.train_data_path
                if train_path.endswith('.json'):
                    train_data = DataLoader.load_json_data(train_path)
                else:
                    train_dataset = DataLoader.load_hf_dataset(train_path, split="train")
                    train_data = list(train_dataset)
                self._train_data_for_lookup = train_data
                logger.info(f"Loaded {len(train_data)} training examples for content-based lookup")
            
            # Enhanced content matching with multiple field name combinations
            for train_ex in self._train_data_for_lookup:
                if self.config.data.data_format == "instruction":
                    # Support multiple field name variations
                    train_instruction = train_ex.get("instruction", "")
                    train_input = train_ex.get("input", train_ex.get("context", ""))
                    train_output = train_ex.get("output", train_ex.get("response", ""))
                    
                    example_instruction = example.get("instruction", "")
                    example_input = example.get("context", example.get("input", ""))
                    example_output = example.get("response", example.get("output", ""))
                    
                    # Exact match (for content_match strategy or auto)
                    if strategy in ["content_match", "auto"] and (
                        train_instruction == example_instruction and
                        train_input == example_input and
                        train_output == example_output):
                        return True
                    
                    # Relaxed match (only instruction and output, ignoring input/context differences)
                    if strategy in ["relaxed_match", "auto"] and (
                        train_instruction == example_instruction and
                        train_output == example_output and
                        train_instruction.strip() and train_output.strip()):
                        logger.debug(f"Found relaxed match for instruction: {train_instruction[:50]}...")
                        return True
                        
                elif self.config.data.data_format == "pretrain":
                    if train_ex.get("text", "") == example.get("text", ""):
                        return True
        
        return False
    
    def _build_query_prompt(self, instruction: str, context: str = "") -> str:
        """Build query prompt for likelihood computation"""
        if context.strip():
            user_msg = f"{instruction.strip()}\n{context.strip()}"
        else:
            user_msg = instruction.strip()
        
        return (
            f"<|start_header_id|>user<|end_header_id|>\n\n{user_msg}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n"
        )
    
    def _truncate_prompt(self, prompt: str) -> str:
        """Truncate prompt if it exceeds max tokens"""
        tokens = self.tokenizer.encode(prompt)
        
        if len(tokens) > self.config.model.max_prompt_tokens:
            tokens = tokens[-self.config.model.max_prompt_tokens:]
            truncated_prompt = self.tokenizer.decode(tokens)
            logger.warning(f"Prompt truncated from {len(tokens)} to {self.config.model.max_prompt_tokens} tokens")
            return truncated_prompt
        
        return prompt
    
    def _compute_nll_loss(self, prompt: str, answer: str, model, tokenizer) -> float:
        """Compute negative log-likelihood loss"""
        full_text = prompt + answer + "<|eot_id|>"
        full_text = self._truncate_prompt(full_text)
        
        # Tokenize
        enc = tokenizer(full_text, return_tensors="pt", truncation=True, 
                       max_length=self.config.model.max_prompt_tokens)
        input_ids = enc.input_ids.to(self.config.model.device)
        labels = input_ids.clone()
        
        # Mask prompt tokens
        prompt_tokens = tokenizer.encode(self._truncate_prompt(prompt))
        if len(prompt_tokens) < len(input_ids[0]):
            labels[:, :len(prompt_tokens)] = -100
        
        # Compute loss
        with torch.no_grad():
            return model(input_ids, labels=labels).loss.item()
    
    def _compute_icp_score(self, prefix_candidates: List[str], target_example: Dict[str, str], 
                          aggregation_strategy: str) -> Tuple[float, float]:
        """Compute ICP score and base log-likelihood, dispatching by data format."""
        if self.config.data.data_format == "instruction":
            return self._compute_icp_score_instruction(prefix_candidates, target_example, aggregation_strategy)
        elif self.config.data.data_format == "pretrain":
            return self._compute_icp_score_pretrain(prefix_candidates, target_example, aggregation_strategy)
        else:
            raise ValueError(f"Unsupported data_format: {self.config.data.data_format}")
    
    def _compute_icp_score_instruction(self, prefix_candidates: List[str], target_example: Dict[str, str], 
                                     aggregation_strategy: str) -> Tuple[float, float]:
        """Compute ICP score for instruction-tuned models - EXACTLY matching old script logic"""
        query_prompt = self._build_query_prompt(
            target_example["instruction"], target_example.get("context", "")
        )
        
        # Compute base likelihood (negative NLL)
        base_ll = -self._compute_nll_loss(
            query_prompt, target_example["response"], self.model, self.tokenizer
        )
        
        if not prefix_candidates:
            return 0.0, base_ll

        ranking_scores = []
        for prefix in prefix_candidates:
            ranking_score = -self._compute_nll_loss(
                prefix + query_prompt, target_example["response"], 
                self.model, self.tokenizer
            )
            ranking_scores.append(ranking_score)

        if aggregation_strategy == "max":
            order = np.argsort(ranking_scores)[::-1]
        elif aggregation_strategy == "min":
            order = np.argsort(ranking_scores) 
        elif aggregation_strategy == "mean" or aggregation_strategy == "median":
            order = np.argsort(ranking_scores)[::-1]
        else:
            raise ValueError(f"Unknown aggregation strategy: {aggregation_strategy}")

        prefix_block = ""
        
        # Reserve tokens for query prompt and response 
        reserved_tokens = len(self.tokenizer.encode(query_prompt)) + 300  # 300 tokens for response
        available_tokens = self.config.model.max_prompt_tokens - reserved_tokens
        
        selected_prefix = ""
        for i in order:  
            candidate_demo = prefix_candidates[i]
            candidate_block = prefix_block + candidate_demo
            

            if len(self.tokenizer.encode(candidate_block)) <= available_tokens:
                selected_prefix = candidate_demo
                break  
        
        if not selected_prefix:
            return 0.0, base_ll
        
        joint_ll = -self._compute_nll_loss(
            selected_prefix + query_prompt, target_example["response"], 
            self.model, self.tokenizer
        )
        
        icp_score = base_ll - joint_ll
        
        return icp_score, base_ll
    
    def _compute_icp_score_pretrain(self, prefix_candidates: List[str], target_example: Dict[str, str],
                                    aggregation_strategy: str) -> Tuple[float, float]:
        """Compute ICP score for pre-training format models."""
        target_text = target_example["text"]

        # Compute base likelihood (negative NLL of the full text)
        base_ll = -self._compute_nll_loss("", target_text, self.model, self.tokenizer)

        if not prefix_candidates:
            return 0.0, base_ll

        # Compute ranking scores (conditional log-likelihood of text given prefix)
        ranking_scores = [-self._compute_nll_loss(p, target_text, self.model, self.tokenizer) for p in prefix_candidates]

        # Apply selection strategy
        if aggregation_strategy == "max":
            order = np.argsort(ranking_scores)[::-1]
        elif aggregation_strategy == "min":
            order = np.argsort(ranking_scores)
        else:  # For mean/median, default to max for single prefix selection
            order = np.argsort(ranking_scores)[::-1]

        # Token-length filtering
        reserved_tokens = len(self.tokenizer.encode(target_text))
        available_tokens = self.config.model.max_prompt_tokens - reserved_tokens

        selected_prefix = ""
        for i in order:
            candidate_prefix = prefix_candidates[i]
            if len(self.tokenizer.encode(candidate_prefix)) <= available_tokens:
                selected_prefix = candidate_prefix
                break
        
        if not selected_prefix:
            return 0.0, base_ll

        # Compute final joint likelihood with the best valid prefix
        joint_ll = -self._compute_nll_loss(selected_prefix, target_text, self.model, self.tokenizer)

        # Final ICP score (maintaining original formula)
        icp_score = base_ll - joint_ll
        
        return icp_score, base_ll
    
    def run_similarity_based_icp(self) -> Tuple[List[int], List[float]]:
        """Run similarity-based ICP-MIA attack."""
        logger.info("Running similarity-based ICP-MIA...")
        
        labels = []
        icp_scores = []
        
        for example in tqdm(self.members, desc="Processing members"):
            candidates = self.similarity_icp.get_prefix_candidates(example)
            icp_score, _ = self._compute_icp_score(
                candidates, example, self.config.similarity_based_icp.aggregation_strategy
            )
            labels.append(1)
            icp_scores.append(icp_score)
        
        for example in tqdm(self.nonmembers, desc="Processing non-members"):
            candidates = self.similarity_icp.get_prefix_candidates(example)
            icp_score, _ = self._compute_icp_score(
                candidates, example, self.config.similarity_based_icp.aggregation_strategy
            )
            labels.append(0)
            icp_scores.append(icp_score)
        
        return labels, icp_scores
    
    def run_self_perturbation_icp(self) -> Tuple[List[int], List[float]]:
        """Run self-perturbation ICP-MIA attack."""
        logger.info("Running self-perturbation ICP-MIA...")
        
        labels = []
        icp_scores = []
        
        for example in tqdm(self.members, desc="Processing members"):
            candidates = self.perturbation_icp.get_prefix_candidates(example)
            icp_score, _ = self._compute_icp_score(
                candidates, example, self.config.self_perturbation_icp.aggregation_strategy
            )
            labels.append(1)
            icp_scores.append(icp_score)
        
        for example in tqdm(self.nonmembers, desc="Processing non-members"):
            candidates = self.perturbation_icp.get_prefix_candidates(example)
            icp_score, _ = self._compute_icp_score(
                candidates, example, self.config.self_perturbation_icp.aggregation_strategy
            )
            labels.append(0)
            icp_scores.append(icp_score)
        
        return labels, icp_scores
    
    def _run_order_effect_analysis(self):
        """Evaluate the impact of training example order on AUC."""
        if self.config.self_perturbation_icp.enabled:
            logger.warning("Order-effect analysis is not supported for self-perturbation ICP attacks because member/non-member samples are fixed by the perturbation file. Skipping...")
            return
            
        if not (self.config.similarity_based_icp.enabled):
            logger.warning("Order-effect analysis currently supports similarity-based ICP only. Skipping…")
            return
        logger.info("Running training order-effect analysis…")
        
        partitions = self.config.experiment.order_effect_partitions
        sample_size = self.config.experiment.order_effect_sample_size
        random_seed = self.config.data.random_seed
        partition_size = len(self.train_data_raw) // partitions
        
        order_results_icp = []
        auc_values_icp = []
        
        for idx in range(partitions):
            start = idx * partition_size
            end = (idx + 1) * partition_size if idx < partitions - 1 else len(self.train_data_raw)
            partition_slice = self.train_data_raw[start:end]
            if len(partition_slice) == 0:
                logger.warning(f"Partition {idx} is empty – skipping.")
                continue
            import random as _random
            rng = _random.Random(random_seed)
            sampled_slice = rng.sample(partition_slice, min(sample_size, len(partition_slice)))
            self.members = DataLoader.prepare_target_examples(
                sampled_slice, len(sampled_slice), random_seed, self.config.data.data_format, enable_shuffle=False
            )
            
            labels, icp_scores = self.run_similarity_based_icp()

            metrics_icp = MIAEvaluator.compute_metrics(labels, icp_scores, self.config.experiment.fpr_thresholds)
            auc_icp = metrics_icp.get("auc", 0.0)
            auc_values_icp.append(auc_icp)
            order_results_icp.append({"partition_index": idx, **metrics_icp})

            logger.info(f"Partition {idx}: ICP AUC={auc_icp:.4f}")
        
        results_file_icp = os.path.join(
            self.config.experiment.output_dir,
            f"{self.config.experiment.experiment_name}_order_effect_icp_results.csv"
        )
        pd.DataFrame(order_results_icp).to_csv(results_file_icp, index=False)
        logger.info(f"Order-effect ICP metrics saved to: {results_file_icp}")
        
        plt.figure()
        plt.plot(range(len(auc_values_icp)), auc_values_icp, marker='o', label="ICP Attack")
        plt.xlabel('Partition Index')
        plt.ylabel('AUC')
        plt.title('Training Order Effect on AUC')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plot_path = os.path.join(self.config.experiment.output_dir, self.config.experiment.order_effect_plot_file)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Order-effect plot saved to: {plot_path}")
    
    def _save_results(self, attack_type: str, metrics: Dict[str, float]):
        """Save results to CSV and log"""
        # Extract model name from path
        def get_model_name(path: str) -> str:
            if not path:
                return "None"
            parts = path.split('/')
            if len(parts) >= 4:
                return '/'.join(parts[-4:])
            return path
        
        result_row = {
            'timestamp': datetime.now().isoformat(),
            'experiment_name': self.config.experiment.experiment_name,
            'attack_type': attack_type,
            'target_model': get_model_name(self.config.model.target_model_path),
            'reference_model': get_model_name(self.config.model.reference_model_path),
            'dataset': f"train:{os.path.basename(self.config.data.train_data_path)}, test:{os.path.basename(self.config.data.test_data_path)}",
            'test_size': self.config.data.test_size,
            **metrics
        }
        
        # Add configuration details
        if attack_type == "Similarity-based ICP-MIA":
            result_row.update({
                'prefix_pool': self.config.similarity_based_icp.prefix_pool_source,
                'top_k': self.config.similarity_based_icp.top_k,
                'aggregation_strategy': self.config.similarity_based_icp.aggregation_strategy,
                'max_prefix_candidates': self.config.similarity_based_icp.max_prefix_candidates
            })
        elif attack_type == "Self-perturbation ICP-MIA":
            result_row.update({
                'perturbation_file': os.path.basename(self.config.self_perturbation_icp.perturbation_file_path or ""),
                'perturbation_key': self.config.self_perturbation_icp.perturbation_key,
                'top_k': self.config.self_perturbation_icp.top_k,
                'aggregation_strategy': self.config.self_perturbation_icp.aggregation_strategy
            })
        
        self.results.append(result_row)
        
        # Save to CSV
        results_file = os.path.join(
            self.config.experiment.output_dir, 
            f"{self.config.experiment.experiment_name}_results.csv"
        )
        
        df = pd.DataFrame(self.results)
        df.to_csv(results_file, index=False)
        
        # Log results
        logger.info(f"{attack_type} Results:")
        for key, value in metrics.items():
            logger.info(f"  {key}: {value:.4f}")
        
        logger.info(f"Results saved to: {results_file}")
    
    def _save_detailed_results(self, attack_type: str, labels: List[int], 
                              icp_scores: List[float]):
        """Save detailed results including raw scores for visualization"""
        if not self.config.experiment.save_detailed_results:
            return
            
        detailed_data = {
            'labels': labels,
            'icp_scores': icp_scores,
            'attack_type': attack_type,
            'experiment_name': self.config.experiment.experiment_name,
            'config': {
                'target_model': self.config.model.target_model_path,
                'test_size': self.config.data.test_size,
                'data_format': self.config.data.data_format
            }
        }
        
        detailed_file = os.path.join(
            self.config.experiment.output_dir,
            f"{self.config.experiment.experiment_name}_{attack_type.replace(' ', '_').replace('-', '_').lower()}_detailed_scores.json"
        )
        
        with open(detailed_file, 'w') as f:
            json.dump(detailed_data, f, indent=2)
        
        logger.info(f"Detailed scores saved to: {detailed_file}")

    def run_experiments(self):
        """Run all enabled MIA experiments"""
        logger.info(f"Starting MIA experiments: {self.config.experiment.experiment_name}")
        
        if self.config.similarity_based_icp.enabled:
            labels, icp_scores = self.run_similarity_based_icp()
            
            icp_metrics = MIAEvaluator.compute_metrics(
                labels, icp_scores, self.config.experiment.fpr_thresholds
            )
            self._save_results("Similarity-based ICP-MIA", icp_metrics)
            self._save_detailed_results("Similarity-based ICP-MIA", labels, icp_scores)
        
        if self.config.self_perturbation_icp.enabled:
            labels, icp_scores = self.run_self_perturbation_icp()

            icp_metrics = MIAEvaluator.compute_metrics(
                labels, icp_scores, self.config.experiment.fpr_thresholds
            )
            self._save_results("Self-perturbation ICP-MIA", icp_metrics)
            self._save_detailed_results("Self-perturbation ICP-MIA", labels, icp_scores)
        
        if self.config.experiment.order_effect_enabled:
            self._run_order_effect_analysis()
        
        logger.info("All experiments completed!")

def main():
    parser = argparse.ArgumentParser(description="Configurable MIA Attack Framework")
    parser.add_argument("--config", type=str, required=True, 
                       help="Path to YAML configuration file")
    
    args = parser.parse_args()
    
    # Load configuration
    config = MIAConfig.from_yaml(args.config)
    
    # Create and run attacker
    attacker = MIAAttacker(config)
    attacker.setup()
    attacker.run_experiments()

if __name__ == "__main__":
    main() 