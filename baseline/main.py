import argparse
import logging
import pickle
from collections import defaultdict

import os
import numpy as np
import torch
from attacks.utils import batch_nlloss
from datasets import Dataset, load_dataset
from sklearn.metrics import auc, roc_curve
from tabulate import tabulate
from transformers import AutoModelForCausalLM, AutoTokenizer

import matplotlib.pyplot as plt

from utils import (get_available_attacks, load_attack, load_config,
                   load_mimir_dataset, set_seed, load_heathcaremagic_dataset,load_MedInstruct_dataset,load_CNNDM_dataset,load_ag_news_dataset)

import seaborn as sns

logging.basicConfig(level=logging.INFO)


def init_model(model_name, device):
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left" 
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def get_header(config):
    header = ["MIA", "AUC"]
    for t in config["fpr_thresholds"]:
        header.append(f"TPR@FPR={t}")
    return header


def get_printable_ds_name(ds_info):
    if "tofu" in ds_info:
        name = ds_info["tofu"]
    elif "mimir_name" in ds_info:
        name = ds_info["mimir_name"]
    elif "chatdoctor" in ds_info:
        name = ds_info["chatdoctor"]
    elif "healthcaremagic" in ds_info:
        name = ds_info["healthcaremagic"]
    elif "MedInstruct" in ds_info:
        name = ds_info["MedInstruct"]
    elif "CNNDM" in ds_info:
        name = ds_info["CNNDM"]
    elif "ag_news" in ds_info:
        name = ds_info["ag_news"]
    else:
        raise ValueError()
    name = f"{name}/{ds_info['split']}"
    name = name.replace("/", "_")
    return name


def results_without_bootstrapping(y_true, y_pred, fpr_thresholds):
    
    if np.isnan(y_pred).any():
        print("[ERROR] y_pred contains NaN values!")
        nan_indices = np.where(np.isnan(y_pred))[0]
        print(f"NaN indices: {nan_indices}")
        print(f"Corresponding y_true: {np.array(y_true)[nan_indices]}")
        raise ValueError("Aborting due to NaN in prediction scores.")

    fpr_all, tpr_all, thresholds_all = roc_curve(y_true, y_pred)
    auc_score = auc(fpr_all, tpr_all)  

    tprs = {}
    members_idx = {}
    correctly_identified_idx = {}
    
    for t in fpr_thresholds:
        closest_idx = np.argmin(np.abs(fpr_all - t))  
        tpr_value = tpr_all[closest_idx]  
        threshold_value = thresholds_all[closest_idx]  

        tprs[t] = tpr_value

        members_mask = (y_pred >= threshold_value)
        members_idx[t] = np.where(members_mask)[0]

        correct_mask = members_mask & (np.array(y_true) == 1)
        
        correctly_identified_idx[t] = np.where(correct_mask)[0]

    results = [f"AUC: {auc_score:.4f}"] + \
              [f"FPR {t}: TPR {tprs[t]:.4f}" for t in fpr_thresholds]
    
    return results, members_idx, correctly_identified_idx
def init_dataset(ds_info, model, tokenizer, device, batch_size):

    if "mimir_name" in ds_info:
        if "name" in ds_info:
            raise ValueError("Cannot specify both 'name' and 'mimir_name' in dataset config")
        dataset = load_mimir_dataset(name=ds_info["mimir_name"], split=ds_info["split"])
        
    elif "healthcaremagic" in ds_info:
        dataset = load_heathcaremagic_dataset(name = ds_info['healthcaremagic'],split="")
    
    elif "MedInstruct" in ds_info:
        dataset = load_MedInstruct_dataset(name=ds_info["MedInstruct"], split=ds_info["split"])
    
    elif "CNNDM" in ds_info:
        dataset = load_CNNDM_dataset(name=ds_info["CNNDM"], split=ds_info["split"])

    elif "ag_news" in ds_info:
        dataset = load_ag_news_dataset(name=ds_info["ag_news"], split=ds_info["split"])
    else:
        raise ValueError("Dataset name is missing")

    # dataset = dataset.map(
    #     lambda x: batch_nlloss(x, model, tokenizer, device),
    #     batched=True,
    #     batch_size=batch_size
    # )
    return dataset
   
def plot_and_save_histogram(data, output_path, bins=50):

    plt.figure(figsize=(10, 6))
    
    sns.histplot(
    data=data,
    bins=bins,
    color='skyblue',
    edgecolor='black',
    alpha=0.7,
    kde=True  
    )
    
    plt.title('Histogram of Data', fontsize=14)
    plt.xlabel('ID', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(output_path, dpi=300)
    plt.close()  


            
from matplotlib_venn import venn2, venn3
def plot_all_venn_diagrams(results_to_save, output_dir=None):
    """
    Automatically iterates over all dataset names and FPR thresholds in results_to_save,
    generates and displays traditional Venn diagrams (only supports 2 or 3 attack methods).
    If output_dir is provided, the plots are saved as PNG files with filenames in the format:
    venn_{ds_name}_FPR_{fpr_threshold}.png.
    
    Parameters:
        results_to_save (dict): Dictionary containing results including correctly identified member IDs.
        output_dir (str): Directory to save the plots. If None, the plots are not saved.
    """
    # Create output directory if saving is enabled.
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    for ds_name, ds_data in results_to_save.items():
        identified = ds_data.get('identified_memberID', {})
        attack_names = list(identified.keys())

        if not (2 <= len(attack_names) <= 3):
            print(f"[Skip] {ds_name}: Venn diagram only supports 2 or 3 attack methods. Found {len(attack_names)}.")
            continue

        # Extract FPR thresholds from one of the attack methods.
        example_attack = attack_names[0]
        fpr_thresholds = identified[example_attack].keys()

        for fpr_threshold in fpr_thresholds:
            sets = [set(identified[attack].get(fpr_threshold, [])) for attack in attack_names]
            if all(len(s) == 0 for s in sets):
                print(f"[Skip] {ds_name}, FPR={fpr_threshold}: All attack results are empty.")
                continue

            # For any empty set, add a dummy value to avoid errors in plotting.
            sets = [s if len(s) > 0 else set([-1 * (i + 1)]) for i, s in enumerate(sets)]
            
            plt.figure(figsize=(6, 5))
            if len(sets) == 2:
                venn2(subsets=sets, set_labels=attack_names)
            else:
                venn3(subsets=sets, set_labels=attack_names)
            plt.title(f"Venn Diagram - {ds_name}, FPR={fpr_threshold}")
            plt.tight_layout()
            
            if output_dir:
                filename = os.path.join(output_dir, f"venn_{ds_name}_FPR_{fpr_threshold}.png")
                plt.savefig(filename)
                print(f"Plot saved: {filename}")
            
            plt.show()

def taskwise_analysis(dataset, attack_name, y_true, fpr_thresholds):
    from sklearn.metrics import roc_curve, roc_auc_score
    from collections import defaultdict

    task_to_scores = defaultdict(list)
    task_to_labels = defaultdict(list)

    for ex, label in zip(dataset, y_true):
        task = ex.get("category", None)
        score = ex.get(attack_name, None)
        if task is not None and score is not None:
            task_to_scores[task].append(score)
            task_to_labels[task].append(label)

    print(f"\n========= [Task-wise Analysis for Attack: {attack_name}] =========")
    for task in sorted(task_to_scores.keys()):
        scores = task_to_scores[task]
        labels = task_to_labels[task]
        if len(set(labels)) < 2:
            continue

        fpr, tpr, _ = roc_curve(labels, scores)
        auc = roc_auc_score(labels, scores)

        print(f"[Task: {task}]")
        print(f"  AUC: {auc:.4f}")
        for fpr_thres in fpr_thresholds:
            tpr_at_fpr = max([t for f, t in zip(fpr, tpr) if f <= fpr_thres], default=0.0)
            print(f"  TPR@FPR={fpr_thres:.2f}: {tpr_at_fpr:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run attacks')
    parser.add_argument('-c', '--config', type=str, help='Config path', default='config_healthcaremagic.yaml')
    # 'bag_of_words','loss','minkplusplus','minkprob','zlib'
    parser.add_argument('--attacks', nargs='*', type=str, help='Attacks to run.',default=["loss","minkprob","minkplusplus","reference","zlib"])
    parser.add_argument('--run-all', action='store_true', help='Run all available attacks')
    parser.add_argument('--seed', type=int, help='Random seed', default=None)
    parser.add_argument('--output', type=str, help="File to store attack results", default='TGneighbor_results.pkl')
    parser.add_argument("--target_model", help="Target model name")

    args = parser.parse_args()
    if args.seed is not None:
        set_seed(args.seed)
        
    config = load_config(args.config)
    
    if args.target_model:
        config['global']['target_model'] = args.target_model
    
    logging.debug(config)
    global_config = config['global']
    device = global_config["device"]
    if args.run_all:
        attacks = get_available_attacks(config)
    else:
        attacks = args.attacks

    model, tokenizer = init_model(global_config['target_model'], device)

    results_to_save = defaultdict(dict)
    results_to_print = {}
    
    for ds_info in global_config['datasets']:
        dataset = init_dataset(
            ds_info=ds_info,
            model=model,
            tokenizer=tokenizer,
            device=device,
            batch_size=global_config["batch_size"]
        )
        
        dataset = dataset.filter(lambda x: x["label"] in [0, 1])
        ds_name = get_printable_ds_name(ds_info)

        results = []
        header = get_header(global_config)
        
        y_true = [x["label"] for x in dataset]
        results_to_save[ds_name]["label"] = y_true
        
        for attack_name in sorted(attacks):
            logging.info(f"Running {attack_name} on {ds_name}")

            attack = load_attack(attack_name, model, tokenizer, config[attack_name])
            dataset = attack.run(dataset)
            
            y = []
            
            for i, x in enumerate(dataset):
                score = x.get(attack_name, None)
                if score is None:
                    print(f"[WARNING] No score for sample {i}, attack={attack_name}")
                    y.append(np.nan)
                    continue

                if np.isnan(score) or np.isinf(score):
                    print(f"\n[NaN Detected] Sample index: {i}")
                    print("Instruction:", x.get("instruction", ""))
                    print("Input:", x.get("input", ""))
                    print("Output:", x.get("output", ""))
                    print("Score:", score)
                    print("===")
                y.append(score)
                
            results_to_save[ds_name][attack_name] = y

            attack_results, members_idx, correctly_identified_idx = results_without_bootstrapping(y_true, y, fpr_thresholds=global_config["fpr_thresholds"])
            
            if "dolly" in ds_name.lower():
                has_category = all("category" in ex for ex in dataset)
                if has_category:
                    taskwise_analysis(dataset, attack_name, y_true, global_config["fpr_thresholds"])
            
            if 'identified_memberID' not in results_to_save[ds_name]:
                results_to_save[ds_name]['identified_memberID'] = {}
    
            results_to_save[ds_name]['identified_memberID'][attack_name] = correctly_identified_idx

            results.append([attack_name] + attack_results)
            # plot_and_save_histogram(correctly_identified_idx,output_path = 'id_hsitogram_Pythia-6.9b.png')
            logging.info(f"AUC {attack_name} on {ds_name}: {attack_results[0]}")
         
        
        results_to_print[ds_name] = tabulate(results, headers=header, tablefmt="outline")


    
    for ds_name, res in results_to_print.items():
        print(f"Dataset: {ds_name}")
        print(f"target_model:{global_config['target_model']}")
        print(res)
        print()
    
    if args.output is not None:
        with open(args.output, 'wb') as f:
            pickle.dump(results_to_save, f)

    