import json
import random
import re
import argparse
from tqdm import tqdm

def mask_words_with_placeholder(text, mask_rate=0.1, placeholder="word"):
    """
    Replace words with placeholder based on mask rate.
    """
    words = text.split()
    word_indices = [i for i, word in enumerate(words) if re.search(r'[a-zA-Z]', word)]
    
    num_to_mask = int(len(word_indices) * mask_rate)
    
    if num_to_mask == 0:
        return text

    mask_indices = sorted(random.sample(word_indices, num_to_mask))
    
    masked_words = words.copy()
    for idx in mask_indices:
        masked_words[idx] = placeholder
    
    return ' '.join(masked_words)

def sample_and_merge_jsonl_files(input_files, samples_per_file, output_file, mask_rate, num_perturbations=5):
    """
    Sample from multiple JSONL files and merge.
    """
    merged_data = []
    
    for input_file in input_files:
        print(f"Reading from {input_file}...")
        try:
            file_data = []
            with open(input_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        file_data.append(json.loads(line))
            
            if len(file_data) >= samples_per_file:
                sampled_data = random.sample(file_data, samples_per_file)
            else:
                print(f"Warning: {input_file} only has {len(file_data)} samples, using all available samples.")
                sampled_data = file_data
            
            for sample in sampled_data:
                sample['source_file'] = input_file
            
            merged_data.extend(sampled_data)
            print(f"Sampled {len(sampled_data)} samples from {input_file}")
            
        except FileNotFoundError:
            print(f"Error: Input file not found at {input_file}")
            continue
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON format in {input_file}: {e}")
            continue
    
    print(f"Total merged samples: {len(merged_data)}")
    print(f"Processing samples with mask rate {mask_rate}...")
    
    for sample in tqdm(merged_data, desc=f"Generating perturbations (rate: {mask_rate})"):
        text_content = sample.get('text', '')
        if not text_content:
            sample['mask_perturbations'] = []
            continue
            
        perturbations = []
        for _ in range(num_perturbations):
            perturbed_text = mask_words_with_placeholder(text_content, mask_rate)
            perturbations.append(perturbed_text)
            
        sample['mask_perturbations'] = perturbations
    
    print(f"Saving merged and processed results to {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, indent=2, ensure_ascii=False)
    
    print(f"Processing completed for {output_file}!")
    return len(merged_data)

def process_jsonl_file_with_placeholder(input_file, output_file, mask_rate, num_perturbations=5):
    """
    Process JSONL file with placeholder perturbations.
    """
    try:
        data = []
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_file}")
        return
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format in {input_file}: {e}")
        return

    print(f"Processing {len(data)} samples for {output_file} with mask rate {mask_rate}...")
    
    for sample in tqdm(data, desc=f"Generating perturbations (rate: {mask_rate})"):
        text_content = sample.get('text', '')
        if not text_content:
            sample['mask_perturbations'] = []
            continue
            
        perturbations = []
        for _ in range(num_perturbations):
            perturbed_text = mask_words_with_placeholder(text_content, mask_rate)
            perturbations.append(perturbed_text)
            
        sample['mask_perturbations'] = perturbations
    
    print(f"Saving updated results to {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        for sample in data:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    print(f"Processing completed for {output_file}!")

def process_json_file_with_placeholder(input_file, output_file, mask_rate, num_perturbations=5):
    """
    Process JSON file with target_example format.
    """
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_file}")
        return

    print(f"Processing {len(data)} samples for {output_file} with mask rate {mask_rate}...")
    
    for sample in tqdm(data, desc=f"Generating perturbations (rate: {mask_rate})"):
        target_output = sample.get('target_example', {}).get('output', '')
        if not target_output:
            sample['mask_perturbations'] = []
            continue
            
        perturbations = []
        for _ in range(num_perturbations):
            perturbed_text = mask_words_with_placeholder(target_output, mask_rate)
            perturbations.append(perturbed_text)
            
        sample['mask_perturbations'] = perturbations
    
    print(f"Saving updated results to {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"Processing completed for {output_file}!")

def process_json_file_with_placeholder_joint(input_file, output_file, mask_rate, num_perturbations=5):
    """
    Process JSON file with joint perturbations on instruction, input, and output.
    """
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_file}")
        return

    print(f"Processing {len(data)} samples for {output_file} with mask rate {mask_rate} (joint: instruction+input+output)...")

    for sample in tqdm(data, desc=f"Generating joint perturbations (rate: {mask_rate})"):
        target_example = sample.get('target_example', {})
        instruction_text = target_example.get('instruction', '') or ''
        input_text = target_example.get('input', '') or ''
        output_text = target_example.get('output', '') or ''

        joint_perturbations = []
        output_only_perturbations = []

        if not any([instruction_text.strip(), input_text.strip(), output_text.strip()]):
            sample['mask_perturbations_joint'] = []
            sample['mask_perturbations'] = []
            continue

        for _ in range(num_perturbations):
            perturbed_instruction = mask_words_with_placeholder(instruction_text, mask_rate)
            perturbed_input = mask_words_with_placeholder(input_text, mask_rate)
            perturbed_output = mask_words_with_placeholder(output_text, mask_rate)

            joint_perturbations.append({
                'instruction': perturbed_instruction,
                'input': perturbed_input,
                'output': perturbed_output
            })
            output_only_perturbations.append(perturbed_output)

        sample['mask_perturbations_joint'] = joint_perturbations
        sample['mask_perturbations'] = output_only_perturbations

    print(f"Saving updated results to {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Processing completed for {output_file}!")

def process_standard_json_to_target_example(input_file, output_file, mask_rate, num_perturbations=5):
    """
    Convert standard format (instruction/input/output) to target_example format with perturbations.
    """
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_file}")
        return

    print(f"Converting {len(data)} samples to target_example format...")
    converted_data = []
    
    for sample in tqdm(data, desc="Converting and generating perturbations"):
        instruction_text = sample.get('instruction', '') or ''
        input_text = sample.get('input', '') or ''
        output_text = sample.get('output', '') or ''
        
        wrapped_sample = {
            'target_example': {
                'instruction': instruction_text,
                'input': input_text,
                'output': output_text
            }
        }
        
        # Preserve label field if present
        if 'label' in sample:
            wrapped_sample['label'] = sample['label']
        
        perturbations = []
        for _ in range(num_perturbations):
            perturbed_text = mask_words_with_placeholder(output_text, mask_rate)
            perturbations.append(perturbed_text)
        wrapped_sample['mask_perturbations'] = perturbations
        
        converted_data.append(wrapped_sample)
    
    print(f"Saving results to {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(converted_data, f, indent=2, ensure_ascii=False)
    
    print(f"Processing completed for {output_file}!")

def main():
    parser = argparse.ArgumentParser(description='Generate text perturbations by replacing words with a "word" placeholder.')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    single_parser = subparsers.add_parser('single', help='Process a single file')
    single_parser.add_argument('--input', '-i', required=True, help='Input JSON or JSONL file path.')
    single_parser.add_argument('--output', '-o', required=True, help='Output JSON or JSONL file path to create or overwrite.')
    single_parser.add_argument('--mask_rate', type=float, required=True, help='Proportion of words to mask (e.g., 0.1 for 10%%).')
    single_parser.add_argument('--num_perturbations', '-p', type=int, default=5, help='Number of perturbations to generate per sample.')
    single_parser.add_argument('--format', choices=['json', 'jsonl'], default='auto', help='File format (json or jsonl). If "auto", detect from file extension.')
    single_parser.add_argument('--joint', action='store_true', help='If set, jointly perturb instruction, input, and output into mask_perturbations_joint (JSON only).')
    
    sample_parser = subparsers.add_parser('sample', help='Sample from multiple JSONL files and merge')
    sample_parser.add_argument('--inputs', '-i', nargs='+', required=True, help='Input JSONL file paths.')
    sample_parser.add_argument('--output', '-o', required=True, help='Output JSON file path.')
    sample_parser.add_argument('--samples_per_file', '-s', type=int, required=True, help='Number of samples to take from each input file.')
    sample_parser.add_argument('--mask_rate', type=float, required=True, help='Proportion of words to mask (e.g., 0.1 for 10%%).')
    sample_parser.add_argument('--num_perturbations', '-p', type=int, default=5, help='Number of perturbations to generate per sample.')
    sample_parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducible sampling.')
    
    convert_parser = subparsers.add_parser('convert', help='Convert standard format to target_example format')
    convert_parser.add_argument('--input', '-i', required=True, help='Input JSON file with standard format (instruction/input/output).')
    convert_parser.add_argument('--output', '-o', required=True, help='Output JSON file with target_example format.')
    convert_parser.add_argument('--mask_rate', type=float, required=True, help='Proportion of words to mask (e.g., 0.1 for 10%%).')
    convert_parser.add_argument('--num_perturbations', '-p', type=int, default=5, help='Number of perturbations to generate per sample.')
    
    args = parser.parse_args()
    
    if args.command == 'sample':
        random.seed(args.seed)
        sample_and_merge_jsonl_files(
            args.inputs,
            args.samples_per_file,
            args.output,
            args.mask_rate,
            args.num_perturbations
        )
    
    elif args.command == 'convert':
        process_standard_json_to_target_example(
            args.input,
            args.output,
            args.mask_rate,
            args.num_perturbations
        )
    
    elif args.command == 'single':
        if args.format == 'auto':
            if args.input.endswith('.jsonl'):
                file_format = 'jsonl'
            elif args.input.endswith('.json'):
                file_format = 'json'
            else:
                print("Error: Cannot determine file format. Please specify --format json or --format jsonl")
                return
        else:
            file_format = args.format
        
        if file_format == 'jsonl':
            if args.joint:
                print('Error: --joint is only supported for JSON array inputs, not JSONL. Remove --joint or convert file to JSON.')
                return
            process_jsonl_file_with_placeholder(
                args.input,
                args.output,
                args.mask_rate,
                args.num_perturbations
            )
        else:
            if args.joint:
                process_json_file_with_placeholder_joint(
                    args.input,
                    args.output,
                    args.mask_rate,
                    args.num_perturbations
                )
            else:
                process_json_file_with_placeholder(
                    args.input,
                    args.output,
                    args.mask_rate,
                    args.num_perturbations
                )
    
    else:
        if hasattr(args, 'input') and args.input:
            if args.input.endswith('.jsonl'):
                file_format = 'jsonl'
            elif args.input.endswith('.json'):
                file_format = 'json'
            else:
                print("Error: Cannot determine file format. Please specify --format json or --format jsonl")
                return
            
            if file_format == 'jsonl':
                process_jsonl_file_with_placeholder(
                    args.input,
                    args.output,
                    args.mask_rate,
                    args.num_perturbations
                )
            else:
                process_json_file_with_placeholder(
                    args.input,
                    args.output,
                    args.mask_rate,
                    args.num_perturbations
                )
        else:
            parser.print_help()

if __name__ == "__main__":
    main()

