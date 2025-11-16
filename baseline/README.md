# Baseline

This baseline customizes the [mia_llms_benchmark](https://github.com/computationalprivacy/mia_llms_benchmark) project; we gratefully acknowledge their work.

## Quick Start

```bash
python main.py -c config_file.yaml --attacks loss minkprob zlib --output results.pkl
```

- `-c, --config`: configuration file path (required)  
- `--attacks`: list of attacks to run (e.g., loss, minkprob, minkplusplus, zlib)  
- `--run-all`: run every attack defined in the config  
- `--target_model`: override the target model name  
- `--output`: where to store serialized results (default `TGneighbor_results.pkl`)

## Examples

```bash
python main.py -c config_healthcaremagic.yaml --attacks loss minkprob --output results.pkl
python main.py -c config_healthcaremagic.yaml --run-all --output all_results.pkl
```

