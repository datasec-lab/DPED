# Differentially Private Embeddings for NLP

This repository contains the implementation of "Differentially Private Embeddings for NLP via Teacher-Student Distillation and Multi-Layer Differential Privacy" as described in the paper.

## Overview

The framework combines teacher-student distillation with multi-layer DP noise injection and rare-word-aware aggregation to learn high-quality, differentially private embeddings for NLP tasks.

### Key Components

1. **Teacher-Student Distillation**: We partition sensitive data into disjoint shards to train an ensemble of teacher models. Their outputs are aggregated with noise to produce a consensus representation for student training.

2. **Multi-Layer Differential Privacy**: Rather than applying noise uniformly (as in DP-SGD), we inject DP noise at multiple layers in the network, distributing the privacy budget more efficiently.

3. **Rare-Word-Aware Aggregation**: We adapt our aggregation mechanism to handle rare tokens, which are prone to memorization.

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/dp-embeddings-nlp.git
cd dp-embeddings-nlp

# Install requirements
pip install -r requirements.txt
```

## Requirements

- Python 3.8+
- PyTorch 1.9+
- Transformers 4.11+
- Opacus 0.15+
- Datasets
- NumPy
- scikit-learn
- tqdm
- matplotlib

## File Structure

- `config.py`: Configuration class for experiment parameters
- `utils.py`: Utility functions
- `data_loader.py`: Functions for loading and preprocessing datasets
- `models.py`: Model classes including TeacherEnsemble, StudentModel, and distillation functions
- `training.py`: Functions for training and evaluating models
- `main.py`: Main script for running individual experiments
- `run_experiments.py`: Script for running batches of experiments

## How to Run

### Basic Usage

```bash
# Train on SST-2 with our proposed DP-Distill method
python main.py --dataset glue --task sst2 --method dp-distill --epsilon 8.0

# Train on CoNLL-2003 NER with DP-SGD baseline
python main.py --dataset conll2003 --method dp-sgd --epsilon 4.0
```

### Running Comprehensive Experiments

```bash
# Run experiments on all GLUE tasks
python run_experiments.py --experiment glue

# Run experiments on CoNLL-2003 NER
python run_experiments.py --experiment conll

# Run privacy-utility tradeoff experiments 
python run_experiments.py --experiment privacy_utility

# Run ablation studies
python run_experiments.py --experiment ablation
```

## Command Line Arguments

### Main Script

- `--dataset`: Dataset to use (glue or conll2003)
- `--task`: Task name for GLUE benchmark (sst2, qqp, mnli, cola)
- `--method`: Method to use (non-private, dp-sgd, pate-distill, dp-distill)
- `--epsilon`: Privacy budget
- `--delta`: Privacy parameter delta
- `--num_teachers`: Number of teacher models
- `--no_multi_layer_noise`: Disable multi-layer noise injection
- `--rare_token_threshold`: Threshold for rare token detection
- `--run_all_methods`: Run all methods for comparison
- `--privacy_utility_tradeoff`: Run privacy-utility tradeoff experiment

### Run Experiments Script

- `--experiment`: Type of experiment to run (glue, conll, privacy_utility, ablation)
- `--output_dir`: Directory to save results

## Methods

1. **non-private**: Standard training without differential privacy (baseline)
2. **dp-sgd**: Training with DP-SGD (baseline)
3. **pate-distill**: PATE-style teacher-student distillation (baseline)
4. **dp-distill**: Our proposed method (teacher-student with multi-layer DP)

## Detailed Example Experiments

### GLUE Results (at ε≈8)

To reproduce the GLUE benchmark results:

```bash
python run_experiments.py --experiment glue
```

Or run individual tasks:

```bash
python main.py --dataset glue --task sst2 --method dp-distill --epsilon 8.0
python main.py --dataset glue --task qqp --method dp-distill --epsilon 8.0
python main.py --dataset glue --task mnli --method dp-distill --epsilon 8.0
python main.py --dataset glue --task cola --method dp-distill --epsilon 8.0
```

### CoNLL-2003 NER Results

For named entity recognition results:

```bash
python main.py --dataset conll2003 --method dp-distill --epsilon 8.0
python main.py --dataset conll2003 --method dp-distill --epsilon 4.0
```

### Privacy-Utility Tradeoff

To generate privacy-utility tradeoff curves:

```bash
python main.py --dataset glue --task sst2 --privacy_utility_tradeoff --run_all_methods
```

### Ablation Studies

To analyze the contribution of each component:

```bash
python run_experiments.py --experiment ablation
```

## Results

The framework achieves significantly improved privacy-utility trade-offs:

- Up to 10 percentage points improvement in accuracy/F1 over baselines
- More efficient use of privacy budget
- Better handling of rare tokens

## Expected Output

Results will be saved in:
- Model checkpoints: `models/best_model_*`
- Privacy-utility plots: `privacy_utility_tradeoff_*.png`
- Results summaries: `results/*/results_summary.txt`

## Citation

If you use this code, please cite:

```
@article{dp-embeddings-nlp,
  title={Differentially Private Embeddings for NLP via Teacher-Student Distillation and Multi-Layer Differential Privacy},
  author={Your Name},
  journal={ArXiv},
  year={2023}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
