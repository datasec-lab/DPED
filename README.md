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
git clone https://github.com/ShuyaFeng/DPED.git
cd DPED

# Install requirements
pip install -r requirements.txt
```

## Requirements

- Python 3.8+
- PyTorch 1.7.0+ (but <2.1.0 for Opacus compatibility)
- Transformers 4.0.0+
- Opacus 1.4.0 (specific version for PyTorch compatibility)
- Datasets 1.0.0+
- NumPy 1.19.0+
- scikit-learn 0.24.0+
- tqdm 4.50.0+
- matplotlib 3.3.0+

**Note**: The specific Opacus version (1.4.0) is required for compatibility with PyTorch 2.0.x. Newer versions of Opacus may not work with this PyTorch version.

## File Structure

- `config.py`: Configuration class for experiment parameters
- `utils.py`: Utility functions
- `data_loader.py`: Functions for loading and preprocessing datasets
- `models.py`: Model classes including TeacherEnsemble, StudentModel, and distillation functions
- `training.py`: Functions for training and evaluating models
- `main.py`: Main script for running individual experiments
- `requirements.txt`: Python package dependencies

## How to Run

### Basic Usage

```bash
# Train on SST-2 with our proposed DP-Distill method
python3 main.py --dataset glue --task sst2 --method dp-distill --epsilon 8.0

# Train on CoNLL-2003 NER with DP-SGD baseline
python3 main.py --dataset conll2003 --method dp-sgd --epsilon 4.0

# Quick test with reduced settings
python3 main.py --dataset glue --task sst2 --method non-private --num_epochs 1 --batch_size 8
```

### Running Comprehensive Experiments

Since the wrapper scripts have been removed to clean up the codebase, use the main script with different flags:

```bash
# Compare all methods on SST-2
python3 main.py --dataset glue --task sst2 --run_all_methods --epsilon 8.0

# Run privacy-utility tradeoff experiments 
python3 main.py --dataset glue --task sst2 --privacy_utility_tradeoff

# Run experiments on all GLUE tasks (run each separately)
python3 main.py --dataset glue --task sst2 --method dp-distill --epsilon 8.0
python3 main.py --dataset glue --task qqp --method dp-distill --epsilon 8.0
python3 main.py --dataset glue --task mnli --method dp-distill --epsilon 8.0
python3 main.py --dataset glue --task cola --method dp-distill --epsilon 8.0
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

### Additional Useful Arguments

- `--batch_size`: Batch size for training (default: 16, use smaller values like 8 for limited memory)
- `--num_epochs`: Number of training epochs (default: 3, use 1 for quick testing)
- `--seed`: Random seed for reproducibility
- `--max_seq_length`: Maximum sequence length for input (default: 128)

## Methods

1. **non-private**: Standard training without differential privacy (baseline)
2. **dp-sgd**: Training with DP-SGD (baseline)
3. **pate-distill**: PATE-style teacher-student distillation (baseline)
4. **dp-distill**: Our proposed method (teacher-student with multi-layer DP)

## Detailed Example Experiments

### GLUE Results (at ε≈8)

To reproduce the GLUE benchmark results, run individual tasks:

```bash
python3 main.py --dataset glue --task sst2 --method dp-distill --epsilon 8.0
python3 main.py --dataset glue --task qqp --method dp-distill --epsilon 8.0
python3 main.py --dataset glue --task mnli --method dp-distill --epsilon 8.0
python3 main.py --dataset glue --task cola --method dp-distill --epsilon 8.0
```

### CoNLL-2003 NER Results

For named entity recognition results:

```bash
python3 main.py --dataset conll2003 --method dp-distill --epsilon 8.0
python3 main.py --dataset conll2003 --method dp-distill --epsilon 4.0
```

### Privacy-Utility Tradeoff

To generate privacy-utility tradeoff curves:

```bash
python3 main.py --dataset glue --task sst2 --privacy_utility_tradeoff --run_all_methods
```

### Ablation Studies

To analyze the contribution of each component, run the full method with different configurations:

```bash
# Full model
python3 main.py --dataset glue --task sst2 --method dp-distill --epsilon 8.0

# Without multi-layer noise
python3 main.py --dataset glue --task sst2 --method dp-distill --epsilon 8.0 --no_multi_layer_noise

# With fewer teachers
python3 main.py --dataset glue --task sst2 --method dp-distill --epsilon 8.0 --num_teachers 3

# With more noise
python3 main.py --dataset glue --task sst2 --method dp-distill --epsilon 8.0 --teacher_noise 1.0
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

## Troubleshooting

### Common Issues

1. **AttributeError: module 'torch.nn' has no attribute 'RMSNorm'**
   - This indicates a version compatibility issue between PyTorch and Opacus
   - Solution: Install the specific compatible version with `pip install opacus==1.4.0`

2. **ModuleNotFoundError: No module named 'transformers'**
   - Install missing dependencies: `pip install -r requirements.txt`

3. **CUDA out of memory errors**
   - Reduce batch size: `--batch_size 8` or `--batch_size 4`
   - Reduce sequence length: `--max_seq_length 64`

4. **Slow training**
   - For quick testing, use: `--num_epochs 1 --batch_size 8`
   - Use CPU if GPU memory is limited (training will be slower)

### Installation Issues

If you encounter package conflicts, try creating a fresh virtual environment:

```bash
python3 -m venv dped_env
source dped_env/bin/activate  # On Windows: dped_env\Scripts\activate
pip install -r requirements.txt
```

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
