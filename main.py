# main.py
# Main script for running experiments with Differentially Private Embeddings

import os
import argparse
import torch
from transformers import BertForSequenceClassification

# Import our modules
from config import Config
from utils import set_seed, get_device, plot_privacy_utility_tradeoff
from data_loader import load_glue_dataset, load_conll_dataset, create_public_dataset
# Updated import from combined models file
from models import StudentModel, BertForTokenClassification, TeacherEnsemble, distill_embeddings, pate_distillation
from training import train_model_with_dpsgd, train_model_without_dp, test_model

def parse_args():
    parser = argparse.ArgumentParser(description="Differentially Private Embeddings for NLP")
    
    # Dataset parameters
    parser.add_argument("--dataset", type=str, default="glue", choices=["glue", "conll2003"],
                        help="Dataset to use (glue or conll2003)")
    parser.add_argument("--task", type=str, default="sst2", 
                        choices=["sst2", "qqp", "mnli", "cola"],
                        help="Task name for GLUE benchmark")
    
    # Model parameters
    parser.add_argument("--bert_model", type=str, default="bert-base-uncased",
                        help="BERT model to use")
    parser.add_argument("--max_seq_length", type=int, default=128,
                        help="Maximum sequence length")
    
    # Privacy parameters
    parser.add_argument("--epsilon", type=float, default=8.0,
                        help="Privacy budget epsilon")
    parser.add_argument("--delta", type=float, default=1e-5,
                        help="Privacy parameter delta")
    parser.add_argument("--no_dp", action="store_true",
                        help="Disable differential privacy (for baseline)")
    
    # Teacher-Student parameters
    parser.add_argument("--num_teachers", type=int, default=5,
                        help="Number of teacher models")
    parser.add_argument("--teacher_noise", type=float, default=0.5,
                        help="Noise for teacher output aggregation")
    parser.add_argument("--rare_token_threshold", type=int, default=2,
                        help="Threshold for rare token detection")
    parser.add_argument("--rare_token_noise_factor", type=float, default=2.0,
                        help="Noise factor for rare tokens")
    
    # Multi-layer DP parameters
    parser.add_argument("--no_multi_layer_noise", action="store_true",
                        help="Disable multi-layer noise injection")
    parser.add_argument("--embedding_noise_std", type=float, default=0.1,
                        help="Standard deviation of noise at embedding layer")
    parser.add_argument("--intermediate_noise_std", type=float, default=0.05,
                        help="Standard deviation of noise at intermediate layer")
    
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for training and evaluation")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    # Experiment parameters
    parser.add_argument("--method", type=str, default="dp-distill",
                        choices=["non-private", "dp-sgd", "pate-distill", "dp-distill"],
                        help="Method to use")
    parser.add_argument("--run_all_methods", action="store_true",
                        help="Run all methods for comparison")
    parser.add_argument("--privacy_utility_tradeoff", action="store_true",
                        help="Run privacy-utility tradeoff experiment")
    
    return parser.parse_args()


def update_config_from_args(config, args):
    """Update config with command line arguments"""
    
    # Dataset parameters
    config.dataset_name = args.dataset
    config.task_name = args.task
    
    # Model parameters
    config.bert_model = args.bert_model
    config.max_seq_length = args.max_seq_length
    
    # Privacy parameters
    config.epsilon = args.epsilon
    config.delta = args.delta
    config.enable_dp = not args.no_dp
    
    # Teacher-Student parameters
    config.num_teachers = args.num_teachers
    config.teacher_aggregation_noise = args.teacher_noise
    config.rare_token_threshold = args.rare_token_threshold
    config.rare_token_noise_factor = args.rare_token_noise_factor
    
    # Multi-layer DP parameters
    config.multi_layer_noise = not args.no_multi_layer_noise
    config.embedding_noise_std = args.embedding_noise_std
    config.intermediate_noise_std = args.intermediate_noise_std
    
    # Training parameters
    config.batch_size = args.batch_size
    config.learning_rate = args.learning_rate
    config.num_epochs = args.num_epochs
    config.weight_decay = args.weight_decay
    
    return config


def run_method(method, train_dataset, eval_dataset, test_dataset, public_dataset, config, device):
    """
    Run a specific method for training with differential privacy.
    
    Args:
        method: Method name (non-private, dp-sgd, pate-distill, dp-distill)
        train_dataset: Training dataset
        eval_dataset: Evaluation dataset
        test_dataset: Test dataset
        public_dataset: Public/unlabeled dataset for distillation
        config: Configuration object
        device: Device to run on
        
    Returns:
        Trained model and test metrics
    """
    print(f"\n{'='*80}")
    print(f"Running method: {method}")
    print(f"Task: {config.task_name}")
    print(f"Epsilon: {config.epsilon}")
    print(f"{'='*80}")
    
    # Create appropriate model based on dataset and task
    if config.dataset_name == "conll2003":
        model = BertForTokenClassification(config).to(device)
    else:
        if method == "dp-distill" or method == "pate-distill":
            model = StudentModel(config).to(device)
        else:
            model = BertForSequenceClassification.from_pretrained(
                config.bert_model,
                num_labels=2 if config.task_name in ["sst2", "qqp", "cola"] else 3
            ).to(device)
    
    # Train based on method
    if method == "non-private":
        # Standard training without DP
        config_copy = Config()
        for key, value in vars(config).items():
            setattr(config_copy, key, value)
        config_copy.enable_dp = False
        
        model, _ = train_model_without_dp(model, train_dataset, eval_dataset, config_copy, device)
    
    elif method == "dp-sgd":
        # DP-SGD training
        config_copy = Config()
        for key, value in vars(config).items():
            setattr(config_copy, key, value)
        config_copy.enable_dp = True
        
        model, _ = train_model_with_dpsgd(model, train_dataset, eval_dataset, config_copy, device)
    
    elif method == "pate-distill":
        # PATE-based distillation
        config_copy = Config()
        for key, value in vars(config).items():
            setattr(config_copy, key, value)
        config_copy.enable_dp = True
        config_copy.multi_layer_noise = False  # No multi-layer noise for PATE baseline
        
        # Create and train teacher ensemble
        teacher_ensemble = TeacherEnsemble(config_copy, train_dataset, device)
        teacher_ensemble.train_teachers()
        
        # Train student with PATE-style distillation
        from torch.utils.data import DataLoader
        train_loader = DataLoader(train_dataset, batch_size=config_copy.batch_size, shuffle=True)
        model = pate_distillation(teacher_ensemble, model, public_dataset, train_loader, config_copy, device)
        
        # Optional: fine-tune with DP-SGD
        model, _ = train_model_with_dpsgd(model, train_dataset, eval_dataset, config_copy, device)
    
    elif method == "dp-distill":
        # Our proposed method: Teacher-Student Distillation with Multi-Layer DP
        # Create and train teacher ensemble
        teacher_ensemble = TeacherEnsemble(config, train_dataset, device)
        teacher_ensemble.train_teachers()
        
        # First distill embeddings
        model = distill_embeddings(teacher_ensemble, model, public_dataset, config, device)
        
        # Then fine-tune with DP-SGD and multi-layer noise
        model, _ = train_model_with_dpsgd(model, train_dataset, eval_dataset, config, device)
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Test the model
    test_metrics = test_model(model, test_dataset, config, device)
    
    # Print final summary
    print(f"\n{'='*80}")
    print(f"Final Results Summary for {method}:")
    print(f"Task: {config.task_name}")
    print(f"Epsilon: {config.epsilon}")
    print("-"*80)
    for metric_name, value in test_metrics.items():
        print(f"{metric_name.capitalize()}: {value:.4f}")
    print(f"{'='*80}\n")
    
    return model, test_metrics


def run_privacy_utility_tradeoff(train_dataset, eval_dataset, test_dataset, public_dataset, config, device):
    """
    Run privacy-utility tradeoff experiment with different epsilon values.
    
    Args:
        train_dataset: Training dataset
        eval_dataset: Evaluation dataset
        test_dataset: Test dataset
        public_dataset: Public/unlabeled dataset for distillation
        config: Configuration object
        device: Device to run on
    """
    print("\nRunning privacy-utility tradeoff experiment...")
    
    # Define methods to compare
    methods = ["non-private", "dp-sgd", "pate-distill", "dp-distill"]
    
    # Define epsilon values to test
    epsilon_values = [1.0, 2.0, 4.0, 8.0, 16.0]
    
    # Store metrics for each method and epsilon
    results = {method: [] for method in methods}
    
    # Run each method with each epsilon value
    for epsilon in epsilon_values:
        print(f"\n{'='*80}")
        print(f"Running with epsilon = {epsilon}")
        print(f"{'='*80}")
        
        # Update config with current epsilon
        config.epsilon = epsilon
        config.update_epsilon(epsilon)
        
        # Run each method
        for method in methods:
            if method == "non-private":
                # Non-private method only needs to be run once
                if not results[method]:
                    _, metrics = run_method(method, train_dataset, eval_dataset, test_dataset, 
                                          public_dataset, config, device)
                    # Repeat the same metrics for all epsilon values
                    results[method] = [metrics] * len(epsilon_values)
                continue
            
            # Run method with current epsilon
            _, metrics = run_method(method, train_dataset, eval_dataset, test_dataset, 
                                  public_dataset, config, device)
            results[method].append(metrics)
    
    # Extract primary metric values (accuracy or F1)
    primary_metric = "accuracy" if config.task_name in ["sst2", "mnli", "cola"] else "f1"
    metric_values = {
        method: [metrics[primary_metric] for metrics in method_results]
        for method, method_results in results.items()
    }
    
    # Plot results
    plot_privacy_utility_tradeoff(epsilon_values, metric_values, methods, primary_metric)
    
    # Save results to file
    with open(f"privacy_utility_tradeoff_{config.task_name}.txt", "w") as f:
        f.write(f"Privacy-Utility Tradeoff Experiment Results for {config.task_name}\n")
        f.write(f"Primary metric: {primary_metric}\n\n")
        
        f.write("Epsilon values: " + ", ".join(str(eps) for eps in epsilon_values) + "\n\n")
        
        for method in methods:
            f.write(f"{method}: " + ", ".join(f"{value:.4f}" for value in metric_values[method]) + "\n")


def main():
    """Main function to run experiments"""
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Get device
    device = get_device()
    print(f"Using device: {device}")
    
    # Create config
    config = Config()
    config = update_config_from_args(config, args)
    
    # Load datasets
    if config.dataset_name == "glue":
        train_dataset, eval_dataset, test_dataset = load_glue_dataset(config)
    elif config.dataset_name == "conll2003":
        train_dataset, eval_dataset, test_dataset = load_conll_dataset(config)
    else:
        raise ValueError(f"Unknown dataset: {config.dataset_name}")
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Eval dataset size: {len(eval_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    
    # Create public dataset for distillation
    public_dataset = create_public_dataset(config)
    print(f"Public dataset size: {len(public_dataset)}")
    
    # Run experiments
    if args.privacy_utility_tradeoff:
        # Run privacy-utility tradeoff experiment
        run_privacy_utility_tradeoff(
            train_dataset, eval_dataset, test_dataset, public_dataset, config, device
        )
    elif args.run_all_methods:
        # Run all methods for comparison
        methods = ["non-private", "dp-sgd", "pate-distill", "dp-distill"]
        results = {}
        
        for method in methods:
            _, metrics = run_method(
                method, train_dataset, eval_dataset, test_dataset, public_dataset, config, device
            )
            results[method] = metrics
        
        # Print comparison
        print("\nResults Summary:")
        print("-" * 80)
        for method, metrics in results.items():
            print(f"{method}: {metrics}")
    else:
        # Run single method
        run_method(
            args.method, train_dataset, eval_dataset, test_dataset, public_dataset, config, device
        )


if __name__ == "__main__":
    main()