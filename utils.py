# utils.py
# Utility functions for Differentially Private Embeddings

import os
import random
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset, Subset, random_split

# Set random seeds for reproducibility
def set_seed(seed):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Get device (CPU or GPU)
def get_device():
    """Return the appropriate device (GPU or CPU)"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Compute metrics for evaluation
def compute_metrics(predictions, labels, task):
    """
    Compute evaluation metrics based on the task
    
    Args:
        predictions: Model predictions
        labels: Ground truth labels
        task: Task name (e.g., 'sst2', 'qqp', 'conll2003')
        
    Returns:
        Dictionary of metrics
    """
    # Convert tensors to numpy if needed
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    
    # Calculate metrics based on task
    results = {}
    
    if task in ["sst2", "mnli", "cola"]:
        # Classification tasks
        results["accuracy"] = accuracy_score(labels, predictions)
        if task != "mnli":  # Binary classification
            results["precision"] = precision_score(labels, predictions, average='binary')
            results["recall"] = recall_score(labels, predictions, average='binary')
            results["f1"] = f1_score(labels, predictions, average='binary')
        else:  # Multi-class classification
            results["precision"] = precision_score(labels, predictions, average='macro')
            results["recall"] = recall_score(labels, predictions, average='macro')
            results["f1"] = f1_score(labels, predictions, average='macro')
    
    elif task == "qqp":
        # Paraphrase detection
        results["accuracy"] = accuracy_score(labels, predictions)
        results["precision"] = precision_score(labels, predictions, average='binary')
        results["recall"] = recall_score(labels, predictions, average='binary')
        results["f1"] = f1_score(labels, predictions, average='binary')
    
    elif task == "conll2003":
        # NER task
        mask = labels != -100  # Remove padding tokens
        filtered_predictions = predictions[mask]
        filtered_labels = labels[mask]
        results["accuracy"] = accuracy_score(filtered_labels, filtered_predictions)
        results["precision"] = precision_score(filtered_labels, filtered_predictions, average='macro')
        results["recall"] = recall_score(filtered_labels, filtered_predictions, average='macro')
        results["f1"] = f1_score(filtered_labels, filtered_predictions, average='macro')
    
    return results

# Plot privacy-utility trade-off
def plot_privacy_utility_tradeoff(epsilon_values, metrics, method_names, metric_name="accuracy"):
    """
    Plot privacy-utility trade-off curves
    
    Args:
        epsilon_values: List of epsilon values
        metrics: Dictionary mapping method names to lists of metric values
        method_names: List of method names to include in the plot
        metric_name: Name of the metric to plot (e.g., 'accuracy', 'f1')
    """
    plt.figure(figsize=(10, 6))
    
    for method in method_names:
        if method in metrics:
            plt.plot(epsilon_values, metrics[method], marker='o', label=method)
    
    plt.xlabel('Privacy Budget (ε)')
    plt.ylabel(f'{metric_name.capitalize()}')
    plt.title(f'Privacy-Utility Trade-off: {metric_name.capitalize()} vs. Privacy Budget')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(f'privacy_utility_tradeoff_{metric_name}.png')
    plt.close()

def split_dataset_for_teachers(dataset, num_teachers):
    """
    Split a dataset evenly among teachers
    
    Args:
        dataset: The dataset to split
        num_teachers: Number of teacher models
    
    Returns:
        List of dataset splits
    """
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    random.shuffle(indices)  # Shuffle for random assignment
    
    split_size = dataset_size // num_teachers
    
    teacher_datasets = []
    for i in range(num_teachers):
        start_idx = i * split_size
        end_idx = (i + 1) * split_size if i < num_teachers - 1 else dataset_size
        teacher_idx = indices[start_idx:end_idx]
        teacher_datasets.append(Subset(dataset, teacher_idx))
    
    return teacher_datasets

def create_dataloaders(train_dataset, eval_dataset, test_dataset, batch_size, eval_batch_size=None):
    """
    Create DataLoaders for training, evaluation, and testing
    
    Args:
        train_dataset: Training dataset
        eval_dataset: Evaluation dataset
        test_dataset: Test dataset
        batch_size: Batch size for training
        eval_batch_size: Batch size for evaluation and testing (defaults to batch_size)
    
    Returns:
        train_loader, eval_loader, test_loader
    """
    if eval_batch_size is None:
        eval_batch_size = batch_size
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True
    )
    
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=eval_batch_size,
        shuffle=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=eval_batch_size,
        shuffle=False
    )
    
    return train_loader, eval_loader, test_loader
