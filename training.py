# training.py
# Functions for training and evaluating models

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager
import numpy as np
import os

from utils import compute_metrics

def train_model_with_dpsgd(model, train_dataset, eval_dataset, config, device):
    """
    Fine-tune model with Differentially Private SGD (DP-SGD).
    
    Args:
        model: Model to train
        train_dataset: Training dataset
        eval_dataset: Evaluation dataset
        config: Configuration object
        device: Device to run on
        
    Returns:
        Trained model, best metrics
    """
    print("Starting fine-tuning with DP-SGD...")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True
    )
    
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=config.batch_size,
        shuffle=False
    )
    
    # Set up optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    # Set up DP-SGD if enabled
    if config.enable_dp:
        privacy_engine = PrivacyEngine()
        
        # Attach privacy engine to optimizer
        model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=train_loader,
            epochs=config.num_epochs,
            target_epsilon=config.epsilon,
            target_delta=config.delta,
            max_grad_norm=config.max_grad_norm,
        )
        
        # Get the actual noise multiplier
        noise_multiplier = privacy_engine.accountant.get_privacy_spent(target_delta=config.delta)[0]
        print(f"For target ε={config.epsilon}, δ={config.delta}:")
        print(f"Using noise multiplier: {noise_multiplier}")
    
    # Set up learning rate scheduler
    total_steps = len(train_loader) * config.num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=total_steps
    )
    
    # Training loop
    best_metric = 0
    best_metrics = {}
    for epoch in range(config.num_epochs):
        # Training
        model.train()
        epoch_loss = 0
        
        with BatchMemoryManager(
            data_loader=train_loader,
            max_physical_batch_size=32,
            optimizer=optimizer
        ) as memory_safe_data_loader:
            for batch in tqdm(memory_safe_data_loader, desc=f"Epoch {epoch+1}"):
                # Move batch to device
                batch = {k: v.to(device) for k, v in batch.items() if k != 'idx'}
                
                # Labels key may be 'label' or 'labels' depending on the dataset
                labels_key = 'label' if 'label' in batch else 'labels'
                
                # Forward pass
                outputs = model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    token_type_ids=batch['token_type_ids'] if 'token_type_ids' in batch else None,
                    labels=batch[labels_key]
                )
                
                loss = outputs[0] if isinstance(outputs, tuple) else outputs.loss
                
                # Backward pass and optimization
                loss.backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1} average loss: {avg_loss:.4f}")
        
        # Evaluation
        metrics = evaluate_model(model, eval_loader, config, device)
        print(f"Epoch {epoch+1} evaluation metrics: {metrics}")
        
        # Save best model based on primary metric (accuracy or F1)
        primary_metric = "accuracy" if config.task_name in ["sst2", "mnli", "cola"] else "f1"
        if metrics[primary_metric] > best_metric:
            best_metric = metrics[primary_metric]
            best_metrics = metrics
            # Save the best model
            os.makedirs("models", exist_ok=True)
            torch.save(model.state_dict(), f"models/best_model_{config.task_name}_eps{config.epsilon}.pt")
    
    # Load the best model
    model.load_state_dict(torch.load(f"models/best_model_{config.task_name}_eps{config.epsilon}.pt"))
    
    return model, best_metrics


def train_model_without_dp(model, train_dataset, eval_dataset, config, device):
    """
    Fine-tune model without differential privacy (standard training).
    
    Args:
        model: Model to train
        train_dataset: Training dataset
        eval_dataset: Evaluation dataset
        config: Configuration object
        device: Device to run on
        
    Returns:
        Trained model, best metrics
    """
    print("Starting standard fine-tuning (no DP)...")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True
    )
    
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=config.batch_size,
        shuffle=False
    )
    
    # Set up optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    # Set up learning rate scheduler
    total_steps = len(train_loader) * config.num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=total_steps
    )
    
    # Training loop
    best_metric = 0
    best_metrics = {}
    for epoch in range(config.num_epochs):
        # Training
        model.train()
        epoch_loss = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items() if k != 'idx'}
            
            # Labels key may be 'label' or 'labels' depending on the dataset
            labels_key = 'label' if 'label' in batch else 'labels'
            
            # Forward pass
            outputs = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                token_type_ids=batch['token_type_ids'] if 'token_type_ids' in batch else None,
                labels=batch[labels_key]
            )
            
            loss = outputs[0] if isinstance(outputs, tuple) else outputs.loss
            
            # Backward pass and optimization
            loss.backward()
            clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1} average loss: {avg_loss:.4f}")
        
        # Evaluation
        metrics = evaluate_model(model, eval_loader, config, device)
        print(f"Epoch {epoch+1} evaluation metrics: {metrics}")
        
        # Save best model based on primary metric (accuracy or F1)
        primary_metric = "accuracy" if config.task_name in ["sst2", "mnli", "cola"] else "f1"
        if metrics[primary_metric] > best_metric:
            best_metric = metrics[primary_metric]
            best_metrics = metrics
            # Save the best model
            os.makedirs("models", exist_ok=True)
            torch.save(model.state_dict(), f"models/best_model_{config.task_name}_no_dp.pt")
    
    # Load the best model
    model.load_state_dict(torch.load(f"models/best_model_{config.task_name}_no_dp.pt"))
    
    return model, best_metrics


def evaluate_model(model, eval_loader, config, device):
    """
    Evaluate model on a dataset.
    
    Args:
        model: Model to evaluate
        eval_loader: DataLoader for evaluation
        config: Configuration object
        device: Device to run on
        
    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()
    
    all_predictions = []
    all_labels = []
    
    # Determine if we're doing token classification (NER) or sequence classification
    is_token_classification = config.dataset_name == "conll2003"
    
    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="Evaluating"):
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items() if k != 'idx'}
            
            # Labels key may be 'label' or 'labels' depending on the dataset
            labels_key = 'label' if 'label' in batch else 'labels'
            
            # Forward pass
            outputs = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                token_type_ids=batch['token_type_ids'] if 'token_type_ids' in batch else None
            )
            
            logits = outputs[1] if isinstance(outputs, tuple) else outputs
            
            # Get predictions
            if is_token_classification:
                # For token classification, predictions shape: [batch_size, seq_len, num_classes]
                predictions = torch.argmax(logits, dim=2)
            else:
                # For sequence classification, predictions shape: [batch_size, num_classes]
                predictions = torch.argmax(logits, dim=1)
            
            # Collect predictions and labels
            all_predictions.append(predictions.cpu())
            all_labels.append(batch[labels_key].cpu())
    
    # Concatenate all predictions and labels
    all_predictions = torch.cat(all_predictions, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    # Calculate metrics
    task = "conll2003" if is_token_classification else config.task_name
    metrics = compute_metrics(all_predictions, all_labels, task)
    
    return metrics


def test_model(model, test_dataset, config, device):
    """
    Test model on a test dataset.
    
    Args:
        model: Model to test
        test_dataset: Test dataset
        config: Configuration object
        device: Device to run on
        
    Returns:
        Dictionary of test metrics
    """
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False
    )
    
    print("\n" + "="*80)
    print(f"Testing model on {config.task_name} test set...")
    print("="*80)
    test_metrics = evaluate_model(model, test_loader, config, device)
    
    # Print results in a more prominent way
    print("\n" + "="*80)
    print(f"Test Results for {config.task_name}:")
    print("-"*80)
    for metric_name, value in test_metrics.items():
        print(f"{metric_name.capitalize()}: {value:.4f}")
    print("="*80 + "\n")
    
    return test_metrics