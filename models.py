# models.py
# Model classes for Differentially Private Embeddings

import torch
import torch.nn as nn
from transformers import BertModel, BertConfig, BertForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm import tqdm
import random
import numpy as np

class StudentModel(nn.Module):
    def __init__(self, config):
        """
        Initialize the student model with multi-layer DP capabilities.
        
        Args:
            config: Configuration object
        """
        super(StudentModel, self).__init__()
        
        # Initialize with pre-trained BERT
        self.config = config
        self.bert_config = BertConfig.from_pretrained(config.bert_model)
        self.bert = BertModel.from_pretrained(config.bert_model)
        self.num_labels = self._get_num_labels()
        
        # Classification layer
        self.dropout = nn.Dropout(self.bert_config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.bert_config.hidden_size, self.num_labels)
        
        # Enable multi-layer noise injection
        self.multi_layer_noise = config.multi_layer_noise
        self.embedding_noise_std = config.embedding_noise_std
        self.intermediate_noise_std = config.intermediate_noise_std
        
    def _get_num_labels(self):
        """Get the number of labels for the task."""
        if self.config.task_name == "sst2":
            return 2
        elif self.config.task_name == "qqp":
            return 2
        elif self.config.task_name == "mnli":
            return 3
        elif self.config.task_name == "cola":
            return 2
        elif self.config.dataset_name == "conll2003":
            return 9  # Number of NER tags in CoNLL-2003
        else:
            return 2  # Default
    
    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        """
        Forward pass with optional multi-layer DP noise injection.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask for padding
            token_type_ids: Segment IDs for sentence pairs
            labels: Ground truth labels
            
        Returns:
            Tuple of (loss, logits) or just logits if no labels provided
        """
        # Get embedding layer output
        embedding_output = self.bert.embeddings(
            input_ids=input_ids,
            token_type_ids=token_type_ids
        )
        
        # Inject noise at embedding layer if enabled
        if self.training and self.multi_layer_noise:
            embedding_noise = torch.randn_like(embedding_output) * self.embedding_noise_std
            embedding_output = embedding_output + embedding_noise
        
        # Process through encoder layers
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        encoder_outputs = self.bert.encoder(
            embedding_output,
            attention_mask=extended_attention_mask
        )
        sequence_output = encoder_outputs[0]
        
        # Inject noise at intermediate layer if enabled (e.g., after 6th layer)
        if self.training and self.multi_layer_noise:
            intermediate_noise = torch.randn_like(sequence_output) * self.intermediate_noise_std
            sequence_output = sequence_output + intermediate_noise
        
        # Classification
        pooled_output = self.bert.pooler(sequence_output)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        # Calculate loss if labels provided
        if labels is not None:
            if self.num_labels == 1:
                loss_fn = nn.MSELoss()
                loss = loss_fn(logits.view(-1), labels.view(-1))
            else:
                loss_fn = nn.CrossEntropyLoss()
                loss = loss_fn(logits.view(-1, self.num_labels), labels.view(-1))
            return loss, logits
        
        return logits


class BertForTokenClassification(nn.Module):
    """BERT model for token-level classification tasks like NER"""
    
    def __init__(self, config):
        super(BertForTokenClassification, self).__init__()
        
        self.config = config
        self.num_labels = 9  # CoNLL-2003 has 9 NER tags
        
        self.bert = BertModel.from_pretrained(config.bert_model)
        self.dropout = nn.Dropout(self.bert.config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.bert.config.hidden_size, self.num_labels)
        
        # Enable multi-layer noise injection
        self.multi_layer_noise = config.multi_layer_noise
        self.embedding_noise_std = config.embedding_noise_std
        self.intermediate_noise_std = config.intermediate_noise_std
    
    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        # Get embedding layer output
        embedding_output = self.bert.embeddings(
            input_ids=input_ids,
            token_type_ids=token_type_ids
        )
        
        # Inject noise at embedding layer if enabled
        if self.training and self.multi_layer_noise:
            embedding_noise = torch.randn_like(embedding_output) * self.embedding_noise_std
            embedding_output = embedding_output + embedding_noise
        
        # Process through encoder layers
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        encoder_outputs = self.bert.encoder(
            embedding_output,
            attention_mask=extended_attention_mask
        )
        sequence_output = encoder_outputs[0]
        
        # Inject noise at intermediate layer if enabled
        if self.training and self.multi_layer_noise:
            intermediate_noise = torch.randn_like(sequence_output) * self.intermediate_noise_std
            sequence_output = sequence_output + intermediate_noise
        
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        
        # Calculate loss if labels provided
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss, 
                    labels.view(-1), 
                    torch.tensor(loss_fct.ignore_index).type_as(labels)
                )
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss, logits
        
        return logits


class TeacherEnsemble:
    def __init__(self, config, train_dataset, device):
        """
        Initialize an ensemble of teacher models.
        
        Args:
            config: Configuration object
            train_dataset: The training dataset to split among teachers
            device: Device to run the models on
        """
        self.config = config
        self.device = device
        self.num_teachers = config.num_teachers
        self.teacher_models = []
        
        # Split dataset among teachers
        from utils import split_dataset_for_teachers
        self.teacher_datasets = split_dataset_for_teachers(train_dataset, self.num_teachers)
        
        # Initialize teacher models based on task type
        for i in range(self.num_teachers):
            if config.dataset_name == "conll2003":
                teacher_model = BertForTokenClassification(config).to(device)
            else:
                teacher_model = BertForSequenceClassification.from_pretrained(
                    config.bert_model, 
                    num_labels=self._get_num_labels()
                ).to(device)
            
            self.teacher_models.append(teacher_model)
            
        print(f"Initialized {self.num_teachers} teacher models.")
    
    def _get_num_labels(self):
        """Get the number of labels for the task."""
        if self.config.task_name == "sst2":
            return 2
        elif self.config.task_name == "qqp":
            return 2
        elif self.config.task_name == "mnli":
            return 3
        elif self.config.task_name == "cola":
            return 2
        elif self.config.dataset_name == "conll2003":
            return 9  # Number of NER tags in CoNLL-2003
        else:
            return 2  # Default
    
    def train_teachers(self):
        """Train all teacher models on their respective datasets."""
        for teacher_idx, (teacher_model, teacher_dataset) in enumerate(zip(self.teacher_models, self.teacher_datasets)):
            print(f"\nTraining Teacher {teacher_idx+1}/{self.num_teachers}")
            
            # Create dataloader for this teacher
            teacher_loader = DataLoader(
                teacher_dataset, 
                batch_size=self.config.batch_size, 
                shuffle=True
            )
            
            # Set up optimizer
            optimizer = AdamW(
                teacher_model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
            
            # Set up scheduler
            total_steps = len(teacher_loader) * self.config.num_epochs
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.config.warmup_steps,
                num_training_steps=total_steps
            )
            
            # Training loop for this teacher
            teacher_model.train()
            for epoch in range(self.config.num_epochs):
                epoch_loss = 0
                for batch in tqdm(teacher_loader, desc=f"Teacher {teacher_idx+1}, Epoch {epoch+1}"):
                    # Move batch to device
                    batch = {k: v.to(self.device) for k, v in batch.items() if k != 'idx'}
                    
                    # Forward pass
                    if self.config.dataset_name == "conll2003":
                        outputs = teacher_model(
                            input_ids=batch['input_ids'],
                            attention_mask=batch['attention_mask'],
                            token_type_ids=batch['token_type_ids'] if 'token_type_ids' in batch else None,
                            labels=batch['labels']
                        )
                    else:
                        outputs = teacher_model(
                            input_ids=batch['input_ids'],
                            attention_mask=batch['attention_mask'],
                            token_type_ids=batch['token_type_ids'] if 'token_type_ids' in batch else None,
                            labels=batch['label']
                        )
                    
                    loss = outputs[0] if isinstance(outputs, tuple) else outputs.loss
                    
                    # Backward pass and optimization
                    loss.backward()
                    clip_grad_norm_(teacher_model.parameters(), self.config.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    
                    epoch_loss += loss.item()
                
                avg_loss = epoch_loss / len(teacher_loader)
                print(f"Teacher {teacher_idx+1}, Epoch {epoch+1} average loss: {avg_loss:.4f}")
    
    def get_token_frequency(self, public_dataset):
        """
        Count token frequency across teacher models.
        Used for rare token detection.
        
        Args:
            public_dataset: The public/unlabeled dataset for distillation
        
        Returns:
            Dictionary mapping token IDs to their frequency across teachers
        """
        from transformers import BertTokenizer
        tokenizer = BertTokenizer.from_pretrained(self.config.bert_model)
        token_counts = {}
        
        # Create dataloader for public dataset
        public_loader = DataLoader(
            public_dataset,
            batch_size=self.config.batch_size,
            shuffle=False
        )
        
        # Count tokens across teachers
        for batch in tqdm(public_loader, desc="Counting token frequency"):
            input_ids = batch['input_ids'].to(self.device)
            for token_id in input_ids.unique().cpu().numpy():
                if token_id not in token_counts:
                    token_counts[token_id] = 0
                    
                # Check how many teachers have seen this token
                for teacher_dataset in self.teacher_datasets:
                    teacher_loader = DataLoader(teacher_dataset, batch_size=16, shuffle=False)
                    for teacher_batch in teacher_loader:
                        if token_id in teacher_batch['input_ids'].unique():
                            token_counts[token_id] += 1
                            break
                            
        return token_counts
    
    def generate_embeddings(self, public_dataset, student_model):
        """
        Generate aggregated embeddings with DP noise for student distillation.
        
        Args:
            public_dataset: Unlabeled dataset for distillation
            student_model: Student model to be trained
        
        Returns:
            Dictionary mapping input_ids to aggregated teacher embeddings
        """
        # First get token frequency
        token_counts = self.get_token_frequency(public_dataset)
        
        # Create dataloader for public dataset
        public_loader = DataLoader(
            public_dataset,
            batch_size=self.config.batch_size,
            shuffle=False
        )
        
        # Extract BERT embedding layer from the student model
        student_embeddings = student_model.bert.embeddings.word_embeddings.weight.data
        
        # Dictionary to store aggregated embeddings
        aggregated_embeddings = {}
        
        # Process batches
        for batch in tqdm(public_loader, desc="Generating teacher embeddings"):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            
            # Get embeddings from each teacher
            teacher_outputs = []
            for teacher in self.teacher_models:
                teacher.eval()
                with torch.no_grad():
                    # Extract the embedding layer output
                    embedding_output = teacher.bert.embeddings.word_embeddings(input_ids)
                    teacher_outputs.append(embedding_output)
            
            # Stack and average teacher outputs
            stacked_outputs = torch.stack(teacher_outputs)  # [num_teachers, batch_size, seq_len, embed_dim]
            mean_output = stacked_outputs.mean(dim=0)  # [batch_size, seq_len, embed_dim]
            
            # Apply DP noise based on token frequency
            for i in range(input_ids.size(0)):  # For each item in batch
                for j in range(input_ids.size(1)):  # For each token position
                    token_id = input_ids[i, j].item()
                    if token_id == 0:  # Skip padding tokens
                        continue
                        
                    # Check if token is rare
                    token_freq = token_counts.get(token_id, 0)
                    is_rare = token_freq < self.config.rare_token_threshold
                    
                    # Apply noise based on token rarity
                    noise_scale = self.config.teacher_aggregation_noise
                    if is_rare:
                        noise_scale *= self.config.rare_token_noise_factor
                        
                    # Generate Gaussian noise
                    noise = torch.randn_like(mean_output[i, j]) * noise_scale
                    
                    # If extremely rare, use student's pre-trained embedding instead
                    if token_freq == 0:
                        aggregated_embeddings[token_id] = student_embeddings[token_id].clone().to(self.device)
                    else:
                        # Add noise to create DP embedding
                        dp_embedding = mean_output[i, j] + noise
                        aggregated_embeddings[token_id] = dp_embedding
        
        return aggregated_embeddings

    def private_ensemble_prediction(self, dataloader, noise_multiplier):
        """
        Generate differentially private predictions from the teacher ensemble for PATE.
        
        Args:
            dataloader: DataLoader for the public/unlabeled dataset
            noise_multiplier: Noise multiplier for the Gaussian mechanism
            
        Returns:
            Private predictions with DP guarantee
        """
        from torch.nn.functional import softmax
        
        all_private_labels = []
        
        for batch in tqdm(dataloader, desc="Generating private ensemble predictions"):
            batch = {k: v.to(self.device) for k, v in batch.items() if k != 'idx'}
            
            # Get predictions from each teacher
            teacher_predictions = []
            for teacher in self.teacher_models:
                teacher.eval()
                with torch.no_grad():
                    if self.config.dataset_name == "conll2003":
                        outputs = teacher(
                            input_ids=batch['input_ids'],
                            attention_mask=batch['attention_mask'],
                            token_type_ids=batch['token_type_ids'] if 'token_type_ids' in batch else None
                        )
                    else:
                        outputs = teacher(
                            input_ids=batch['input_ids'],
                            attention_mask=batch['attention_mask'],
                            token_type_ids=batch['token_type_ids'] if 'token_type_ids' in batch else None
                        )
                    
                    # Get class probabilities
                    logits = outputs[1] if isinstance(outputs, tuple) else outputs
                    probs = softmax(logits, dim=-1)
                    teacher_predictions.append(probs)
            
            # Stack teacher predictions
            stacked_preds = torch.stack(teacher_predictions, dim=0)  # [num_teachers, batch_size, num_classes]
            
            # Compute vote counts for each class
            votes = stacked_preds.sum(dim=0)  # [batch_size, num_classes]
            
            # Add DP noise
            noise = torch.randn_like(votes) * noise_multiplier * self.config.teacher_aggregation_noise
            noisy_votes = votes + noise
            
            # Get most voted class
            private_labels = torch.argmax(noisy_votes, dim=-1)
            all_private_labels.append(private_labels)
            
        return torch.cat(all_private_labels, dim=0)


def distill_embeddings(teacher_ensemble, student_model, public_dataset, config, device):
    """
    Train the student model by distilling knowledge from teacher embeddings.
    
    Args:
        teacher_ensemble: Ensemble of teacher models
        student_model: Student model to be trained
        public_dataset: Unlabeled dataset for distillation
        config: Configuration object
        device: Device to run on
        
    Returns:
        Trained student model
    """
    print("Starting the embedding distillation process...")
    
    # Generate aggregated teacher embeddings with DP
    teacher_embeddings = teacher_ensemble.generate_embeddings(public_dataset, student_model)
    
    # Set up optimizer
    optimizer = AdamW(
        student_model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    # Training loop
    student_model.train()
    for epoch in range(config.num_epochs):
        epoch_loss = 0
        
        # Iterate through token embeddings (in random order for better training)
        token_ids = list(teacher_embeddings.keys())
        random.shuffle(token_ids)
        
        for token_id in tqdm(token_ids, desc=f"Distillation Epoch {epoch+1}"):
            # Get student embedding for this token
            student_embedding = student_model.bert.embeddings.word_embeddings.weight[token_id].clone().to(device)
            
            # Get teacher embedding
            teacher_embedding = teacher_embeddings[token_id]
            
            # Compute MSE loss
            loss = nn.MSELoss()(student_embedding, teacher_embedding)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(student_model.parameters(), config.max_grad_norm)
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(teacher_embeddings)
        print(f"Distillation Epoch {epoch+1} average loss: {avg_loss:.4f}")
    
    return student_model


def pate_distillation(teacher_ensemble, student_model, public_dataset, train_loader, config, device):
    """
    Train the student model using PATE-style distillation.
    
    Args:
        teacher_ensemble: Ensemble of teacher models
        student_model: Student model to be trained
        public_dataset: Unlabeled dataset for distillation
        train_loader: DataLoader for the labeled training dataset
        config: Configuration object
        device: Device to run on
        
    Returns:
        Trained student model
    """
    print("Starting PATE-style distillation...")
    
    # Create dataloader for public dataset
    public_loader = DataLoader(
        public_dataset,
        batch_size=config.batch_size,
        shuffle=True
    )
    
    # Generate private labels from teacher ensemble
    private_labels = teacher_ensemble.private_ensemble_prediction(
        public_loader, 
        config.noise_multiplier
    )
    
    # Create a new dataloader with public dataset and private labels
    from torch.utils.data import TensorDataset
    
    # Collect public dataset features
    all_input_ids = []
    all_attention_masks = []
    all_token_type_ids = []
    
    for batch in public_loader:
        all_input_ids.append(batch['input_ids'])
        all_attention_masks.append(batch['attention_mask'])
        if 'token_type_ids' in batch:
            all_token_type_ids.append(batch['token_type_ids'])
    
    all_input_ids = torch.cat(all_input_ids, dim=0)
    all_attention_masks = torch.cat(all_attention_masks, dim=0)
    
    if len(all_token_type_ids) > 0:
        all_token_type_ids = torch.cat(all_token_type_ids, dim=0)
        private_dataset = TensorDataset(
            all_input_ids, all_attention_masks, all_token_type_ids, private_labels
        )
    else:
        private_dataset = TensorDataset(
            all_input_ids, all_attention_masks, private_labels
        )
    
    private_loader = DataLoader(
        private_dataset,
        batch_size=config.batch_size,
        shuffle=True
    )
    
    # Set up optimizer and scheduler
    optimizer = AdamW(
        student_model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    total_steps = len(private_loader) * config.num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=total_steps
    )
    
    # Training loop
    student_model.train()
    for epoch in range(config.num_epochs):
        epoch_loss = 0
        
        for batch in tqdm(private_loader, desc=f"PATE Distillation Epoch {epoch+1}"):
            # Move batch to device
            if len(batch) == 4:  # With token_type_ids
                input_ids, attention_mask, token_type_ids, labels = [t.to(device) for t in batch]
            else:  # Without token_type_ids
                input_ids, attention_mask, labels = [t.to(device) for t in batch]
                token_type_ids = None
            
            # Forward pass
            outputs = student_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                labels=labels
            )
            
            loss = outputs[0] if isinstance(outputs, tuple) else outputs.loss
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(student_model.parameters(), config.max_grad_norm)
            optimizer.step()
            scheduler.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(private_loader)
        print(f"PATE Distillation Epoch {epoch+1} average loss: {avg_loss:.4f}")
    
    return student_model