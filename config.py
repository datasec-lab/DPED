# config.py
# Configuration for the Differentially Private Embeddings

class Config:
    def __init__(self):
        # Model parameters
        self.bert_model = "bert-base-uncased"
        self.max_seq_length = 128
        self.embedding_dim = 768
        
        # Privacy parameters
        self.epsilon = 8.0
        self.delta = 1e-5
        self.noise_multiplier = 1.0  # Will be calculated based on epsilon
        self.max_grad_norm = 1.0
        self.enable_dp = True
        self.rare_token_threshold = 2
        self.rare_token_noise_factor = 2.0
        self.multi_layer_noise = True
        self.embedding_noise_std = 0.1
        self.intermediate_noise_std = 0.05
        
        # Teacher-Student parameters
        self.num_teachers = 5
        self.teacher_aggregation_noise = 0.5  # Noise for teacher output aggregation
        
        # Training parameters
        self.batch_size = 16
        self.learning_rate = 2e-5
        self.num_epochs = 3
        self.warmup_steps = 0
        self.weight_decay = 0.01
        
        # Dataset parameters
        self.dataset_name = "glue"
        self.task_name = "sst2"  # Options: sst2, qqp, mnli, cola
        self.train_test_split_ratio = 0.8
        
    def update_epsilon(self, new_epsilon):
        """Update epsilon and related parameters"""
        self.epsilon = new_epsilon
        # Noise multiplier could be adjusted based on epsilon
        # This is a simplified relationship - in practice, use the privacy accountant
        if new_epsilon >= 8.0:
            self.noise_multiplier = 0.7
        elif new_epsilon >= 4.0:
            self.noise_multiplier = 1.0
        elif new_epsilon >= 2.0:
            self.noise_multiplier = 1.5
        else:
            self.noise_multiplier = 2.0
