# data_loader.py
# Functions for loading and preprocessing datasets

from datasets import load_dataset
from transformers import BertTokenizer
from torch.utils.data import random_split

def load_glue_dataset(config):
    """
    Load datasets from the GLUE benchmark.
    
    Args:
        config: Configuration object
    
    Returns:
        train_dataset, eval_dataset, test_dataset
    """
    # Load dataset using the Hugging Face datasets library
    if config.task_name == "mnli":
        datasets = load_dataset("glue", "mnli")
        # For MNLI, there are two validation sets: matched and mismatched
        train_dataset = datasets["train"]
        eval_dataset = datasets["validation_matched"]
        test_dataset = datasets["validation_mismatched"]
    else:
        datasets = load_dataset("glue", config.task_name)
        train_dataset = datasets["train"]
        if "validation" in datasets:
            eval_dataset = datasets["validation"]
            test_dataset = datasets["test"] if "test" in datasets else datasets["validation"]
        else:
            # Split the training set if no validation set is available
            train_size = int(config.train_test_split_ratio * len(train_dataset))
            eval_size = len(train_dataset) - train_size
            train_dataset, eval_dataset = random_split(
                train_dataset, 
                [train_size, eval_size]
            )
            test_dataset = eval_dataset
    
    # Load tokenizer
    tokenizer = BertTokenizer.from_pretrained(config.bert_model)
    
    # Define preprocessing function
    def tokenize_function(examples):
        if config.task_name == "qqp":
            # For sentence pair tasks
            return tokenizer(
                examples["question1"],
                examples["question2"],
                padding="max_length",
                truncation=True,
                max_length=config.max_seq_length
            )
        elif config.task_name == "mnli":
            # For MNLI sentence pair task
            return tokenizer(
                examples["premise"],
                examples["hypothesis"],
                padding="max_length",
                truncation=True,
                max_length=config.max_seq_length
            )
        else:
            # For single sentence tasks
            text_key = "sentence" if config.task_name == "cola" else "sentence"
            return tokenizer(
                examples[text_key],
                padding="max_length",
                truncation=True,
                max_length=config.max_seq_length
            )
    
    # Apply tokenization
    train_dataset = train_dataset.map(
        tokenize_function,
        batched=True,
        desc="Tokenizing training dataset"
    )
    eval_dataset = eval_dataset.map(
        tokenize_function,
        batched=True,
        desc="Tokenizing evaluation dataset"
    )
    test_dataset = test_dataset.map(
        tokenize_function,
        batched=True,
        desc="Tokenizing test dataset"
    )
    
    # Set format for PyTorch
    train_dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "token_type_ids", "label"]
    )
    eval_dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "token_type_ids", "label"]
    )
    test_dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "token_type_ids", "label"]
    )
    
    return train_dataset, eval_dataset, test_dataset


def load_conll_dataset(config):
    """
    Load the CoNLL-2003 NER dataset.
    
    Args:
        config: Configuration object
    
    Returns:
        train_dataset, eval_dataset, test_dataset
    """
    # Load dataset using the Hugging Face datasets library
    datasets = load_dataset("conll2003")
    train_dataset = datasets["train"]
    eval_dataset = datasets["validation"]
    test_dataset = datasets["test"]
    
    # Load tokenizer
    tokenizer = BertTokenizer.from_pretrained(config.bert_model)
    
    # Define preprocessing function
    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(
            examples["tokens"],
            truncation=True,
            is_split_into_words=True,
            padding="max_length",
            max_length=config.max_seq_length
        )
        
        labels = []
        for i, label in enumerate(examples["ner_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            
            for word_idx in word_ids:
                # Special tokens have a word id that is None
                if word_idx is None:
                    label_ids.append(-100)
                # We set the label for the first token of each word
                elif word_idx != previous_word_idx:
                    label_ids.append(label[word_idx])
                # For the other tokens in a word, we set the label to -100
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx
            
            labels.append(label_ids)
        
        tokenized_inputs["labels"] = labels
        return tokenized_inputs
    
    # Apply tokenization
    train_dataset = train_dataset.map(
        tokenize_and_align_labels,
        batched=True,
        desc="Tokenizing and aligning training dataset"
    )
    eval_dataset = eval_dataset.map(
        tokenize_and_align_labels,
        batched=True,
        desc="Tokenizing and aligning evaluation dataset"
    )
    test_dataset = test_dataset.map(
        tokenize_and_align_labels,
        batched=True,
        desc="Tokenizing and aligning test dataset"
    )
    
    # Set format for PyTorch
    train_dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "token_type_ids", "labels"]
    )
    eval_dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "token_type_ids", "labels"]
    )
    test_dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "token_type_ids", "labels"]
    )
    
    return train_dataset, eval_dataset, test_dataset


def create_public_dataset(config, size=10000):
    """
    Create a public dataset for distillation.
    
    Args:
        config: Configuration object
        size: Size of the public dataset
        
    Returns:
        Public dataset
    """
    # For real implementation, this could be a separate unlabeled dataset
    # For simplicity, we use a subset of the train dataset without labels
    
    if config.dataset_name == "glue":
        dataset = load_dataset("glue", config.task_name)
        public_dataset = dataset["train"].select(range(min(size, len(dataset["train"]))))
    elif config.dataset_name == "conll2003":
        dataset = load_dataset("conll2003")
        public_dataset = dataset["train"].select(range(min(size, len(dataset["train"]))))
    else:
        raise ValueError(f"Unsupported dataset: {config.dataset_name}")
    
    # Load tokenizer
    tokenizer = BertTokenizer.from_pretrained(config.bert_model)
    
    # Tokenize the public dataset
    if config.dataset_name == "glue":
        if config.task_name == "qqp":
            # For sentence pair tasks
            def tokenize_function(examples):
                return tokenizer(
                    examples["question1"],
                    examples["question2"],
                    padding="max_length",
                    truncation=True,
                    max_length=config.max_seq_length
                )
        elif config.task_name == "mnli":
            def tokenize_function(examples):
                return tokenizer(
                    examples["premise"],
                    examples["hypothesis"],
                    padding="max_length",
                    truncation=True,
                    max_length=config.max_seq_length
                )
        else:
            # For single sentence tasks
            text_key = "sentence" if config.task_name == "cola" else "sentence"
            def tokenize_function(examples):
                return tokenizer(
                    examples[text_key],
                    padding="max_length",
                    truncation=True,
                    max_length=config.max_seq_length
                )
    else:  # CoNLL
        def tokenize_function(examples):
            return tokenizer(
                examples["tokens"],
                truncation=True,
                is_split_into_words=True,
                padding="max_length",
                max_length=config.max_seq_length
            )
    
    # Apply tokenization
    public_dataset = public_dataset.map(
        tokenize_function,
        batched=True,
        desc="Tokenizing public dataset"
    )
    
    # Set format for PyTorch (input features only, no labels)
    public_dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "token_type_ids"]
    )
    
    return public_dataset
