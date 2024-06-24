import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import pandas as pd
import json
from pytorch_pretrained_bert.tokenization import BertTokenizer

class DataProcessor:
    """Process input data for training/testing."""

    def __init__(self, tokenizer, max_seq_length):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def prepare_features(self, examples):
        """Convert list of `InputExample`s to list of tensors suitable for the model."""
        features = []
        for example in examples:
            tokens = self.tokenizer.tokenize(example.text)
            if len(tokens) > self.max_seq_length - 2:
                tokens = tokens[:(self.max_seq_length - 2)]
            tokens = ['[CLS]'] + tokens + ['[SEP]']
            segment_ids = [0] * len(tokens)
            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding = [0] * (self.max_seq_length - len(input_ids))
            input_ids += padding
            input_mask += padding
            segment_ids += padding

            features.append((input_ids, input_mask, segment_ids))
        return features

    def get_data_loader(self, data_file, batch_size, is_training=True):
        """Create DataLoader from a data file."""
        with open(data_file, 'r') as file:
            data = json.load(file)

        examples = [InputExample(**item) for item in data['data']]
        features = self.prepare_features(examples)

        all_input_ids = torch.tensor([f[0] for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f[1] for f in features], dtype=torch.long)
        all_segment_ids = torch.tensor([f[2] for f in features], dtype=torch.long)

        if is_training:
            all_scores = torch.tensor([example.score for example in examples], dtype=torch.float)
            dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_scores)
        else:
            dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids)

        sampler = RandomSampler(dataset) if is_training else SequentialSampler(dataset)
        return DataLoader(dataset, sampler=sampler, batch_size=batch_size)
