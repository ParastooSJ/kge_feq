import torch

class Config:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    train_batch_size = 16
    test_batch_size = 1000
    seed = 42
    warmup_proportion = 0.1
    cache_dir = "../cache"
    loss_scale = 0.0
    max_seq_length = 200
    train_data_path = '../data/train/train_sample.json'
    test_data_path = '../data/test/test_sample.json'
    output_path = '../data/test/scored_file.json'
