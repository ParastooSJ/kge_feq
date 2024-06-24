# config.py
import torch
import logging

class Config:
    logger = logging.getLogger("answer_selection")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    batch_size = 16
    dataset = ""
    seed = 42
    doc_no = 100
    MAX_SEQ_LENGTH = 200
    cache_dir = "../cache/"
    data_dir = "../data/"
    model_dir = "../model/"+dataset+"/answer_selection_model.pth"
    test_data_path = data_dir+dataset+"/scored_test.json"
    train_data_path = data_dir+dataset+"/train_sample.json"
    output_file_path = data_dir+dataset+"/top100-results.txt"
    

    max_seq_length = 200
    num_labels = 2  # Example: change based on your model's specifics
    logging_dir = 'logs/'
    samples = 50
    BATCH_SIZE = 32
    WARMUP_PROPORTION =  0.1

    def __init__(self,dataset):
        self.dataset = dataset 
        self.model_dir = "../model/"+self.dataset+"/answer_selection_model.pth"
        self.test_data_path = self.data_dir+self.dataset+"/test/scored_test.json"
        self.train_data_path = self.data_dir+self.dataset+"/train/train_sample.json"
        self.output_file_path = self.data_dir+self.dataset+"/test/top100-results.txt"
        torch.manual_seed(self.seed)
        if self.n_gpu > 0:
            torch.cuda.manual_seed_all(self.seed)
