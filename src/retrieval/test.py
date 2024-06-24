import torch
import logging
from data_loader import create_data_loader
from model import BertForSequenceRegression
from config import Config

# Function to test the model
def test_model(model, dataloader):
    model.eval()
    all_scores = []
    for step, batch in enumerate(dataloader):
        batch = tuple(t.to(Config.device) for t in batch)
        input_ids, input_mask, segment_ids = batch
        with torch.no_grad():
            outputs = model(input_ids, segment_ids, input_mask)
        all_scores.extend(outputs.squeeze().tolist())
    return all_scores
