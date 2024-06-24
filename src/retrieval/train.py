import torch
import logging
from data_loader import create_data_loader
from model import BertForSequenceRegression

# Training the BERT model for regression
def train_model(model, dataloader, optimizer, epochs):
    model.train()
    for epoch in range(epochs):
        for step, batch in enumerate(dataloader):
            batch = tuple(t.to(Config.device) for t in batch)
            input_ids, input_mask, segment_ids, scores = batch
            model.zero_grad()
            outputs = model(input_ids, segment_ids, input_mask, scores)
            loss = outputs[0]
            loss.backward()
            optimizer.step()
            if step % 50 == 0:
                logger.info("Epoch: {}, Step: {}, Loss: {}".format(epoch, step, loss.item()))
