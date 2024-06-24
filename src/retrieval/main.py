import sys
from config import Config
from model import BertForSequenceRegression
from train import train_model
from data_loader import create_data_loader

def main():
    model = BertForSequenceRegression.from_pretrained('bert-base-uncased', cache_dir=Config.cache_dir)
    dataloader = create_data_loader('../data/train/train_sample.json', tokenizer, Config.batch_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
    train_model(model, dataloader, optimizer, epochs=3)

if __name__ == "__main__":
    main()
