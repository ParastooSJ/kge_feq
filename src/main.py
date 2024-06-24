# main.py
import sys
import os
from Answer.train import train_model
from Answer.test import test, load_model
from Answer.config import Config

def main():
    if len(sys.argv) != 3:
        print("Usage: python main.py [train/test] [dataset_name]")
        sys.exit(1)

    command = sys.argv[1].lower()
    dataset_name = sys.argv[2]
    config = Config(dataset_name)
    #config.dataset = dataset_name

    if command == 'train':
        print("Starting training process...")
        train_model(config)
        print("Training completed.")
    elif command == 'test':
        
        model_path = config.model_dir
        print(model_path)
        if not os.path.exists(model_path):
            print("Error: Model does not exist. Please train the model first or check the model directory.")
            sys.exit(1)
        print(f"Loading model from: {model_path}")
        model,tokenizer = load_model(model_path,config)
        print("Starting testing process...")
        test(model, tokenizer, config, config.test_data_path, config.output_file_path)  # Use the standardized test data file from config
        print("Testing completed.")
    else:
        print("Invalid command. Use 'train' or 'test'.")

if __name__ == "__main__":
    main()
