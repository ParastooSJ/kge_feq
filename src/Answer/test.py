import torch
from Answer.model import BertForAnswerSelection
from Answer.config import Config
from torch.utils.data import DataLoader
import numpy as np
import json
from transformers import BertTokenizer,BertModel,AdamW
from Answer.data_loader import DataProcessor,convert_examples_to_features


def load_model(model_path,config):
    # Load the trained model from file
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertForAnswerSelection.from_pretrained('bert-base-uncased',
                                                   state_dict=torch.load(model_path))
    model.to(config.device)
    return model, tokenizer


def test(model, tokenizer, config, test_data_path, output_file_path):
    """Function to test the model on the given dataset."""
    
    counter = 0
    total_count = 0

    model.eval()

    with open(test_data_path, 'r') as file:
        data = json.load(file)
    
    with open(output_file_path, 'w') as f_out:
        for entry in data:
            total_count += 1
            print(f"Processing entry {total_count}...")
            entry['triples'] = entry['triples'][:config.doc_no]  # Assuming 'doc_no' is defined in Config
            test_examples = DataProcessor().get_test_examples(entry)
            test_features = convert_examples_to_features(test_examples, config.max_seq_length, tokenizer)

            if len(test_examples)>0:
                # Preparing the tensor dataset

                input_ids = torch.tensor([f.input_ids for f in test_features], dtype=torch.long).to(config.device)
                input_mask = torch.tensor([f.input_mask for f in test_features], dtype=torch.long).to(config.device)
                all_subject_ids = torch.tensor([f.subject_ids for f in test_features], dtype=torch.long).to(config.device)
                all_subject_mask = torch.tensor([f.subject_mask for f in test_features], dtype=torch.long).to(config.device)
                all_output_ids = torch.tensor([f.output_ids for f in test_features], dtype=torch.long).to(config.device)
                all_output_mask = torch.tensor([f.output_mask for f in test_features], dtype=torch.long).to(config.device)
                score = torch.tensor([f.score for f in test_features], dtype=torch.float).to(config.device)

                with torch.no_grad():
                        neg = model(input_ids, attention_mask=input_mask,subject_ids=all_subject_ids,subject_mask=all_subject_mask,output_ids=all_output_ids,output_attention_mask=all_output_mask)
                neg = neg.cpu().detach().numpy()
                score = score.cpu().detach().numpy()
                answer = False
                max_score = float('-inf')


                # Evaluate and update entries
                for triple, score in zip(entry["triples"], neg):
                    
                    if score > max_score:
                        answer  = triple['answer']
                        max_score = score
                        
                    triple["ranking_score"] = score.item()
                    
                
                if answer:
                        counter += 1
                sorted_list = sorted(entry["triples"], key=lambda x: x["ranking_score"], reverse=True)
                entry["triples"] = sorted_list

                # Sort and store the result
                data_new ={'question':entry['question'],'answer':entry['answer'],'triples':entry["triples"]}
                f_out.write(json.dumps(data_new)+"\n")
                if answer==False and entry["triples"][0]['object'] in entry['answer'] :
                    data_new ={'question':entry['question'],'answer':entry['answer'],'triples':entry["triples"][0]}
                 

        # Write all results to file at once at the end
        accuracy = counter / total_count
        print(f"Overall Accuracy: {accuracy * 100:.2f}%")
        f_out.write(f"Overall accuracy: {accuracy * 100:.2f}%\n")
