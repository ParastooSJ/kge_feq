# train.py
from Answer.model import BertForAnswerSelection
from Answer.config import Config
import torch.optim as optim
import torch.nn as nn
from Answer.train_utils import get_optimizer
from fastprogress import master_bar, progress_bar
from Answer.data_loader import DataProcessor,convert_examples_to_features
import gc
import sys
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from Answer.train_utils import set_trainable, count_model_parameters
from transformers import EncoderDecoderModel, BertTokenizer,BertModel,AdamW,get_scheduler

def train(model: nn.Module, config, train_dataloader, num_epochs: int, learning_rate: float):
    
    num_train_optimization_steps = len(train_dataloader) * num_epochs 
    optimizer = get_optimizer(num_train_optimization_steps, learning_rate)
    assert all([x["lr"] == learning_rate for x in optimizer.param_groups])
    global_step = 0
    nb_tr_steps = 0
    tr_loss = 0  
    model.train()
    mb = master_bar(range(num_epochs))
    tr_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0    
    for _ in mb:
        for step, batch in enumerate(progress_bar(train_dataloader, parent=mb)):
            batch = tuple(t.to(config.device) for t in batch)
            b_all_input_ids,b_all_input_masks,b_all_subject_ids,b_all_subject_masks,b_all_output_ids,b_all_output_masks,score = batch
            loss = model(b_all_input_ids, attention_mask=b_all_input_masks,subject_ids=b_all_subject_ids,subject_mask=b_all_subject_masks,output_ids=b_all_output_ids,output_attention_mask=b_all_output_masks,targets=score)
            if config.n_gpu > 1:
                loss = loss.mean() 
            loss.backward()
            if tr_loss == 0:
                tr_loss = loss.item()
            else:
                tr_loss = tr_loss * 0.9 + loss.item() * 0.1
            nb_tr_examples += b_all_input_ids.size(0)
            nb_tr_steps += 1
            

            optimizer.step()
            optimizer.zero_grad()
            global_step += 1
            mb.child.comment = f'loss: {tr_loss:.4f} lr: {optimizer.get_lr()[0]:.2E}'
    config.logger.info("  train loss = %.4f", tr_loss) 
    return tr_loss



def train_model(config):
    
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertForAnswerSelection.from_pretrained('bert-base-uncased')
    model.to(config.device)

    train_examples = DataProcessor().get_train_examples(config)
    train_features = convert_examples_to_features(train_examples, config.MAX_SEQ_LENGTH, tokenizer)
    del train_examples
    gc.collect()
   

    all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
    all_subject_ids = torch.tensor([f.subject_ids for f in train_features], dtype=torch.long)
    all_subject_mask = torch.tensor([f.subject_mask for f in train_features], dtype=torch.long)
    all_output_ids = torch.tensor([f.output_ids for f in train_features], dtype=torch.long)
    all_output_mask = torch.tensor([f.output_mask for f in train_features], dtype=torch.long)
    all_score = torch.tensor([f.score for f in train_features], dtype=torch.float)
    train_data = TensorDataset(all_input_ids, all_input_mask,all_subject_ids,all_subject_mask,all_output_ids, all_output_mask,all_score)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=config.BATCH_SIZE)

    set_trainable(model, True)
    set_trainable(model.bert_relation.embeddings, False)
    set_trainable(model.bert_relation.encoder, False)
    set_trainable(model.bert_subject.embeddings, False)
    set_trainable(model.bert_subject.encoder, False)
    set_trainable(model.bert_object.embeddings, False)
    set_trainable(model.bert_object.encoder, False)
    count_model_parameters(model,config)
    train(model,config, train_dataloader, num_epochs = 2, learning_rate = 5e-4)
    

    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
    torch.save(model_to_save.state_dict(), config.model_dir)
    gc.collect()



    set_trainable(model.bert_relation.encoder.layer[11], True)
    set_trainable(model.bert_relation.encoder.layer[10], True)
    set_trainable(model.bert_subject.encoder.layer[11], True)
    set_trainable(model.bert_subject.encoder.layer[10], True)
    set_trainable(model.bert_object.encoder.layer[11], True)
    set_trainable(model.bert_object.encoder.layer[10], True)
    count_model_parameters(model,config)
    train(model,config,train_dataloader, num_epochs = 2, learning_rate = 5e-5)
  

    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
    torch.save(model_to_save.state_dict(), config.model_dir)

    set_trainable(model, True)
    count_model_parameters(model,config)
    train(model,config,train_dataloader, num_epochs = 1, learning_rate = 1e-5)

    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
    torch.save(model_to_save.state_dict(), config.model_dir)
    gc.collect()



