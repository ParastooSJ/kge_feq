import logging

import sys


from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from pytorch_pretrained_bert.modeling import BertPreTrainedModel, BertModel
from pytorch_pretrained_bert.tokenization import BertTokenizer

import random

import json
import pandas as pd
import numpy as np
from torch import linalg as LA
import gc
import torch
import torch.nn as nn
from pytorch_pretrained_bert import BertForSequenceClassification
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from pytorch_pretrained_bert.modeling import BertPreTrainedModel ,BertConfig
from pytorch_pretrained_bert.optimization import BertAdam, WarmupLinearSchedule, SCHEDULES
from fastprogress import master_bar, progress_bar

from sentence_transformers import SentenceTransformer, util
from transformers import EncoderDecoderModel, BertTokenizer,BertModel,AdamW,get_scheduler


from tqdm.auto import tqdm

import math


FP16 = False
BATCH_SIZE = 32
SEED = 42
WARMUP_PROPORTION = 0.1
PYTORCH_PRETRAINED_BERT_CACHE = "../cache"
LOSS_SCALE = 0.
MAX_SEQ_LENGTH = 200
logger = logging.getLogger("SQ-50-bert-regressor")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
logger.info("device: {} n_gpu: {}, 16-bits training: {}".format(device, n_gpu, FP16))
random.seed(SEED)
np.random.seed(SEED)
dataset = "../data/"+sys.argv[1]+"/"
torch.manual_seed(SEED)
if n_gpu > 0:
    torch.cuda.manual_seed_all(SEED)


class BertForSequenceRegression(BertPreTrainedModel):

    def __init__(self,config):
        super(BertForSequenceRegression, self).__init__(config)
        self.linear_subject = nn.Linear(50,768)
        self.linear_object = nn.Linear(50,768)
        self.bert_subject = BertModel.from_pretrained('bert-base-uncased')
        self.bert_relation = BertModel.from_pretrained('bert-base-uncased')
        self.bert_object = BertModel.from_pretrained('bert-base-uncased')
        self.loss_fct = torch.nn.MSELoss()
        
        #self.linear_layer = nn.Linear(in_features=1, out_features=1)

    def forward(self, input_ids, attention_mask,subject_ids,subject_mask,output_ids,output_attention_mask,targets=None):

        subject = self.bert_subject(subject_ids,subject_mask).pooler_output
        objectt = self.bert_object(output_ids,output_attention_mask).pooler_output
        relation = self.bert_relation(input_ids, attention_mask).pooler_output

        subject = torch.FloatTensor(subject.to('cpu')).to('cuda')
        objectt = torch.FloatTensor(objectt.to('cpu')).to('cuda')
        relation = torch.FloatTensor(relation.to('cpu')).to('cuda')

        zero = torch.zeros(subject.shape).to('cuda')
        
        Es = LA.norm(subject+relation-objectt,dim=1)
        
        Es = Es+ 0.0001
        #Es = Es.unsqueeze(-1)
        

        #scaled_scores = self.linear_layer(Es.unsqueeze(1))
        #scaled_scores = torch.tanh(scaled_scores)
        #Es = scaled_scores.squeeze(1)
        
        #print(Es)
        #score = -1 * Es
        #print(score)
        
        
        score = -1 * torch.log(Es)
        
        
        if targets is not None:
            loss = self.loss_fct(score.view(-1), targets.view(-1))
            loss = torch.FloatTensor(loss.to('cpu')).to('cuda')

            return loss
        else:
            return score 



class InputExample(object):

    def __init__(self,question,subject,answer,score):

        self.question = question
        self.subject = subject
        self.answer = answer
        self.score = score


class InputFeatures(object):
    def __init__(self,input_ids,input_mask,subject_ids,subject_mask,output_ids,output_mask,score):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.output_ids = output_ids
        self.output_mask = output_mask
        self.subject_ids = subject_ids
        self.subject_mask = subject_mask
        self.score = score



class DataProcessor:

    
    def __init__(self):

        x=2

    def get_train_examples(self):
        question = []
        subject = []
        answer = []
        score = []
        for line in data_train:
            #line= json.loads(line)
            
            for triple in line['triples'][:50]:
                
                   question.append(line['question'][:100]+" "+triple['relation'])
                   subject.append(triple['subject'])
                   answer.append(triple['object'])
                   
                   #score.append(triple['relation_object_score'])
                   #score.append(1 if triple['answer'] else -1)
                   score.append((0.3*triple['relation_object_score']) +triple['relation_object_score'] if triple['answer'] else (-0.3*triple['relation_object_score'])+triple['relation_object_score'])
        score=(score-np.min(score))/(np.max(score)-np.min(score))
        return self._create_examples(question,subject,answer,score)

        

    def get_test_examples(self,data):

        question = []
        subject = []
        answer = []
        score = []
        value = data['triples']
        
        for triple in value:
            question.append(data['question'][:100]+" "+triple['relation'])
            answer.append(triple['object'])
            subject.append(triple['subject'])
            
            #score.append(triple['prune_score'])
            score.append(1 if triple['answer'] else -1)
        return self._create_examples(question,subject,answer,score)



    def _create_examples(self,question,subject,answer,score):
 
        examples = []
        for (i, (question,subject,answer,score)) in enumerate(zip(question,subject,answer,score)):
            examples.append(InputExample(question=question,subject=subject,answer=answer,score=score))

        return examples


n = 0
def convert_examples_to_features(examples, max_seq_length, tokenizer):
    
    features = []
    for (ex_index, example) in enumerate(examples):

        #-----------------------------question------------------------------

        question_input_ids = tokenizer.encode(example.question,add_special_tokens=True, max_length=max_seq_length,truncation=True)
        if len(question_input_ids) > max_seq_length:
            question_input_ids = question_input_ids[:(max_seq_length)]
        question_input_mask = [1] * len(question_input_ids)
        padding = [0] * (max_seq_length - len(question_input_ids))
        question_input_ids += padding
        question_input_mask += padding

        #-----------------------------subject----------------------------

        subject_ids = tokenizer.encode(example.subject,add_special_tokens=True, max_length=max_seq_length,truncation=True)
        if len(subject_ids) > max_seq_length:
            subject_ids = subject_ids[:(max_seq_length)]
        subject_mask = [1] * len(subject_ids)
        padding = [0] * (max_seq_length - len(subject_ids))
        subject_ids += padding
        subject_mask += padding
        
            


        #-----------------------------answer------------------------------

        answer_input_ids = tokenizer.encode(example.answer,add_special_tokens=True, max_length=max_seq_length,truncation=True)
        if len(answer_input_ids) > max_seq_length:
            answer_input_ids = answer_input_ids[:(max_seq_length)]
        answer_input_mask = [1] * len(answer_input_ids)
        padding = [0] * (max_seq_length - len(answer_input_ids))
        answer_input_ids += padding
        answer_input_mask += padding
        
        


        #--------------------------all to gether----------------------------

        input_ids=question_input_ids
        input_mask=question_input_mask
        output_ids=answer_input_ids
        output_mask=answer_input_mask
        
        features.append(

                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              output_ids=output_ids,
                              output_mask=output_mask,
                              subject_ids=subject_ids,
                              subject_mask=subject_mask,
                              score=example.score))

    return features

class FreezableBertAdam(BertAdam):
    def get_lr(self):
        lr = []
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                if len(state) == 0:
                    continue
                lr_scheduled = group['lr']
                lr.append(lr_scheduled)
        return lr 

def children(m):
    return m if isinstance(m, (list, tuple)) else list(m.children())

def set_trainable_attr(m, b):
    m.trainable = b
    for p in m.parameters():
        p.requires_grad = b

def apply_leaf(m, f):
    c = children(m)
    if isinstance(m, nn.Module):
        f(m)
    if len(c) > 0:
        for l in c:
            apply_leaf(l, f)

def set_trainable(l, b):
    apply_leaf(l, lambda m: set_trainable_attr(m, b))

def count_model_parameters(model):
    logger.info(
        "# of paramters: {:,d}".format(
            sum(p.numel() for p in model.parameters())))
    logger.info(
        "# of trainable paramters: {:,d}".format(
            sum(p.numel() for p in model.parameters() if p.requires_grad)))

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceRegression.from_pretrained('bert-base-uncased')
model.to(device)
param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]

def get_optimizer(num_train_optimization_steps: int, learning_rate: float):
    grouped_parameters = [
       x for x in optimizer_grouped_parameters if any([p.requires_grad for p in x["params"]])
    ]
    for group in grouped_parameters:
        group['lr'] = learning_rate
    if FP16:
        try:
            from apex.optimizers import FP16_Optimizer
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex "
                              "to use distributed and fp16 training.")
        optimizer = FusedAdam(grouped_parameters,
                              lr=learning_rate, bias_correction=False,
                              max_grad_norm=1.0)
        if args.loss_scale == 0:
            optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
        else:
            optimizer = FP16_Optimizer(optimizer, static_loss_scale=LOSS_SCALE)
    else:
        optimizer = FreezableBertAdam(grouped_parameters,
                             lr=learning_rate, warmup=WARMUP_PROPORTION,
                             t_total=num_train_optimization_steps)

    return optimizer


def train(model: nn.Module,train_dataloader, num_epochs: int, learning_rate: float):
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
            batch = tuple(t.to(device) for t in batch)
            b_all_input_ids,b_all_input_masks,b_all_subject_ids,b_all_subject_masks,b_all_output_ids,b_all_output_masks,score = batch
            loss = model(b_all_input_ids, attention_mask=b_all_input_masks,subject_ids=b_all_subject_ids,subject_mask=b_all_subject_masks,output_ids=b_all_output_ids,output_attention_mask=b_all_output_masks,targets=score)
            if n_gpu > 1:
                loss = loss.mean() 
            if FP16:
                optimizer.backward(loss)
            else:
                loss.backward()
            if tr_loss == 0:
                tr_loss = loss.item()
            else:
                tr_loss = tr_loss * 0.9 + loss.item() * 0.1
            nb_tr_examples += b_all_input_ids.size(0)
            nb_tr_steps += 1
            if FP16:
                lr_this_step = (
                     LR * warmup_linear(global_step/num_train_optimization_steps, WARMUP_PROPORTION))
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_this_step

            optimizer.step()
            optimizer.zero_grad()
            global_step += 1
            mb.child.comment = f'loss: {tr_loss:.4f} lr: {optimizer.get_lr()[0]:.2E}'
    logger.info("  train loss = %.4f", tr_loss) 
    return tr_loss

train_dir = dataset + "train/train_sample.json"
data_train = json.load(open(train_dir, "r"))

def train_model():
    
   
    train_examples = DataProcessor().get_train_examples()
    train_features = convert_examples_to_features(train_examples, MAX_SEQ_LENGTH, tokenizer)
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
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=BATCH_SIZE)

    set_trainable(model, True)
    set_trainable(model.bert_relation.embeddings, False)
    set_trainable(model.bert_relation.encoder, False)
    set_trainable(model.bert_subject.embeddings, False)
    set_trainable(model.bert_subject.encoder, False)
    set_trainable(model.bert_object.embeddings, False)
    set_trainable(model.bert_object.encoder, False)
    count_model_parameters(model)
    train(model,train_dataloader, num_epochs = 2, learning_rate = 5e-4)

    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
    output_model_file = "../model/"+sys.argv[1]+"/embedding_transe_50.pth"
    torch.save(model_to_save.state_dict(), output_model_file)
    #model.load_state_dict(torch.load("../model/"+sys.argv[1]+"/embedding_transe_50.pth"))

    gc.collect()



    set_trainable(model.bert_relation.encoder.layer[11], True)
    set_trainable(model.bert_relation.encoder.layer[10], True)
    set_trainable(model.bert_subject.encoder.layer[11], True)
    set_trainable(model.bert_subject.encoder.layer[10], True)
    set_trainable(model.bert_object.encoder.layer[11], True)
    set_trainable(model.bert_object.encoder.layer[10], True)
    count_model_parameters(model)
    train(model,train_dataloader, num_epochs = 2, learning_rate = 5e-5)
  

    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
    output_model_file = "../model/"+sys.argv[1]+"/embedding_transe_50.pth"
    torch.save(model_to_save.state_dict(), output_model_file)
    #model.load_state_dict(torch.load("../model/"+sys.argv[1]+"/embedding_transe_50.pth"))


    set_trainable(model, True)
    count_model_parameters(model)
    train(model,train_dataloader, num_epochs = 1, learning_rate = 1e-5)

    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
    output_model_file = "../model/"+sys.argv[1]+"/embedding_transe_50.pth"
    torch.save(model_to_save.state_dict(), output_model_file)
    gc.collect()





def test(model,write=False):
    doc_no = 100
    test_dir = dataset + "test/scored_test.json"
    output_test_file = dataset + 'test/resuls'+str(doc_no)+'.txt'
    f_out = open(output_test_file,'w')  


    #model = BertForSequenceRegression.from_pretrained('bert-base-uncased',cache_dir=PYTORCH_PRETRAINED_BERT_CACHE)
    #model.load_state_dict(torch.load("../model/"+sys.argv[1]+"/embedding_transe_prune_second.pth"))
    #model.to(device)
   
    model.eval()
    BATCH_SIZE=100
    counter=0
    total_count=0
    model.eval()
    in_f = []
    with open(test_dir, 'r') as f:
        f_j = json.load(f)
        for line in f_j:
            total_count += 1
            print(total_count)
            data= line #json.loads(line)
            data['triples'] = line['triples'][:doc_no]
            test_examples = DataProcessor().get_test_examples(data)
            test_features = convert_examples_to_features(test_examples, MAX_SEQ_LENGTH, tokenizer)
            if len(test_examples)!=0:
                all_input_ids = torch.tensor([f.input_ids for f in test_features], dtype=torch.long)
                all_input_mask = torch.tensor([f.input_mask for f in test_features], dtype=torch.long)
                all_subject_ids = torch.tensor([f.subject_ids for f in test_features], dtype=torch.long)
                all_subject_mask = torch.tensor([f.subject_mask for f in test_features], dtype=torch.long)
                all_output_ids = torch.tensor([f.output_ids for f in test_features], dtype=torch.long)
                all_output_mask = torch.tensor([f.output_mask for f in test_features], dtype=torch.long)
                all_score = torch.tensor([f.score for f in test_features], dtype=torch.float)
                test_data = TensorDataset(all_input_ids, all_input_mask,all_subject_ids,all_subject_mask, all_output_ids, all_output_mask,all_score)
                test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE)#not used each line is has 100 sample so it is batched naturally

                

                input_ids = all_input_ids.to(device)
                input_mask = all_input_mask.to(device)
                all_subject_ids = all_subject_ids.to(device)
                all_subject_mask = all_subject_mask.to(device)
                all_output_ids = all_output_ids.to(device)
                all_output_mask = all_output_mask.to(device)
                score = all_score.to(device)
                with torch.no_grad():
                    neg = model(input_ids, attention_mask=input_mask,subject_ids=all_subject_ids,subject_mask=all_subject_mask,output_ids=all_output_ids,output_attention_mask=all_output_mask)
                neg = neg.cpu().detach().numpy()
                score = score.cpu().detach().numpy()
                
                answer = False
                max_score = float('-inf')
                
                
                for triple,ne in zip(data["triples"],neg):
                    score = ne.item() #+  math.log(doc_no-15)*0.005 *triple["relation_object_score"] #0.5*triple["relation_score"]
                    print(score)
                    #if ne.item() > max_score:
                    if score > max_score:
                        answer  = triple['answer']
                        max_score = score
                        
                    triple["ranking_score"]=ne.item()
                    #triple["ranking_score"] = 0.2*triple["ranking_score"]+0.8*triple["prune_score"]
                if answer:
                    counter += 1
                sorted_list = sorted(data["triples"], key=lambda x: x["ranking_score"], reverse=True)
                data["triples"]=sorted_list
                

                data_new ={'question':data['question'],'answer':data['answer'],'triples':data["triples"]}

                

                if write:
                    f_out.write(json.dumps(data_new)+"\n")
                if answer==False and data["triples"][0]['object'] in data['answer'] :
                    data_new ={'question':data['question'],'answer':data['answer'],'triples':data["triples"][0]}
                    
     
    print(counter)
    if write:
        f_out.write(str(counter/total_count *100))
        f_out.close()
    print(counter/total_count *100)
    
    return counter/total_count *100

    



train_model()
model = BertForSequenceRegression.from_pretrained('bert-base-uncased',cache_dir=PYTORCH_PRETRAINED_BERT_CACHE)
model.load_state_dict(torch.load("../model/"+sys.argv[1]+"/embedding_transe_50.pth"), strict=False)
model.to(device)
test(model, write=True)




