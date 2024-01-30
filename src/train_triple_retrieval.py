import gc
import logging
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import sys
import json
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from pytorch_pretrained_bert.modeling import BertPreTrainedModel, BertModel
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam, WarmupLinearSchedule, SCHEDULES
from fastprogress import master_bar, progress_bar

import random
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
BATCH_SIZE = 16
SEED = 42
WARMUP_PROPORTION = 0.1
PYTORCH_PRETRAINED_BERT_CACHE = "../cache"
LOSS_SCALE = 0.
MAX_SEQ_LENGTH = 200
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger("Qblink-20-bert-regressor")


class BertForSequenceRegression(BertPreTrainedModel):
    def __init__(self, config):
        super(BertForSequenceRegression, self).__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.regressor = nn.Linear(config.hidden_size, 1)
        self.apply(self.init_bert_weights)
        self.loss_fct = torch.nn.MSELoss()
        

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, targets=None):
        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        pooled_output = self.dropout(pooled_output)
        outputs = self.regressor(pooled_output).clamp(-1, 1)
        if targets is not None:
            
            loss = self.loss_fct(outputs.view(-1), targets.view(-1))
            return loss
        else:
            return outputs


class InputExample(object):

    def __init__(self, question,relation,object,score=None):
        self.question=question
        self.relation=relation
        self.object=object
        self.score=score

class InputFeatures(object):

    def __init__(self, input_ids, input_mask, segment_ids , score):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.score = score

class DataProcessor:
    

    def get_train_examples(self,df_train):
        question_train=[]
        relation_train=[]
        object_train=[]
        score_train=[]

        for value in df_train:
            question = value['question'][:1000].strip()
            triples =  value['triples']

            for triple in triples:
                relation_object_score = 1 if triple["ans"] else 0
                object = triple['object']
                subject = triple['subject']
                relation = triple['relation']

                question_train.append(question)
                relation_train.append(relation[:1000].strip())
                object_train.append(subject[:1000].strip()+relation[:1000].strip()+" "+object[:1000].strip())
                score_train.append(relation_object_score)

        score_train=(score_train-np.min(score_train))/(np.max(score_train)-np.min(score_train))
        return self._create_examples(question_train,relation_train,object_train,score_train)



    def get_test_examples(self,df_test):
        question_test=[]
        relation_test=[]
        object_test=[]
        score_test=[]
        
        question = df_test['question'][:1000].strip()
        triples = df_test['triples']
        for triple in triples:
            relation_object_score = 0
            relation  = triple["relation"]
            subject = triple['subject']
            object = triple['object']
            
            question_test.append(question)
            relation_test.append(relation[:1000].strip())
            object_test.append(subject[:1000].strip()+relation[:1000].strip()+" "+object[:1000].strip())
            score_test.append(relation_object_score)
        
        score_test=(score_test-np.min(score_test))/(np.max(score_test)-np.min(score_test))
        return self._create_examples(question_test,relation_test,object_test,score_test)

    def _create_examples(self,question,relation,object,score):
        examples = []
        for (i, (question,relation,object,score)) in enumerate(zip(question,relation,object,score)):
            examples.append(
                InputExample(question=question,relation=relation,object=object,score=score))
        return examples


def convert_examples_to_features(examples, max_seq_length, tokenizer):
    
    features = []
    for (ex_index, example) in enumerate(examples):
        question_tokens = tokenizer.tokenize(example.question)
        
        if len(question_tokens) > max_seq_length - 1:
            question_tokens = question_tokens[:(max_seq_length - 1)]

        
        question_tokens = ["[CLS]"] + question_tokens
        question_segment_ids = [0] * len(question_tokens)

        question_input_ids = tokenizer.convert_tokens_to_ids(question_tokens)

        
        question_input_mask = [1] * len(question_input_ids)

        padding = [0] * (max_seq_length - len(question_input_ids))
        question_input_ids += padding
        question_input_mask += padding
        question_segment_ids += padding

        #--------------------------object----------------------------

        object_tokens = tokenizer.tokenize(example.object)
        
        if len(object_tokens) > max_seq_length - 2:
            object_tokens = object_tokens[:(max_seq_length - 2)]

        object_tokens = ["[SEP]"] + object_tokens + ["[SEP]"]
        object_segment_ids = [1] * len(object_tokens)

        object_input_ids = tokenizer.convert_tokens_to_ids(object_tokens)
        object_input_mask = [1] * len(object_input_ids)
        padding = [0] * (max_seq_length - len(object_input_ids))
        object_input_ids += padding
        object_input_mask += padding
        object_segment_ids += padding

        #--------------------------all to gether----------------------------
        input_ids=question_input_ids+object_input_ids
        input_mask=question_input_mask+object_input_mask
        segment_ids=question_segment_ids+object_segment_ids
        
        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
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

def get_optimizer(num_train_optimization_steps: int, learning_rate: float):
    grouped_parameters = [
       x for x in optimizer_grouped_parameters if any([p.requires_grad for p in x["params"]])
    ]
    for group in grouped_parameters:
        group['lr'] = learning_rate
    
    optimizer = FreezableBertAdam(grouped_parameters,lr=learning_rate, warmup=WARMUP_PROPORTION,t_total=num_train_optimization_steps)
    return optimizer

model = BertForSequenceRegression.from_pretrained('bert-base-uncased',cache_dir=PYTORCH_PRETRAINED_BERT_CACHE)
model.to(device)

param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]




def train(model: nn.Module, num_epochs: int, learning_rate: float,train_dataloader):
    num_train_optimization_steps = len(train_dataloader) * num_epochs 
    optimizer = get_optimizer(num_train_optimization_steps, learning_rate)
    assert all([x["lr"] == learning_rate for x in optimizer.param_groups])
    global_step = 0
    nb_tr_steps = 0
    tr_loss = 0    
    model.train()
    mb = master_bar(range(num_epochs))
    nb_tr_examples, nb_tr_steps = 0, 0    
    for _ in mb:
        for step, batch in enumerate(progress_bar(train_dataloader, parent=mb)):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, score = batch

            loss = model(input_ids, segment_ids, input_mask, score)
            if n_gpu > 1:
                loss = loss.mean() 
            loss.backward()

            if tr_loss == 0:
                tr_loss = loss.item()
            else:
                tr_loss = tr_loss * 0.9 + loss.item() * 0.1
            nb_tr_examples += input_ids.size(0)
            nb_tr_steps += 1
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1
            mb.child.comment = f'loss: {tr_loss:.4f} lr: {optimizer.get_lr()[0]:.2E}'
    logger.info("  train loss = %.4f", tr_loss) 
    return tr_loss



class TrainingModule:
    def __init__(self, args):
        self.train_dir = args.train_dir
        self.bert_model = args.bert_model
        self.cache_dir = args.cache_dir
        self.max_seq_length = args.max_seq_length
        self.batch_size = args.batch_size
        self.learning_rate = args.learning_rate
        self.device = args.device
        self.model = args.model
    
    def train_model(self):
        df_train = json.load(open(self.train_dir,'r'))


        tokenizer = BertTokenizer.from_pretrained(self.bert_model, do_lower_case=True, cache_dir=self.cache_dir)
        train_examples = DataProcessor().get_train_examples(df_train)
        train_features = convert_examples_to_features(train_examples, self.max_seq_length, tokenizer)

        del train_examples
        gc.collect()

        

        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_scores = torch.tensor([f.score for f in train_features], dtype=torch.float)
        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_scores)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=self.batch_size)

        logger.info("***** Training Started *****")

        set_trainable(model, True)
        set_trainable(model.bert.embeddings, False)
        set_trainable(model.bert.encoder, False)
        count_model_parameters(model)
        train(model, num_epochs = 2, learning_rate = 5e-4,train_dataloader=train_dataloader)

        set_trainable(model.bert.encoder.layer[11], True)
        set_trainable(model.bert.encoder.layer[10], True)
        count_model_parameters(model)
        train(model, num_epochs = 1, learning_rate = 5e-5,train_dataloader=train_dataloader)
        
        set_trainable(model, True)
        count_model_parameters(model)
        train(model, num_epochs = 1, learning_rate = 1e-5,train_dataloader=train_dataloader)

        model_to_save = model.module if hasattr(model, 'module') else model  
        output_model_file = "../model/"+sys.argv[1]+"/triple_retrieval_model.pth"
        torch.save(model_to_save.state_dict(), output_model_file)


if __name__ == "__main__":
    
    args = {
        'train_dir': '../data/'+sys.argv[1]+'/train/train_sample.json',
        'bert_model': 'bert-base-uncased',
        'cache_dir': PYTORCH_PRETRAINED_BERT_CACHE,
        'max_seq_length': MAX_SEQ_LENGTH,
        'batch_size': BATCH_SIZE,
        'device': device,
        'model':model
    }
    training_module = TrainingModule(args)
    training_module.train_model()