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
logger = logging.getLogger("SQ-bert-regressor")
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
        
        Es = LA.norm(subject+relation-objectt,dim=1)
        score = -1 * Es
        
        
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

    def get_train_examples(self):
        question = []
        subject = []
        answer = []
        score = []
        for line in data_train:
            #line= json.loads(line)
            
            for triple in line['triples'][:100]:
                
                   question.append(line['question'][:100]+" "+triple['relation'])
                   subject.append(triple['subject'])
                   answer.append(triple['object'])
                   score.append(1 if triple['answer'] else 0)
                   
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
            
            score.append(1 if triple['answer'] else 0)
        return self._create_examples(question,subject,answer,score)



    def _create_examples(self,question,subject,answer,score):
 
        examples = []
        for (i, (question,subject,answer,score)) in enumerate(zip(question,subject,answer,score)):
            examples.append(InputExample(question=question,subject=subject,answer=answer,score=score))

        return examples



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


def test(model,write=False):
    doc_no = 100
    test_dir = dataset + "test/scored_test.json"
    output_test_file = dataset + 'test/resuls'+str(doc_no)+'.txt'
    f_out = open(output_test_file,'w')  

   
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
            data= line 
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
                test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE)

                

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
                    score = ne.item()
                    if score > max_score:
                        answer  = triple['answer']
                        max_score = score
                        
                    triple["ranking_score"]=ne.item()
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

    

if __name__ == "__main__":
    model = BertForSequenceRegression.from_pretrained('bert-base-uncased',cache_dir=PYTORCH_PRETRAINED_BERT_CACHE)
    model.load_state_dict(torch.load("../model/"+sys.argv[1]+"/answer_selection_model.pth"), strict=False)
    model.to(device)
    test(model, write=True)



