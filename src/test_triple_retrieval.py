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

def get_score(test_features,batch_size):
    
    all_input_ids = torch.tensor([f.input_ids for f in test_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in test_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in test_features], dtype=torch.long)
    test_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids)
    
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)

    model.eval()
    counter=0
    
    mb = progress_bar(test_dataloader)
    for input_ids, input_mask, segment_ids in mb:
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        
        with torch.no_grad():
            outputs = model(input_ids, segment_ids, input_mask)

        outputs = outputs.detach().cpu().numpy()
        if counter==0:
            
            all_outputs=torch.from_numpy(outputs)
            counter+=1
        else:
            all_outputs=torch.cat([all_outputs, torch.from_numpy(outputs)], dim=0)
            
    return all_outputs

###Testing###

placeholder =  sys.argv[1]
model=BertForSequenceRegression.from_pretrained('bert-base-uncased',cache_dir=PYTORCH_PRETRAINED_BERT_CACHE)
model.load_state_dict(torch.load("../model/"+placeholder+"/triple_retrieval_model.pth"))
model.to(device)


class TestingModule:
    def __init__(self, args):
        self.test_dir = args.train_dir
        self.out_dir = args.out_dir
        self.bert_model = args.bert_model
        self.cache_dir = args.cache_dir
        self.max_seq_length = args.max_seq_length
        self.batch_size = args.batch_size
        self.device = args.device
        self.model = args.model
    
    def test_model(self):

        test_f = open(self.test_dir,'r')

        scored_test_f = open(self.out_dir,'w')
        tokenizer = BertTokenizer.from_pretrained(self.bert_model, do_lower_case=True, cache_dir=self.cache_dir)
        

        input_f = [test_f]
        output_f = [scored_test_f]
        counter = 0
        

        for i in range(0,len(input_f)):
            final_result = []
            file_inp = json.load(input_f[i])
            for line in file_inp:
                data = line

                counter+=1
                
                if len(data['triples'])>0:
                    print('here')
                    test_examples = DataProcessor().get_test_examples(data)
                    test_features = convert_examples_to_features(test_examples, self.max_seq_length, tokenizer)        
                    
                    scores=get_score(test_features,self.batch_size)
                    print(scores)
                    lists = []
                    for triple,score in zip(data["triples"],scores):
                        triple["prune_score"] = score.item()
                        lists.append((triple,score.item()))
                    lists.sort(key=lambda s:s[1], reverse=True)
                    #lists = lists[:100]
                    
                    r = 0
                    keep_candidates = []
                    
                    for l in lists:
                        keep_candidates.append(l[0])
                    new_candidates = []
                    has_answer = False
                    for triple in keep_candidates:
                        new_candidates.append(triple)
                        if triple['answer']:
                            has_answer = True
                    
                    data['triples'] = new_candidates
                    if has_answer:
                        final_result.append(data)
            json_string = json.dumps(final_result, indent=2)
            output_f[i].write(json_string)
            output_f[i].close()



if __name__ == "__main__":
    
    args = {
        'train_dir': '../data/'+sys.argv[1]+'/train/test_sample.json',
        'out_dir':'../data/'+sys.argv[1]+'/test/scored_file.json',
        'bert_model': 'bert-base-uncased',
        'cache_dir': PYTORCH_PRETRAINED_BERT_CACHE,
        'max_seq_length': MAX_SEQ_LENGTH,
        'batch_size': 1000,
        'device': device,
        'model':model
    }
    training_module = TestingModule(args)
    training_module.test_model()