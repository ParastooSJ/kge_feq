# data_loader.py
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import json
import numpy as np
from Answer.config import Config



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
 

    def get_train_examples(self,config):
        train_file = json.load(open(config.train_data_path, "r"))
        question = []
        subject = []
        answer = []
        score = []
        for line in train_file:
            
            for triple in line['triples'][:config.samples]:
                   question.append(line['question'][:100]+" "+triple['relation'])
                   subject.append(triple['subject'])
                   answer.append(triple['object'])
                   score.append(triple['relation_object_score'])
                  
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
            score.append(1 if triple['answer'] else -1)
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

