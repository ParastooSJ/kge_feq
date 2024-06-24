# model.py
import torch
import torch.nn as nn
from pytorch_pretrained_bert.modeling import BertPreTrainedModel, BertModel
from transformers import EncoderDecoderModel, BertTokenizer,BertModel,AdamW,get_scheduler
from torch import linalg as LA

class BertForAnswerSelection(BertPreTrainedModel):
    #def __init__(self, config):
     #   super(BertForAnswerSelection, self).__init__(config)
      #  self.bert_subject = BertModel.from_pretrained('bert-base-uncased')
       # self.bert_object = BertModel.from_pretrained('bert-base-uncased')
        #self.bert_relation = BertModel.from_pretrained('bert-base-uncased')
        #self.loss_function = nn.MSELoss()

    def __init__(self,config):
        super(BertForAnswerSelection, self).__init__(config)
        self.bert_subject = BertModel.from_pretrained('bert-base-uncased')
        self.bert_relation = BertModel.from_pretrained('bert-base-uncased')
        self.bert_object = BertModel.from_pretrained('bert-base-uncased')
        self.loss_fct = torch.nn.MSELoss()
        
    

    def forward(self, input_ids, attention_mask,subject_ids,subject_mask,output_ids,output_attention_mask,targets=None):

        subject = self.bert_subject(subject_ids,subject_mask).pooler_output
        objectt = self.bert_object(output_ids,output_attention_mask).pooler_output
        relation = self.bert_relation(input_ids, attention_mask).pooler_output

        subject = torch.FloatTensor(subject.to('cpu')).to('cuda')
        objectt = torch.FloatTensor(objectt.to('cpu')).to('cuda')
        relation = torch.FloatTensor(relation.to('cpu')).to('cuda')

        Es = LA.norm(subject+relation-objectt,dim=1)
        Es = Es+ 0.0001
        score = -1 * torch.log(Es)
        
        
        if targets is not None:
            loss = self.loss_fct(score.view(-1), targets.view(-1))
            loss = torch.FloatTensor(loss.to('cpu')).to('cuda')

            return loss
        else:
            return score 


    #def forward(self, subject_ids, subject_masks, object_ids, object_masks, relation_ids, relation_masks, labels=None):
     #   subject_embeddings = self.bert_subject(subject_ids, attention_mask=subject_masks).pooler_output
      #  object_embeddings = self.bert_object(object_ids, attention_mask=object_masks).pooler_output
       # relation_embeddings = self.bert_relation(relation_ids, attention_mask=relation_masks).pooler_output

        # Example score calculation (customizable)
        #score = -torch.norm(subject_embeddings + relation_embeddings - object_embeddings, p=2, dim=1)

        #if labels is not None:
         #   loss = self.loss_function(score, labels.float())
          #  return loss
        #return score


