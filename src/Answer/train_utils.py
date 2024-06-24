# train_utils.py
import torch
import torch.nn as nn
from torch.optim import Adam
from pytorch_pretrained_bert.optimization import BertAdam
from Answer.config import Config

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

def count_model_parameters(model,config):
    config.logger.info(
        "# of paramters: {:,d}".format(
            sum(p.numel() for p in model.parameters())))
    config.logger.info(
        "# of trainable paramters: {:,d}".format(
            sum(p.numel() for p in model.parameters() if p.requires_grad)))




def get_optimizer(model, num_train_optimization_steps: int, learning_rate: float):
    config = Config()
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    grouped_parameters = [
       x for x in optimizer_grouped_parameters if any([p.requires_grad for p in x["params"]])
    ]
    for group in grouped_parameters:
        group['lr'] = learning_rate
    
    optimizer = FreezableBertAdam(grouped_parameters,
                             lr=learning_rate, warmup=config.WARMUP_PROPORTION,
                             t_total=num_train_optimization_steps)

    return optimizer
