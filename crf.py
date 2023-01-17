import torch.nn as nn
import torch
#from torchcrf import CRF
from fastNLP.modules import ConditionalRandomField
from transformers import  BertModel
import math
import numpy as np
import os

from utils_crf.evaluator import crf_decode

class BaseModel(nn.Module):
    def __init__(self,
                 bert_dir,
                 dropout_prob):
        super(BaseModel, self).__init__()
        config_path = os.path.join(bert_dir, 'config.json')

        assert os.path.exists(bert_dir) and os.path.exists(config_path), \
            'pretrained bert file does not exist'

        self.bert_module = BertModel.from_pretrained(bert_dir,
                                                     output_hidden_states=True,
                                                     hidden_dropout_prob=dropout_prob)

        self.bert_config = self.bert_module.config

    @staticmethod
    def _init_weights(blocks, **kwargs):
        """
        参数初始化，将 Linear / Embedding / LayerNorm 与 Bert 进行一样的初始化
        """
        for block in blocks:
            for module in block.modules():
                if isinstance(module, nn.Linear):
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                elif isinstance(module, nn.Embedding):
                    nn.init.normal_(module.weight, mean=0, std=kwargs.pop('initializer_range', 0.02))
                elif isinstance(module, nn.LayerNorm):
                    nn.init.ones_(module.weight)
                    nn.init.zeros_(module.bias)


# baseline
class CRFModel(BaseModel):
    def __init__(self,
                 bert_dir,
                 num_tags,
                 dropout_prob=0.1,
                 **kwargs):
        super(CRFModel, self).__init__(bert_dir=bert_dir, dropout_prob=dropout_prob)
        self.num_tags = num_tags
        #out_dims = self.bert_config.hidden_size
        #out_dims = self.num_tags

        #mid_linear_dims = kwargs.pop('mid_linear_dims', 64)

        #self.mid_linear = nn.Sequential(
            #nn.Linear(out_dims, mid_linear_dims),
            #nn.ReLU(),
            #nn.Dropout(dropout_prob)
        #)

        #out_dims = mid_linear_dims

        #self.classifier = nn.Linear(mid_linear_dims * 2, num_tags)

        #self.loss_weight = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        #self.loss_weight.data.fill_(-0.2)
        #num_layers = 1
        #self.lstm_net = nn.LSTM(out_dims, mid_linear_dims,
                                #num_layers=num_layers, dropout=dropout_prob,
                                #bidirectional=True)
        self.crf = ConditionalRandomField(num_tags, include_start_end_trans=True)
        #init_blocks = [self.mid_linear, self.classifier]

        #self._init_weights(init_blocks, initializer_range=self.bert_config.initializer_range)

    def forward(self,
                logits,
                attention_masks,
                labels=None,
                ):
        if labels is not None:
            logits_new = logits.numpy()
            labels_new = labels.numpy()
            labels_new[labels_new==-100] = 56
            logits_new = torch.from_numpy(logits_new)
            labels_new = torch.from_numpy(labels_new)
            label_length = torch.from_numpy(attention_masks)
            #logits_new = self.lstm_net(logits_new.to('cuda'))[0]
            #logits_new = self.classifier(logits_new)
            loss = self.crf(logits_new.long().to('cuda'), labels_new.long().to('cuda'), label_length.to('cuda')).mean()
            return loss,logits
            
        else:
            loss = None
            logits_new = logits.numpy()
            logits_new = torch.from_numpy(logits_new)
            label_length = torch.from_numpy(attention_masks)
            logits, scores =  self.crf.viterbi_decode(logits_new.to('cuda'), label_length.to('cuda'))
            return logits
            
        



