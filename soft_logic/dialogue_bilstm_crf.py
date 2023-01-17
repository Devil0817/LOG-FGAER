#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
import time
import logging
import six
import json
from random import random
from tqdm import tqdm
from collections import OrderedDict
from functools import reduce, partial
from pathlib import Path
from visualdl import LogWriter
import paddle.fluid as fluid

import numpy as np
import multiprocessing
import pickle
import logging

import warnings

from sklearn.metrics import f1_score,recall_score,precision_score
import paddle as P

from propeller import log
import propeller.paddle as propeller

log.setLevel(logging.DEBUG)
logging.getLogger().setLevel(logging.DEBUG)

from demo.utils import create_if_not_exists, get_warmup_and_linear_decay
from ernie.modeling_ernie_crf import ErnieModel, ErnieModelForSequenceClassification, ErnieModelForTokenClassification
from ernie.tokenizing_ernie import ErnieTokenizer
#from ernie.optimization import AdamW, LinearDecay
from bilstmcrf import CRFModel
from utils_crf.options import Args
from utils_crf.tools import save_model, get_model_path_list, load_model_and_parallel
from utils_crf.evaluator import crf_evaluation
from utils_crf.optimizer import build_optimizer_and_scheduler
from transformers import BertTokenizer
import torch

parser = propeller.ArgumentParser('NER model with ERNIE')
parser.add_argument('--max_seqlen', type=int, default=256)
parser.add_argument('--bsz', type=int, default=32)
parser.add_argument('--data_dir', type=str, required=True)
parser.add_argument('--epoch', type=int, default=1)
parser.add_argument(
    '--warmup_proportion',
    type=float,
    default=0.1,
    help='if use_lr_decay is set, '
    'learning rate will raise to `lr` at `warmup_proportion` * `max_steps` and decay to 0. at `max_steps`'
)
parser.add_argument(
    '--max_steps',
    type=int,
    required=True,
    help='max_train_steps, set this to EPOCH * NUM_SAMPLES / BATCH_SIZE, used in learning rate scheduler'
)
parser.add_argument(
    '--use_amp',
    action='store_true',
    help='only activate AMP(auto mixed precision accelatoin) on TensorCore compatible devices'
)

parser.add_argument('--from_pretrained', type=Path, required=True)
parser.add_argument('--lr', type=float, default=5e-5, help='learning rate')
parser.add_argument(
    '--save_dir', type=Path, required=True, help='model output directory')
parser.add_argument(
    '--wd', type=float, default=0.01, help='weight decay, aka L2 regularizer')
args = parser.parse_args()

tokenizer = ErnieTokenizer.from_pretrained(args.from_pretrained)


def tokenizer_func(inputs):
    ret = inputs.split(b'\2')
    tokens, orig_pos = [], []
    for i, r in enumerate(ret):
        t = tokenizer.tokenize(r)
        for tt in t:
            tokens.append(tt)
            orig_pos.append(i)
    assert len(tokens) == len(orig_pos)
    return tokens + orig_pos


def tokenizer_func_for_label(inputs):
    return inputs.split(b'\2')


feature_map = {
    b"B-prov": 0, 
    b"E-prov": 1, 
    b"B-city": 2, 
    b"I-city": 3, 
    b"E-city": 4, 
    b"B-district": 5, 
    b"I-district": 6, 
    b"E-district": 7, 
    b"B-town": 8, 
    b"I-town": 9, 
    b"E-town": 10, 
    b"B-community": 11, 
    b"I-community": 12, 
    b"E-community": 13, 
    b"B-poi": 14, 
    b"E-poi": 15, 
    b"I-prov": 16, 
    b"I-poi": 17, 
    b"B-road": 18, 
    b"E-road": 19,
    b"B-roadno": 20,
    b"I-roadno": 21, 
    b"E-roadno": 22, 
    b"I-road": 23, 
    b"B-subpoi": 24, 
    b"I-subpoi": 25, 
    b"E-subpoi": 26, 
    b"B-devzone": 27, 
    b"I-devzone": 28, 
    b"E-devzone": 29, 
    b"B-houseno": 30, 
    b"I-houseno": 31, 
    b"E-houseno": 32, 
    b"B-intersection": 33, 
    b"I-intersection": 34, 
    b"E-intersection": 35, 
    b"B-assist": 36, 
    b"I-assist": 37, 
    b"E-assist": 38, 
    b"B-cellno": 39, 
    b"I-cellno": 40, 
    b"E-cellno": 41, 
    b"B-floorno": 42, 
    b"E-floorno": 43, 
    b"S-assist": 44, 
    b"I-floorno": 45, 
    b"B-distance": 46, 
    b"I-distance": 47, 
    b"E-distance": 48, 
    b"B-village_group": 49, 
    b"E-village_group": 50, 
    b"I-village_group": 51, 
    b"S-poi": 52, 
    b"S-intersection": 53, 
    b"S-district": 54, 
    b"S-community":55,
    b"O": 56,
}
other_tag_id = feature_map[b'O']

feature_column = propeller.data.FeatureColumns([
    propeller.data.TextColumn(
        'text_a',
        unk_id=tokenizer.unk_id,
        vocab_dict=tokenizer.vocab,
        tokenizer=tokenizer_func), propeller.data.TextColumn(
            'label',
            unk_id=other_tag_id,
            vocab_dict=feature_map,
            tokenizer=tokenizer_func_for_label, )
])


def before(seg, label):
    seg, orig_pos = np.split(seg, 2)
    aligned_label = label[orig_pos]
    seg, _ = tokenizer.truncate(seg, [], args.max_seqlen)
    aligned_label, _ = tokenizer.truncate(aligned_label, [], args.max_seqlen)
    orig_pos, _ = tokenizer.truncate(orig_pos, [], args.max_seqlen)

    sentence, segments = tokenizer.build_for_ernie(
        seg
    )  #utils.data.build_1_pair(seg, max_seqlen=args.max_seqlen, cls_id=cls_id, sep_id=sep_id)
    aligned_label = np.concatenate([[0], aligned_label, [0]], 0)
    orig_pos = np.concatenate([[0], orig_pos, [0]])

    assert len(aligned_label) == len(sentence) == len(orig_pos), (
        len(aligned_label), len(sentence), len(orig_pos))  # alinged
    return sentence, segments, aligned_label, label, orig_pos

train_ds = feature_column.build_dataset('train', data_dir=os.path.join(args.data_dir, 'train'), shuffle=False, repeat=False, use_gz=False) \
                               .map(before) \
                               .padded_batch(args.bsz, (0,0,-100, other_tag_id + 1, 0)) \

dev_ds = feature_column.build_dataset('dev', data_dir=os.path.join(args.data_dir, 'dev'), shuffle=False, repeat=False, use_gz=False) \
                               .map(before) \
                               .padded_batch(args.bsz, (0,0,-100, other_tag_id + 1,0)) \

test_ds = feature_column.build_dataset('test', data_dir=os.path.join(args.data_dir, 'test'), shuffle=False, repeat=False, use_gz=False) \
                               .map(before) \
                               .padded_batch(args.bsz, (0,0,-100, other_tag_id + 1,0)) \


def change_vector(la,merged_preds):
    new_la = np.zeros((len(la)))
    new_merged_preds = np.zeros((len(la)))

    for i in range(len(la)):
        if la[i] == 0 or la[i] == 1 or la[i] == 16:
            new_la[i] = 0 #prov
        elif la[i] == 2 or la[i] == 3 or la[i] == 4:
            new_la[i] = 1 #city
        elif la[i] == 5 or la[i] == 6 or la[i] == 7 or la[i] == 54:
            new_la[i] = 2 #district
        elif la[i] == 8 or la[i] == 9 or la[i] == 10:
            new_la[i] = 3 #town
        elif la[i] == 11 or la[i] == 12 or la[i] == 13 or la[i] == 55:
            new_la[i] = 4 #community
        elif la[i] == 14 or la[i] == 15 or la[i] == 17 or la[i] == 52:
            new_la[i] = 5 #poi
        elif la[i] == 18 or la[i] == 19 or la[i] == 23:
            new_la[i] = 6 #road
        elif la[i] == 20 or la[i] == 21 or la[i] == 22:
            new_la[i] = 7 #roadno
        elif la[i] == 24 or la[i] == 25 or la[i] == 26:
            new_la[i] = 8 #subpoi
        elif la[i] == 27 or la[i] == 28 or la[i] == 29:
            new_la[i] = 9 #devzone
        elif la[i] == 30 or la[i] == 31 or la[i] == 32:
            new_la[i] = 10 #houseno
        elif la[i] == 33 or la[i] == 34 or la[i] == 35 or la[i] == 53:
            new_la[i] = 11 #intersection
        elif la[i] == 36 or la[i] == 37 or la[i] == 38 or la[i] == 44:
            new_la[i] = 12 #assist
        elif la[i] == 39 or la[i] == 40 or la[i] == 41:
            new_la[i] = 13 #cellno
        elif la[i] == 42 or la[i] == 43 or la[i] == 45:
            new_la[i] = 14 #floorno
        elif la[i] == 46 or la[i] == 47 or la[i] == 48:
            new_la[i] = 15 #distance
        elif la[i] == 49 or la[i] == 50 or la[i] == 51:
            new_la[i] = 16 #village_group 
        else:
            new_la[i] = 17

    for i in range(len(merged_preds)):
        if merged_preds[i] == 0 or merged_preds[i] == 1 or merged_preds[i] == 16:
            new_merged_preds[i] = 0 #prov
        elif merged_preds[i] == 2 or merged_preds[i] == 3 or merged_preds[i] == 4:
            new_merged_preds[i] = 1 #city
        elif merged_preds[i] == 5 or merged_preds[i] == 6 or merged_preds[i] == 7 or merged_preds[i] == 54:
            new_merged_preds[i] = 2 #district
        elif merged_preds[i] == 8 or merged_preds[i] == 9 or merged_preds[i] == 10:
            new_merged_preds[i] = 3 #town
        elif merged_preds[i] == 11 or merged_preds[i] == 12 or merged_preds[i] == 13 or merged_preds[i] == 55:
            new_merged_preds[i] = 4 #community
        elif merged_preds[i] == 14 or merged_preds[i] == 15 or merged_preds[i] == 17 or merged_preds[i] == 52:
            new_merged_preds[i] = 5 #poi
        elif merged_preds[i] == 18 or merged_preds[i] == 19 or merged_preds[i] == 23:
            new_merged_preds[i] = 6 #road
        elif merged_preds[i] == 20 or merged_preds[i] == 21 or merged_preds[i] == 22:
            new_merged_preds[i] = 7 #roadno
        elif merged_preds[i] == 24 or merged_preds[i] == 25 or merged_preds[i] == 26:
            new_merged_preds[i] = 8 #subpoi
        elif merged_preds[i] == 27 or merged_preds[i] == 28 or merged_preds[i] == 29:
            new_merged_preds[i] = 9 #devzone
        elif merged_preds[i] == 30 or merged_preds[i] == 31 or merged_preds[i] == 32:
            new_merged_preds[i] = 10 #houseno
        elif merged_preds[i] == 33 or merged_preds[i] == 34 or merged_preds[i] == 35 or merged_preds[i] == 53:
            new_merged_preds[i] = 11 #intersection
        elif merged_preds[i] == 36 or merged_preds[i] == 37 or merged_preds[i] == 38 or merged_preds[i] == 44:
            new_merged_preds[i] = 12 #assist
        elif merged_preds[i] == 39 or merged_preds[i] == 40 or merged_preds[i] == 41:
            new_merged_preds[i] = 13 #cellno
        elif merged_preds[i] == 42 or merged_preds[i] == 43 or merged_preds[i] == 45:
            new_merged_preds[i] = 14 #floorno
        elif merged_preds[i] == 46 or merged_preds[i] == 47 or merged_preds[i] == 48:
            new_merged_preds[i] = 15 #distance
        elif merged_preds[i] == 49 or merged_preds[i] == 50 or merged_preds[i] == 51:
            new_merged_preds[i] = 16 #vilmerged_predsge_group 
        else:
            new_merged_preds[i] = 17
        
    return new_la, new_merged_preds
    
    
def caculate_f1(label, pred):
    result_list_large = []
    result_list_small = []
    for i in range(57):
        cur_label = [1 if l == i else 0 for l in label]
        cur_pred = [1 if p == i else 0 for p in pred]
        cur_result = f1_score(cur_label, cur_pred, average='binary')
        result_list_large.append(cur_result)
    for i in range(18):
        cur_value = 0
        if i == 0:
            cur_value = (result_list_large[0] + result_list_large[1] + result_list_large[16])/3
        elif i == 1:
            cur_value = (result_list_large[2] + result_list_large[3] + result_list_large[4])/3
        elif i == 2:
            cur_value = (result_list_large[5] + result_list_large[6] + result_list_large[7] + result_list_large[54])/4
        elif i == 3:
            cur_value = (result_list_large[8] + result_list_large[9] + result_list_large[10])/3
        elif i == 4:
            cur_value = (result_list_large[11] + result_list_large[12] + result_list_large[13] + result_list_large[55])/4
        elif i == 5:
            cur_value = (result_list_large[14] + result_list_large[15] + result_list_large[17] + result_list_large[52])/4
        elif i == 6:
            cur_value = (result_list_large[18] + result_list_large[19] + result_list_large[23])/3
        elif i == 7:
            cur_value = (result_list_large[20] + result_list_large[21] + result_list_large[22])/3
        elif i == 8:
            cur_value = (result_list_large[24] + result_list_large[25] + result_list_large[26])/3
        elif i == 9:
            cur_value = (result_list_large[27] + result_list_large[28] + result_list_large[29])/3
        elif i == 10:
            cur_value = (result_list_large[30] + result_list_large[31] + result_list_large[32])/3
        elif i == 11:
            cur_value = (result_list_large[33] + result_list_large[34] + result_list_large[35] + result_list_large[53])/4
        elif i == 12:
            cur_value = (result_list_large[36] + result_list_large[37] + result_list_large[38] + result_list_large[44])/4
        elif i == 13:
            cur_value = (result_list_large[39] + result_list_large[40] + result_list_large[41])/3
        elif i == 14:
            cur_value = (result_list_large[42] + result_list_large[43] + result_list_large[45])/3
        elif i == 15:
            cur_value = (result_list_large[46] + result_list_large[47] + result_list_large[48])/3
        elif i == 16:
            cur_value = (result_list_large[49] + result_list_large[50] + result_list_large[51])/3
        else:
            cur_value = result_list_large[56]
        result_list_small.append(cur_value)
            
    weight_f1 = f1_score(label, pred, labels=range(57),average='weighted')       
    print('###macro:',(f1_score(label, pred, labels=range(57),average='macro')))
    print('###weighted:',(f1_score(label, pred, labels=range(57),average='weighted')))
    return result_list_small,weight_f1
    
    

def evaluate(model, dataset):
    model.eval()
    final_label = []
    final_pred = []
    with P.no_grad():
        chunkf1 = propeller.metrics.ChunkF1(None, None, None, len(feature_map))
        for step, (ids, sids, aligned_label, label, orig_pos
                   ) in enumerate(P.io.DataLoader(
                       dataset, batch_size=None)):
            loss, logits = model(ids, sids)
            crf_decode = fluid.layers.crf_decoding(input=logits,param_attr=fluid.ParamAttr(name="crfw"))
            print('##')
            print('@@'.type)

            assert orig_pos.shape[0] == logits.shape[0] == ids.shape[
                0] == label.shape[0]
            for pos, lo, la, id in zip(orig_pos.numpy(),
                                       logits.numpy(),
                                       label.numpy(), ids.numpy()):
                _dic = OrderedDict()
                assert len(pos) == len(lo) == len(id)
                for _pos, _lo, _id in zip(pos, lo, id):
                    if _id > tokenizer.mask_id:  # [MASK] is the largest special token
                        _dic.setdefault(_pos, []).append(_lo)
                merged_lo = np.array(
                    [np.array(l).mean(0) for _, l in six.iteritems(_dic)])
                #print('$$merged_lo shape:',merged_lo.shape)
                #print('$$merged_lo:',merged_lo)
                merged_preds = np.argmax(merged_lo, -1)
                la = la[np.where(la != (other_tag_id + 1))]  #remove pad
               
                if len(la) > len(merged_preds):
                    log.warn(
                        'accuracy loss due to truncation: label len:%d, truncate to %d'
                        % (len(la), len(merged_preds)))
                    merged_preds = np.pad(merged_preds,
                                          [0, len(la) - len(merged_preds)],
                                          mode='constant',
                                          constant_values=57)
                else:
                    assert len(la) == len(
                        merged_preds
                    ), 'expect label == prediction, got %d vs %d' % (
                        la.shape, merged_preds.shape)
                chunkf1.update((merged_preds, la, np.array(len(la))))
                #new_la, new_merged_preds = change_vector(la,merged_preds)
                final_label += list(la)
                final_pred += list(merged_preds)

        each_f1,weight_f1 = caculate_f1(final_label, final_pred)
        f1 = chunkf1.eval()

    model.train()
    return f1, each_f1, weight_f1

def evaluate_crf(model, model_crf, dataset):
    model.eval()
    final_label = []
    final_pred = []
    with P.no_grad():
        chunkf1 = propeller.metrics.ChunkF1(None, None, None, len(feature_map))
        for step, (ids, sids, aligned_label, label, orig_pos
                   ) in enumerate(P.io.DataLoader(
                       dataset, batch_size=None)):
            step_length = []
            for pos, la, id, sid in zip(orig_pos.numpy(),
                                       label.numpy(), ids.numpy(),sids.numpy()):
                count_length = 0
                for _pos, _id in zip(pos, id):
                    if _id > tokenizer.mask_id:
                        count_length += 1
                step_length.append([1 if i >= 0 and i <= count_length else 0 for i in range(ids.shape[1])])

            label_length = np.array(step_length).astype('int64')
            loss, logits = model(ids, sids)
            logits = model_crf(logits,label_length)
            logits = logits.cpu().detach().numpy()
            logits = P.to_tensor(logits)
            #print('orig_pos',orig_pos.shape)
            #print('ids',ids.shape)
            #print('logits',logits.shape)
            
            assert orig_pos.shape[0] == logits.shape[0] == ids.shape[0] == label.shape[0]
            for pos, lo, la, id in zip(orig_pos.numpy(),
                                       logits.numpy(),
                                       label.numpy(), ids.numpy()):
                #print('##logits',lo)
                _dic = OrderedDict()
                assert len(pos) == len(lo) == len(id)
                for _pos, _lo, _id in zip(pos, lo, id):
                    if _id > tokenizer.mask_id:  # [MASK] is the largest special token
                        _dic.setdefault(_pos, []).append(_lo)
                merged_preds = np.array(
                    [np.array(l).mean(0) for _, l in six.iteritems(_dic)])
                #print('$$merged_preds shape:',merged_preds.shape)
                #print('$$merged_lo:',merged_lo)
                #merged_preds = np.argmax(merged_lo, -1)
                la = la[np.where(la != (other_tag_id + 1))]  #remove pad
                #print('##la shape',la.shape)
                #print('##'.shape)
               
                if len(la) > len(merged_preds):
                    log.warn(
                        'accuracy loss due to truncation: label len:%d, truncate to %d'
                        % (len(la), len(merged_preds)))
                    merged_preds = np.pad(merged_preds,
                                          [0, len(la) - len(merged_preds)],
                                          mode='constant',
                                          constant_values=57)
                else:
                    assert len(la) == len(
                        merged_preds
                    ), 'expect label == prediction, got %d vs %d' % (
                        la.shape, merged_preds.shape)
                chunkf1.update((merged_preds, la, np.array(len(la))))
                #new_la, new_merged_preds = change_vector(la,merged_preds)
                final_label += list(la)
                final_pred += list(merged_preds)

        each_f1,weight_f1 = caculate_f1(final_label, final_pred)
        f1 = chunkf1.eval()
        
    model.train()
    return f1, each_f1, weight_f1

result_str = ''
warnings.filterwarnings("ignore")
model = ErnieModelForTokenClassification.from_pretrained(
    args.from_pretrained,
    num_labels=len(feature_map),
    name='',
    has_pooler=False)

g_clip = P.nn.ClipGradByGlobalNorm(1.0)  #experimental
param_name_to_exclue_from_weight_decay = re.compile(
    r'.*layer_norm_scale|.*layer_norm_bias|.*b_0')
lr_scheduler = P.optimizer.lr.LambdaDecay(
    args.lr,
    get_warmup_and_linear_decay(args.max_steps,
                                int(args.warmup_proportion * args.max_steps)))
opt = P.optimizer.AdamW(
    lr_scheduler,
    parameters=model.parameters(),
    weight_decay=args.wd,
    apply_decay_param_fun=lambda n: not param_name_to_exclue_from_weight_decay.match(n),
    grad_clip=g_clip)

scaler = P.amp.GradScaler(enable=args.use_amp)


opt_crf = Args().get_parser()
opt_crf.output_dir = opt_crf.output_dir + '/' + str(opt_crf.no)
bert_dir = opt_crf.bert_dir
crf_max_seq_len = opt_crf.max_seq_len
device = 'cpu' if opt_crf.gpu_ids == '-1' else 'cuda'
#tokenizer = BertTokenizer(os.path.join(bert_dir, 'vocab.txt'))
model_crf = CRFModel(bert_dir=bert_dir,
                 num_tags=57,
                 dropout_prob=0.1,
                 mid_linear_dims=64)

model_crf= model_crf.to(device)
count_total = 0
for step, (ids, sids, aligned_label, label, orig_pos) in enumerate(P.io.DataLoader(
                    train_ds, batch_size=None)):
    count_total += 1
t_total = count_total
optimizer, scheduler = build_optimizer_and_scheduler(opt_crf, model_crf, t_total)


with LogWriter(
        logdir=str(create_if_not_exists(args.save_dir / 'vdl'))) as log_writer:
    with P.amp.auto_cast(enable=args.use_amp):
        for epoch in range(args.epoch):
            for step, (
                    (ids, sids, aligned_label, label, orig_pos), (ids2, sids2, aligned_label2, label2, orig_pos2)
            ) in enumerate(zip(P.io.DataLoader(
                    train_ds, batch_size=None),P.io.DataLoader(
                    train_ds, batch_size=None))):
                model_crf.train()
                model_crf.zero_grad()
                label_length = []
                for pos, la, id in zip(orig_pos.numpy(),
                                       label.numpy(), ids.numpy()):
                    assert len(pos)  == len(id)
                    count_length = 0
                    for _pos, _id in zip(pos, id):
                        if _id > tokenizer.mask_id:
                            count_length += 1
                    label_length.append([1 if i >= 0 and i <= count_length else 0 for i in range(ids.shape[1])])
                label_length = np.array(label_length).astype('int64')
                        
                loss, logits = model(ids, sids, labels=aligned_label)
                loss_crf = model_crf(logits,label_length,aligned_label)[0]
                print('$$%',loss_crf)
                loss_crf.backward()
                torch.nn.utils.clip_grad_norm_(model_crf.parameters(), opt_crf.max_grad_norm)
                optimizer.step()
                scheduler.step()
                
                #loss = loss_crf.cpu().detach().numpy()
                #loss = P.to_tensor(loss)
                #loss = scaler.scale(loss)
                loss.backward()
                scaler.minimize(opt, loss)
                model.clear_gradients()
                lr_scheduler.step()
                #print('@@'.shape)
                

                if step % 10 == 0:
                    _lr = lr_scheduler.get_lr()
                    if args.use_amp:
                        _l = (loss / scaler._scale).numpy()
                        msg = '[step-%d] train loss %.5f lr %.3e scaling %.3e' % (
                            step, _l, _lr, scaler._scale.numpy())
                    else:
                        _l = loss.numpy()
                        msg = '[step-%d] train loss %.5f lr %.3e' % (step, _l,
                                                                     _lr)
                    log.debug(msg)
                    log_writer.add_scalar('loss', _l, step=step)
                    log_writer.add_scalar('lr', _lr, step=step)

                if step % 100 == 0:
                    f1, each_f1,weight_f1 = evaluate_crf(model,model_crf, dev_ds)
                    model_crf.eval()
                    log.debug('eval f1: %.5f' % f1)
                    result_str += str(step) + (': eval f1: %.5f'% f1) + '\n'
                    log.debug('eval each_f1:',each_f1)
                    result_str += str(step) + ': eval each_f1:'+ str(each_f1) + '\n'
                    result_str += str(step) + (': eval weight_f1: %.5f'% weight_f1) + '\n'
                    result_str += '\n'
                    log_writer.add_scalar('eval/f1', f1, step=step)
                    if args.save_dir is not None:
                        P.save(model.state_dict(),str( args.save_dir / 'ckpt.bin'))
                        

f1, each_f1,weight_f1 = evaluate_crf(model, model_crf,dev_ds)
log.debug('final eval f1: %.5f' % f1)
log_writer.add_scalar('eval/f1', f1, step=step)
result_str += ('final eval f1: %.5f' % f1) + '\n'
result_str += ('final eval weight_f1: %.5f' % weight_f1) + '\n'
result_str += str(each_f1) + '\n'
with open('./log/ernie-bilstm-crf-64.txt', 'w+', encoding='utf-8') as f1:
    f1.write(result_str)
if args.save_dir is not None:
    P.save(model.state_dict(),str( args.save_dir / 'ckpt.bin'))
