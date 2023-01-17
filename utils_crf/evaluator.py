import torch

import numpy as np
#from torchcrf import CRF



def get_base_out(model, loader, device):
    """
    每一个任务的 forward 都一样，封装起来
    """
    model.eval()

    with torch.no_grad():
        for idx, _batch in enumerate(loader):

            for key in _batch.keys():
                _batch[key] = _batch[key].to(device)


            # crf decode得到的就是不带mask的结果，gt需要用attention masks去掉padding
            gt = _batch["labels"].cpu().numpy()
            _batch["labels"] = None
            attention_masks= _batch["attention_masks"].cpu().numpy()
            gt = [x[mask > 0].tolist() for x , mask in zip(gt, attention_masks)]

            pred, scores = model(**_batch)
            # pred = [x[mask > 0].tolist() for x , mask in zip(pred, attention_masks)]
            yield pred, gt, scores

def get_base_out_nn(model, loader, device):
    """
    每一个任务的 forward 都一样，封装起来
    """
    model.eval()

    with torch.no_grad():
        for idx, _batch in enumerate(loader):

            for key in _batch.keys():
                _batch[key] = _batch[key].to(device)


            # crf decode得到的就是不带mask的结果，gt需要用attention masks去掉padding
            gt = _batch["labels"].cpu().numpy()
            _batch["labels"] = None
            attention_masks= _batch["attention_masks"].cpu().numpy()
            gt = [x[mask > 0].tolist() for x , mask in zip(gt, attention_masks)]

            pred, scores = model(**_batch)

            yield pred, gt

def calculate_metric(gt, predict):
    """
    计算 tp fp fn
    """
    tp, fp, fn = 0, 0, 0
    for entity_predict in predict:
        flag = 0
        for entity_gt in gt:
            if entity_predict[0] == entity_gt[0] and entity_predict[1] == entity_gt[1]:
                flag = 1
                tp += 1
                break
        if flag == 0:
            fp += 1

    fn = len(gt) - tp

    return np.array([tp, fp, fn])


def get_p_r_f(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp != 0 else 0
    r = tp / (tp + fn) if tp + fn != 0 else 0
    f1 = 2 * p * r / (p + r) if p + r != 0 else 0
    return np.array([p, r, f1])


def crf_decode(decode_tokens, id2ent):
    """
    CRF 解码，用于解码 time loc 的提取
    """
    predict_entities = {}

    decode_tokens = decode_tokens[1:-1]  # 除去 CLS SEP token

    index_ = 0

    while index_ < len(decode_tokens):

        token_label = id2ent[decode_tokens[index_]].split('-')

        if token_label[0].startswith('S'):
            token_type = token_label[1]


            if token_type not in predict_entities:
                predict_entities[token_type] = [(index_, index_)]
            else:
                predict_entities[token_type].append((index_, int(index_)))

            index_ += 1

        elif token_label[0].startswith('B'):
            token_type = token_label[1]
            start_index = index_

            index_ += 1
            while index_ < len(decode_tokens):
                temp_token_label = id2ent[decode_tokens[index_]].split('-')

                if temp_token_label[0].startswith('I') and token_type == temp_token_label[1]:
                    index_ += 1
                elif temp_token_label[0].startswith('E') and token_type == temp_token_label[1]:
                    end_index = index_
                    index_ += 1



                    if token_type not in predict_entities:
                        predict_entities[token_type] = [(start_index, end_index)]
                    else:
                        predict_entities[token_type].append((start_index, int(end_index)))

                    break
                else:
                    break
        else:
            index_ += 1

    return predict_entities


def crf_evaluation(model,
                   data_loader,
                   device,
                   ent2id,
                   entity_types):
    pred_tokens = []
    gt_tokens = []
    pred_scores = []


    for tmp_pred, labels, tmp_score in get_base_out(model, data_loader, device):

        pred_tokens.extend(tmp_pred)
        pred_scores.extend(tmp_score)
        gt_tokens.extend(labels)

    id2ent = {ent2id[key]: key for key in ent2id.keys()}

    role_metric = np.zeros([17, 3])

    mirco_metrics = np.zeros(3)
    type_weight = {k : 1/17 for k in entity_types} # TODO： 按照分布给权重

    for tmp_pred, tmp_gt in zip(pred_tokens, gt_tokens):

        tmp_metric = np.zeros([17, 3])

        pred_entities = crf_decode(tmp_pred, id2ent)
        gt_entities = crf_decode(tmp_gt, id2ent)

        for idx, _type in enumerate(entity_types):
            if _type not in pred_entities:
                pred_entities[_type] = []
            if _type not in gt_entities:
                gt_entities[_type] = []

            tmp_metric[idx] += calculate_metric(gt_entities[_type], pred_entities[_type])

        role_metric += tmp_metric

    for idx, _type in enumerate(entity_types):
        temp_metric = get_p_r_f(role_metric[idx][0], role_metric[idx][1], role_metric[idx][2])

        mirco_metrics += temp_metric * type_weight[_type]

    metric_str = f'[MIRCO] precision: {mirco_metrics[0]:.4f}, ' \
                 f'recall: {mirco_metrics[1]:.4f}, f1: {mirco_metrics[2]:.4f}'

    return metric_str, mirco_metrics[2], pred_tokens, pred_scores

def fix(labels):
    # labels = labels.split(' ')
    last = ''
    flag = 'O'
    fix_labels = []
    for idx, l in enumerate(labels):
        if l == last:
            if l == 'O':  # l == last and l == O
                fix_labels.append('O')
                flag = 'O'
            else:  # l == last and l != O
                fix_labels.append('I-' + l)
                flag = 'I'
        else:  # l != last
            if l == 'O': # l != last and l == O
                if flag == 'B':
                    fix_labels[-1] = 'S-' + fix_labels[-1][2:]
                else:  # l == I
                    if idx != 0:
                        fix_labels[-1] = 'E-' + fix_labels[-1][2:]
                fix_labels.append('O')
                flag = 'O'
            else:  # l != last and l != O
                if flag == 'I':
                    fix_labels[-1] = 'E-' + fix_labels[-1][2:]
                if flag == 'B': # flag = O
                    fix_labels[-1] = 'S-' + fix_labels[-1][2:]
                flag = 'B'
                fix_labels.append('B-' + l)
        last = l
    if flag == 'I':
        fix_labels[-1] = 'E-' + fix_labels[-1][2:]
    return fix_labels

def classify_decode(decode_tokens, id2ent):
    predict_entities = {}


    decode_tokens = decode_tokens[1:-1]  # 除去 CLS SEP token
    ent_label = [id2ent[decode_tokens[index_]] for index_ in range(len(decode_tokens))]
    decode_tokens = fix(ent_label)

    index_ = 0

    while index_ < len(decode_tokens):

        token_label = decode_tokens[index_].split('-')

        if token_label[0].startswith('S'):
            token_type = token_label[1]

            if token_type not in predict_entities:
                predict_entities[token_type] = [(index_, index_)]
            else:
                predict_entities[token_type].append((index_, int(index_)))

            index_ += 1

        elif token_label[0].startswith('B'):
            token_type = token_label[1]
            start_index = index_

            index_ += 1
            while index_ < len(decode_tokens):
                temp_token_label = decode_tokens[index_].split('-')

                if temp_token_label[0].startswith('I') and token_type == temp_token_label[1]:
                    index_ += 1
                elif temp_token_label[0].startswith('E') and token_type == temp_token_label[1]:
                    end_index = index_
                    index_ += 1

                    if token_type not in predict_entities:
                        predict_entities[token_type] = [(start_index, end_index)]
                    else:
                        predict_entities[token_type].append((start_index, int(end_index)))

                    break
                else:
                    break
        else:
            index_ += 1

    return predict_entities


def classify_evaluation(model,
                   data_loader,
                   device,
                   ent2id,
                   entity_types):
    pred_tokens = []
    gt_tokens = []

    id2ent = {ent2id[key]: key for key in ent2id.keys()}

    for tmp_pred, tmp_label in get_base_out_nn(model, data_loader, device):
        pred_tokens.extend(tmp_pred)
        gt_tokens.extend(tmp_label)

    role_metric = np.zeros([17, 3])

    mirco_metrics = np.zeros(3)
    type_weight = {k : 1/17 for k in entity_types} # TODO： 按照分布给权重

    for tmp_pred, tmp_gt in zip(pred_tokens, gt_tokens):

        tmp_metric = np.zeros([17, 3])

        # pred_label = [id2ent[t] for t in tmp_pred]
        # gt_label = [id2ent[g] for g in tmp_gt]
        # pred_label = fix(pred_label)
        # gt_label = fix(gt_label)

        pred_entities = classify_decode(tmp_pred, id2ent)
        gt_entities = classify_decode(tmp_gt, id2ent)

        entity_types_new = []
        # for et in entity_types:
        #     if et !='O':
        #         entity_types_new.extend([i + et for i in ['I-','B-','E-','S-']])

        for idx, _type in enumerate(entity_types):
            if _type not in pred_entities:
                pred_entities[_type] = []
            if _type not in gt_entities:
                gt_entities[_type] = []

            tmp_metric[idx] += calculate_metric(gt_entities[_type], pred_entities[_type])

        role_metric += tmp_metric

    for idx, _type in enumerate(entity_types):
        temp_metric = get_p_r_f(role_metric[idx][0], role_metric[idx][1], role_metric[idx][2])

        mirco_metrics += temp_metric * type_weight[_type]

    metric_str = f'[MIRCO] precision: {mirco_metrics[0]:.4f}, ' \
                 f'recall: {mirco_metrics[1]:.4f}, f1: {mirco_metrics[2]:.4f}'

    return metric_str, mirco_metrics[2]