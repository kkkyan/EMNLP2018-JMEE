import gc
import json
import math
import sys

import torch
import torch.nn as nn
from torch.nn import init

from enet.corpus.Sentence import Token


class SparseMM(torch.autograd.Function):
    """
    Sparse x dense matrix multiplication with autograd support.

    Implementation by Soumith Chintala:
    https://discuss.pytorch.org/t/
    does-pytorch-support-autograd-on-sparse-matrix/6156/7
    """

    def __init__(self, sparse):
        super(SparseMM, self).__init__()
        self.sparse = sparse

    def forward(self, dense):
        return torch.mm(self.sparse, dense)

    def backward(self, grad_output):
        grad_input = None
        if self.needs_input_grad[0]:
            grad_input = torch.mm(self.sparse.t(), grad_output)
        return grad_input


class Bottle(nn.Module):
    ''' Perform the reshape routine before and after an operation '''

    def forward(self, input):
        if len(input.size()) <= 2:
            return super(Bottle, self).forward(input)
        size = input.size()[:2]
        out = super(Bottle, self).forward(input.view(size[0] * size[1], -1))
        return out.view(size[0], size[1], -1)


class BatchBottle(nn.Module):
    ''' Perform the reshape routine before and after an operation '''

    def forward(self, input):
        if len(input.size()) <= 2:
            return super(BatchBottle, self).forward(input)
        size = input.size()[1:]
        out = super(BatchBottle, self).forward(input.view(-1, size[0] * size[1]))
        return out.view(-1, size[0], size[1])


class XavierLinear(nn.Module):
    '''
    Simple Linear layer with Xavier init

    Paper by Xavier Glorot and Yoshua Bengio (2010):
    Understanding the difficulty of training deep feedforward neural networks
    http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
    '''

    def __init__(self, in_features, out_features, bias=True):
        super(XavierLinear, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        init.xavier_normal_(self.linear.weight)

    def forward(self, x):
        return self.linear(x)


class OrthogonalLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(OrthogonalLinear, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        init.orthogonal_(self.linear.weight)

    def forward(self, x):
        return self.linear(x)


class BottledLinear(Bottle, nn.Linear):
    pass


class BottledXavierLinear(Bottle, XavierLinear):
    pass


class BottledOrthogonalLinear(Bottle, OrthogonalLinear):
    pass


def log(*args, **kwargs):
    print(file=sys.stdout, flush=True, *args, **kwargs)


def logerr(*args, **kwargs):
    print(file=sys.stderr, flush=True, *args, **kwargs)


def logonfile(fp, *args, **kwargs):
    fp.write(*args, **kwargs)


def progressbar(cur, total, other_information):
    percent = '{:.2%}'.format(cur / total)
    if type(other_information) is str:
        log("\r[%-50s] %s %s" % ('=' * int(math.floor(cur * 50 / total)), percent, other_information))
    else:
        log("\r[%-50s] %s" % ('=' * int(math.floor(cur * 50 / total)), percent))


def save_hyps(hyps, fp):
    json.dump(hyps, fp)


def load_hyps(fp):
    hyps = json.load(fp)
    return hyps


def add_tokens(words, y, y_, x_len, all_tokens, word_i2s, label_i2s):
    words = words.tolist()
    for s, ys, ys_, sl in zip(words, y, y_, x_len):
        s = s[:sl]
        ys = ys[:sl]
        ys_ = ys_[:sl]
        tokens = []
        for w, yw, yw_ in zip(s, ys, ys_):
            atoken = Token(word=word_i2s[w], triggerLabel=label_i2s[yw])
            atoken.addPredictedLabel(label_i2s[yw_])
            tokens.append(atoken)
        all_tokens.append(tokens)


def run_over_data(model, optimizer, data_iter, MAX_STEP, need_backward, tester, hyps, device, word_i2s, label_i2s,
                  role_i2s, maxnorm, weight, ae_weight, save_output, back_step = 1):
    if need_backward:
        model.test_mode_off()
    else:
        model.test_mode_on()

    running_loss = 0.0

    # 事件标签
    e_y = []
    e_y_ = []
    # argument label
    ae_y = []
    ae_y_= []
    # 字典集
    all_events = []
    all_events_ = []
    
    cnt = 0
    for batch in data_iter:
        optimizer.zero_grad()
        cnt += 1

        tokens, x_len = batch.WORDS
        words = batch.WLIST
        
        labels = batch.LABEL
        entities = batch.ENTITIES
        
        # 获取所有 events
        events = batch.EVENT
        all_events.extend(events)

        tokens = tokens.to(tokens)
        x_len = x_len.to(device)
        labels = labels.to(device)

        # forward
        trigger_logits, ent_logits, ent_keys \
            = model.forward(tokens, words, x_len, entities, label_i2s, events)
        
        # calculate loss
        loss_ed, trigger = model.calculate_loss_ed(labels, trigger_logits, words, weight)
        if len(ent_keys) != 0:
            ae_ = torch.argmax(ent_logits, 1).tolist()
            loss_ae, events_, ae = model.calculate_loss_ae(events, ent_logits, ent_keys, ae_weight)
            loss = loss_ed + hyps["loss_alpha"] * loss_ae
        else:
            loss = loss_ed
            ae = []
            ae_ = []
        
        # metrics
        trigger_ = torch.argmax(trigger_logits, 1).tolist()

        summary = tester.summary_report(trigger, trigger_, ae, ae_, label_i2s, role_i2s)
        
        e_y.extend(trigger)
        e_y_.extend(trigger_)
        ae_y.extend(ae)
        ae_y_.extend(ae_)

        other_information = ""

        if need_backward:
            if cnt % back_step == 0:
                loss.backward()
                if 1e-6 < maxnorm and model.parameters_requires_grad_clipping() is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters_requires_grad_clipping(), maxnorm)
    
                optimizer.step()
                other_information = 'Iter[{}] loss: {:.6f} TI: {:.4f}% TC: {:.4f}% AI: {:.4f}% AC: {:.4f}%'\
                                        .format(cnt,
                                                loss.item(),
                                                summary["t-i"][-1] * 100.0,
                                                summary["t-c"][-1] * 100.0,
                                                summary["a-i"][-1] * 100.0,
                                                summary["a-c"][-1] * 100.0,
                                                )
                optimizer.zero_grad()
            else:
                gc.collect()
                other_information = "Iter[{}] gc collect".format(cnt)
                
        progressbar(cnt, MAX_STEP, other_information)
        running_loss += loss.item()

    # if save_output:
    #     with open(save_output, "w", encoding="utf-8") as f:
    #         for tokens in all_tokens:
    #             for token in tokens:
    #                 # to match conll2000 format
    #                 f.write("%s %s %s\n" % (token.word, token.triggerLabel, token.predictedLabel))
    #             f.write("\n")

    running_loss = running_loss / cnt
    all_summary = tester.summary_report(e_y, e_y_, ae_y, ae_y, label_i2s, role_i2s)
    
    def display(report):
        d = lambda s: print(s.center(30, "-"))
        d("")
        d(" loss : {:.6f} ".format(running_loss))
        d(" Trigger Identification ")
        d(" P: {:.6f} R: {:.6f} F1: {:.6f}".format(*all_summary["t-i"]))
        d(" Trigger Classification ")
        d(" P: {:.6f} R: {:.6f} F1: {:.6f}".format(*all_summary["t-c"]))
        d(" Argument Identification ")
        d(" P: {:.6f} R: {:.6f} F1: {:.6f}".format(*all_summary["a-i"]))
        d(" Argument Classification ")
        d(" P: {:.6f} R: {:.6f} F1: {:.6f}".format(*all_summary["a-c"]))
    
    display(all_summary)
    return running_loss, all_summary["t-c"][-1], all_summary["a-c"][-1]

