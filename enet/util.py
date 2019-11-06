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

        words, x_len = batch.WORDS
        labels = batch.LABEL

        BATCH = words.size()[0]
        SEQ_LEN = words.size()[1]
        
        # 获取所有 events
        # events = batch.EVENT
        # all_events.extend(events)

        words = words.to(device)
        x_len = x_len.to(device)
        labels = labels.to(device)

        # forward
        y_, mask = model.forward(words, x_len, labels)
        
        # calculate loss
        # Now we just have ar loss
        loss = model.calculate_loss(y_, labels, weight)

        y__ = torch.max(y_, 1)[1].tolist()
        y = labels.tolist()

        # add_tokens(words, y, y__, x_len, all_tokens, word_i2s, label_i2s)

        # unpad
        label_i = [x for x in range(len(label_i2s))]
        p, r, f1, _ = tester.summary_report(y, y__, label_i2s)
        
        ae_y.extend(y)
        ae_y_.extend(y__)

        other_information = ""

        if need_backward:
            if cnt % back_step == 0:
                loss.backward()
                if 1e-6 < maxnorm and model.parameters_requires_grad_clipping() is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters_requires_grad_clipping(), maxnorm)
    
                optimizer.step()
                other_information = 'Iter[{}] loss: {:.6f} edP: {:.4f}% edR: {:.4f}% edF1: {:.4f}%'.format(cnt, loss.item(),
                                                                                                       p * 100.0,
                                                                                                       r * 100.0,
                                                                                                       f1 * 100.0)
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
    pp, rr, ff, report = tester.summary_report(ae_y, ae_y_, label_i2s)
    print(report)
    # ap, ar, af = tester.calculate_sets(all_events, all_events_)
    
    return running_loss, pp, rr, ff
