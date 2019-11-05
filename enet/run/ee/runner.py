import argparse
import os
import pickle
import sys
from functools import partial

import numpy as np
import torch
from tensorboardX import SummaryWriter
from torchtext.data import Field
from torchtext.vocab import Vectors

from enet import consts
from enet.corpus.Data import ACE2005Dataset, MultiTokenField, SparseField, EventField, EntityField
from enet.models.ee import EDModel
from enet.testing import EDTester
from enet.training import train
from enet.util import log
from pytorch_pretrained_bert import BertTokenizer


class EERunner(object):
    def __init__(self):
        parser = argparse.ArgumentParser(description="neural networks trainer")
        parser.add_argument("--test", help="validation set", default="../../../ace-05-splits/test.json")
        parser.add_argument("--train", help="training set", default="../../../ace-05-splits/train.json", required=False)
        parser.add_argument("--dev", help="development set", required=False, default="../../../ace-05-splits/dev.json")
        parser.add_argument("--webd", help="word embedding", required=False, default="../../../embedding/glove.6B.300d.txt")

        parser.add_argument("--batch", help="batch size", default=16, type=int)
        parser.add_argument("--epochs", help="n of epochs", default=99999, type=int)

        parser.add_argument("--seed", help="RNG seed", default=42, type=int)
        parser.add_argument("--lb_weight", help="label weight", default=1, type=int)
        parser.add_argument("--ae_lb_weight", help="label weight", default=5, type=int)
        parser.add_argument("--optimizer", default="adadelta")
        parser.add_argument("--lr", default=0.5, type=float)
        parser.add_argument("--l2decay", default=1e-5, type=float)
        parser.add_argument("--maxnorm", default=3, type=float)

        parser.add_argument("--out", help="output model path", default="out")
        parser.add_argument("--finetune", help="pretrained model path")
        parser.add_argument("--earlystop", default=10, type=int)
        parser.add_argument("--restart", default=999999, type=int)

        parser.add_argument("--device", default="cuda:0")
        parser.add_argument("--back_step", default=1, type=int)
        parser.add_argument("--hps", help="model hyperparams", required=False, default="{'wemb_dim': 300, 'wemb_ft': True, 'wemb_dp': 0.5, 'psemb_dim': 50, 'psemb_dp': 0.5, 'efemb_dim': 50, 'efemb_dp': 0.5, 'lstm_dim': 220, 'lstm_layers': 1, 'lstm_dp': 0, 'lstm_use_bn':True, 'gcn_et': 3, 'gcn_use_bn': True, 'gcn_layers': 0, 'gcn_dp': 0.5, 'sa_dim': 300, 'use_highway': True, 'loss_alpha': 5}")

        self.a = parser.parse_args()

    def set_device(self, device="cpu"):
        self.device = torch.device(device)
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def get_device(self):
        return self.device

    def load_model(self, fine_tune, embeddingMatrix=None):

        mymodel = None
        if fine_tune is None:
            mymodel = EDModel(self.a.hps,self.get_device(), embeddingMatrix)
        else:
            mymodel = EDModel(self.a.hps,self.get_device())
            mymodel.load_model(fine_tune)

        mymodel.to(self.get_device())
        
        return mymodel
    
    def get_tester(self, voc_i2s):
        return EDTester(voc_i2s)

    def run(self):
        print("Running on", self.a.device)
        self.set_device(self.a.device)

        np.random.seed(self.a.seed)
        torch.manual_seed(self.a.seed)

        # create training set
        if self.a.train:
            log('loading corpus from %s' % self.a.train)

        def tokenize(text):
            tokens = tokenizer.tokenize(text)
            ids = tokenizer.convert_tokens_to_ids(tokens)
            return ids
        
        tokenizer = BertTokenizer.from_pretrained("/home/yk/.pytorch_pretrained_bert/bert-base-uncased",
                                                  never_split=["[CLS]", "[SEP]",  "[unused0]"])
        WordsField = Field(lower=False, batch_first=True, include_lengths=True,
                           tokenize=(lambda s: tokenize(s)), pad_token=0, use_vocab=False)
        LabelField = Field(sequential=False, lower=False, batch_first=True, pad_token=None, unk_token=None)
        EventsField = EventField(lower=False, batch_first=True)
        EntitiesField = EntityField(lower=False, batch_first=True, use_vocab=False)

        # 这里的 fields 会自动映射 json 文件里的结果
        train_set = ACE2005Dataset(path=self.a.train, min_len=10,
                                   fields={"words": ("WORDS", WordsField),
                                           # "sentence": ("SENTENCE", SentenceField),
                                           "golden-event-mentions": ("LABEL", LabelField),
                                           "all-events": ("EVENT", EventsField),
                                           "all-entities": ("ENTITIES", EntitiesField)},
                                   keep_events=0)
        

        dev_set = ACE2005Dataset(path=self.a.dev,
                                 fields={"words": ("WORDS", WordsField),
                                         # "sentence": ("SENTENCE", SentenceField),
                                         "golden-event-mentions": ("LABEL", LabelField),
                                         "all-events": ("EVENT", EventsField),
                                         "all-entities": ("ENTITIES", EntitiesField)},
                                 keep_events=0)

        
        # WordsField.build_vocab(tokenizer.vocab.keys())
        LabelField.build_vocab(train_set.LABEL)
        EventsField.build_vocab(train_set.EVENT)

        consts.O_LABEL = LabelField.vocab.stoi["O"]
        # print("O label is", consts.O_LABEL)
        consts.ROLE_O_LABEL = EventsField.vocab.stoi["OTHER"]
        # print("O label for AE is", consts.ROLE_O_LABEL)
        
        test_set = ACE2005Dataset(path=self.a.test,
                                  fields={"words": ("WORDS", WordsField),
                                          # "sentence": ("SENTENCE", SentenceField),
                                          "golden-event-mentions": ("LABEL", LabelField),
                                          "all-events": ("EVENT", EventsField),
                                          "all-entities": ("ENTITIES", EntitiesField)},
                                  keep_events=0)

        # 这里给label加了权重
        self.a.label_weight = torch.ones([len(LabelField.vocab.itos)]) * self.a.lb_weight
        self.a.label_weight[consts.O_LABEL] = 1.0
        self.a.ae_label_weight = torch.ones([len(EventsField.vocab.itos)]) * self.a.ae_lb_weight
        self.a.ae_label_weight[consts.ROLE_O_LABEL] = 1.0

        self.a.hps = eval(self.a.hps)
        
        # 词向量大小
        # 事件预测空间
        if "oc" not in self.a.hps:
            self.a.hps["oc"] = len(LabelField.vocab.itos)
        # 事件种类空间
        if "ae_oc" not in self.a.hps:
            self.a.hps["ae_oc"] = len(EventsField.vocab.itos)
        
        # 评测
        tester = self.get_tester(LabelField.vocab.itos)

        if self.a.finetune:
            log('init model from ' + self.a.finetune)
            model = self.load_model(self.a.finetune)
            log('model loaded, there are %i sets of params' % len(model.parameters_requires_grads()))
        else:
            model = self.load_model(None)
            log('model created from scratch, there are %i sets of params' % len(model.parameters_requires_grads()))

        if self.a.optimizer == "adadelta":
            optimizer_constructor = partial(torch.optim.Adadelta, params=model.parameters_requires_grads(),
                                            weight_decay=self.a.l2decay)
        elif self.a.optimizer == "adam":
            optimizer_constructor = partial(torch.optim.Adam, params=model.parameters_requires_grads(),
                                            weight_decay=self.a.l2decay)
        else:
            optimizer_constructor = partial(torch.optim.SGD, params=model.parameters_requires_grads(),
                                            weight_decay=self.a.l2decay,
                                            momentum=0.9)

        log('optimizer in use: %s' % str(self.a.optimizer))

        if not os.path.exists(self.a.out):
            os.mkdir(self.a.out)
        # with open(os.path.join(self.a.out, "word.vec"), "wb") as f:
        #     pickle.dump(WordsField.vocab, f)
        # with open(os.path.join(self.a.out, "pos.vec"), "wb") as f:
        #     pickle.dump(PosTagsField.vocab.stoi, f)
        # with open(os.path.join(self.a.out, "entity.vec"), "wb") as f:
        #     pickle.dump(EntityLabelsField.vocab.stoi, f)
        # with open(os.path.join(self.a.out, "label.vec"), "wb") as f:
        #     pickle.dump(LabelField.vocab.stoi, f)
        # with open(os.path.join(self.a.out, "role.vec"), "wb") as f:
        #     pickle.dump(EventsField.vocab.stoi, f)

        log('init complete\n')

        self.a.word_i2s = None
        self.a.label_i2s = LabelField.vocab.itos
        self.a.role_i2s = EventsField.vocab.itos
        writer = SummaryWriter(os.path.join(self.a.out, "exp"))
        self.a.writer = writer

        train(
            model=model,
            train_set=train_set,
            dev_set=dev_set,
            test_set=test_set,
            optimizer_constructor=optimizer_constructor,
            epochs=self.a.epochs,
            tester=tester,
            parser=self.a
        )
        log('Done!')


if __name__ == "__main__":
    EERunner().run()
