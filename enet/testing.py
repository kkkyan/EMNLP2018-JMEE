from seqeval.metrics import f1_score, precision_score, recall_score


class EDTester():
    def __init__(self, voc_i2s):
        self.voc_i2s = voc_i2s

    def calculate_report(self, y, y_, transform=True):
        '''
        calculating F1, P, R

        :param y: golden label, list
        :param y_: model output, list
        :return:
        '''
        if transform:
            for i in range(len(y)):
                for j in range(len(y[i])):
                    y[i][j] = self.voc_i2s[y[i][j]]
            for i in range(len(y_)):
                for j in range(len(y_[i])):
                    y_[i][j] = self.voc_i2s[y_[i][j]]
        return precision_score(y, y_), recall_score(y, y_), f1_score(y, y_)

    @staticmethod
    def merge_segments(y):
        '''
        :param y:  y是一个BIO序列，标注了事件label
        :return: segs: segs 是一个字典
            key => event label start index
            value => [ed, tt] : ed is event label end index, tt is the event type
        '''
        segs = {}
        tt = ""
        st, ed = -1, -1
        for i, x in enumerate(y):
            # B 开头
            if x.startswith("B-"):
                if tt == "":
                    # 获得事件类型
                    tt = x[2:]
                    # 起始位置
                    st = i
                else:
                    # 这个 else 处理了两个 B- 连续的情况
                    ed = i
                    segs[st] = (ed, tt)
                    # 新 B-
                    tt = x[2:]
                    st = i
            # I 开头
            elif x.startswith("I-"):
                # 预测出了 I- 却丢了 B-
                if tt == "":
                    ''' 源代码， 把第一个I修订成B'''
                    # y[i] = "B" + y[i][1:]
                    # tt = x[2:]
                    # st = i
                    
                    ''' 我的处理: 没有遵循BIO就跳过它 '''
                    pass
                else:
                    # 这个 if 属于连续出现I但是后面的I和前面不同
                    if tt != x[2:]:
                        ed = i
                        segs[st] = (ed, tt)
                        ''' 源代码， 把第一个I修订成B'''
                        # y[i] = "B" + y[i][1:]
                        # tt = x[2:]
                        # st = i
                        
                        ''' 我的处理: 没有遵循BIO就跳过它 '''
            # O 开头
            else:
                ed = i
                if tt != "":
                    segs[st] = (ed, tt)
                tt = ""

        # 最后处理一下尾部, 可能有非O结尾的情况
        if tt != "":
            segs[st] = (len(y), tt)
            
        return segs

    def calculate_sets(self, y, y_):
        ct, p1, p2 = 0, 0, 0
        for sent, sent_ in zip(y, y_):
            for key, value in sent.items():
                p1 += len(value)
                if key not in sent_:
                    continue
                # matched sentences
                arguments = value
                arguments_ = sent_[key]

                ct += len(set(arguments) & set(arguments_))  # count any argument in golden
                
                # for item, item_ in zip(arguments, arguments_):
                #     if item[2] == item_[2]:
                #         ct += 1

            for key, value in sent_.items():
                p2 += len(value)

        if ct == 0 or p1 == 0 or p2 == 0:
            return 0.0, 0.0, 0.0
        else:
            p = 1.0 * ct / p2
            r = 1.0 * ct / p1
            f1 = 2.0 * p * r / (p + r)
            return p, r, f1
