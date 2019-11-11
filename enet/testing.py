from seqeval.metrics import f1_score, precision_score, recall_score
from sklearn.metrics import classification_report, precision_recall_fscore_support
import numpy as np

class EDTester():
    def __init__(self, voc_i2s):
        self.voc_i2s = voc_i2s
        
    def summary_report(self, trigger, trigger_, ent, ent_, label_i2s, role_i2s, output_dict=True):
        ret = {}
        # trigger identification
        ret["t-i"] = (self._identification(trigger, trigger_))
        # trigger classification
        ret["t-c"] = self._classification(trigger, trigger_, label_i2s)
        # argument identification
        ret["a-i"] = (self._identification(ent, ent_))
        # argument classification
        ret["a-c"] = self._classification(ent, ent_, role_i2s)
        
        return ret

    def binarize_label(self, labels):
        np_labels = np.array(labels)
        np_labels[np_labels > 0] = 1
        
        return np_labels.tolist()
    
    def _identification(self, y_true, y_pred):
        if len(y_true) == 0:
            return 0, 0, 0
        
        if len(set(y_true)) == 1 and y_true[0] == 0:
            return 0, 0, 0
        
        y_true = self.binarize_label(y_true)
        y_pred = self.binarize_label(y_pred)
        p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
        
        return p, r, f1
    
    def _classification(self, y_true, y_pred, label_i2s=None, exclude_other=True, output_dict=True):
        if len(y_true) == 0:
            return 0, 0, 0
    
        if len(set(y_true)) == 1 and y_true[0] == 0:
            return 0, 0, 0
        
        labels = None
        if label_i2s:
            labels = [i for i in range(len(label_i2s))]
            if exclude_other:
                labels = labels[1:]
                label_i2s = label_i2s[1:]
        
        report = classification_report(y_true, y_pred, digits=4,
                                       labels=labels, target_names=label_i2s, output_dict=output_dict)
        
        if output_dict:
            p = report["weighted avg"]["precision"]
            r = report["weighted avg"]["recall"]
            f = report["weighted avg"]["f1-score"]
            return p, r, f
        
        return report
    
    def argument_identification(self, events, events_):
        pass
    
    def argument_classification(self, events, events_, role_i2s):
        pass
