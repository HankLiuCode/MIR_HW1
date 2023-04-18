import mir_eval
from sklearn.metrics import accuracy_score
import numpy as np
import template

def raw_accuracy(ground_truth, prediction):
    if len(ground_truth) != len(prediction):
        print("ground truth must have the same length as prediction")
        return
    return accuracy_score(ground_truth, prediction)

def weighted_accuracy(ground_truth, prediction):
    if len(ground_truth) != len(prediction):
        print("ground truth must have the same length as prediction")
        return
    n = len(ground_truth)
    score = 0
    for i in range(n):
        score += mir_eval.key.weighted_score(ground_truth[i], prediction[i])
    return score / n

def sum_of_chroma(chroma_feat, start=None, end=None):
    start = 0 if (start == None or start < 0) else start
    end = chroma_feat.shape[1] if (end == None or end < 0) else end
    chroma_feat_t = np.transpose(chroma_feat)
    segment = chroma_feat_t[start:end+1]
    return sum(segment)

def convert(val):
    d = {"min": "minor", "maj":"major"}
    note, mm = val.split(":")
    if note in template.flat2sharp:
        return f'{template.flat2sharp[note]} {d[mm]}'
    else:
        return f'{note} {d[mm]}'