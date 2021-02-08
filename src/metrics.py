from sklearn import metrics
import numpy as np
import tensorflow as tf
def sigmoid(x):
    return 1/(1+np.exp(np.clip(-x, a_min=-1e50, a_max=1e20)))

def cal_auc(label, pos_prob):
    fpr, tpr, thresholds = metrics.roc_curve(label, pos_prob, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    return auc

def stable_log1pex(x):
    return -np.minimum(x, 0) + np.log(1+np.exp(-np.abs(x)))

def cal_llloss_with_logits(label, logits):
    ll = -np.mean(label*(-stable_log1pex(logits)) + (1-label)*(-logits - stable_log1pex(logits)))
    return ll

def prob_clip(x):
    return np.clip(x, a_min=1e-20, a_max=1)

def cal_llloss_with_neg_log_prob(label, neg_log_prob):
    ll = -np.mean((1-label)*neg_log_prob + label*(np.log(prob_clip(1 - prob_clip(np.exp(neg_log_prob))))))
    return ll

def cal_llloss_with_prob(label, prob):
    ll = -np.mean(label*np.log(prob_clip(prob)) + (1-label)*(np.log(prob_clip(1-prob))))
    return ll

def ece_score(y_test, py, n_bins=10):
    py = np.array(py).reshape(-1)
    y_test = np.array(y_test).reshape(-1)

    temp = []
    for i in range(py.shape[0]):
        temp.append((py[i],y_test[i]))

    pairs = temp #= [(1, 'one'), (2, 'two'), (3, 'three'), (4, 'four')]
    pairs.sort(key=lambda pair: pair[0])

    #print y_test.shape
    pair0=[]
    pair1=[]
    for (it0, it1) in pairs:
        pair0.append(it0)
        pair1.append(it1)

    py = np.array(pair0).reshape(-1)
    y_test = np.array(pair1).reshape(-1)

    py_value = py #np.array(py_value)
    #print py_value
    acc, conf = np.zeros(n_bins), np.zeros(n_bins)
    Bm = np.zeros(n_bins)
    for m in range(n_bins):
        a, b = int(m*1. / n_bins*py.shape[0]), int((m + 1)*1. / n_bins*py.shape[0])
        for i in range(py.shape[0]):
            if i > a and i <=b:
                Bm[m] += 1
                acc[m] += y_test[i]
                conf[m] += py_value[i]
        if True:#Bm[m] != 0:
            #print 'acc conf 1 ', a, b
            #print Bm[m],acc[m], conf[m]
            acc[m] = acc[m] / Bm[m]
            conf[m] = conf[m] / Bm[m]
    ece = 0
    for m in range(n_bins):
        ece += Bm[m] * np.abs((acc[m] - conf[m]))
    return ece / sum(Bm)

def cal_softmax_cross_entropy_loss(targets, outputs):
    z = targets #["label"]
    x = outputs #["logits"]
    x = tf.reshape(x, (-1,3))
    z = tf.cast(z, tf.float32)
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=z, logits=x))
    return loss


def cal_prauc(label, pos_prob):
    precision, recall, thresholds = metrics.precision_recall_curve(label, pos_prob)
    area = metrics.auc(recall, precision)
    return area

def cal_acc(label, prob):
    label = np.reshape(label, (-1,))
    prob = np.reshape(label, (-1,))
    prob_acc = np.mean(label*prob)
    return prob_acc

def stable_softplus(x):
    return np.log(1 + np.exp(-np.abs(x))) + np.maximum(x,0)
