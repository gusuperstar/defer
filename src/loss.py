import tensorflow as tf
import numpy as np
from tensorflow.python.ops import array_ops, math_ops


def stable_log1pex(x):
    return -tf.minimum(x, 0) + tf.math.log(1+tf.math.exp(-tf.abs(x)))

def stable_sigmoid_pos(x):
    return tf.where(x>=0, 1.0/ ( 1 + tf.math.exp(-x)), 1 - 1.0 / (1 + tf.math.exp(x)))
    #    return 1.0/ ( 1 + tf.math.exp(-x))
    #else:
    #    return 1 - 1.0 / (1 + tf.math.exp(x))

def stable_sigmoid_neg(x):
    return tf.where(x>=0, 1.0/ ( 1 + tf.math.exp(x)), 1 - 1.0 / (1 + tf.math.exp(-x)))
    #if x >= 0 :
    #    return 1.0 / (1 + tf.math.exp(x))
    #else:
    #    return tf.math.exp(-x) / (1 + tf.math.exp(-x))

def fake_negative_weighted_loss(targets, outputs, params=None):
    z = targets["label"]
    x = outputs["logits"]
    x = tf.reshape(x, (-1,))
    z = tf.cast(z, tf.float32)
    p_no_grad = tf.sigmoid(tf.stop_gradient(x))
    pos_loss = (1+p_no_grad)*stable_log1pex(x)
    neg_loss = -(1-p_no_grad)*(1+p_no_grad)*(-x-stable_log1pex(x))
    loss = tf.reduce_mean(pos_loss*z + neg_loss*(1-z))
    return {"loss": loss}

def newfnw_test_loss(targets, outputs, params=None):
    z = tf.cast(targets["label"], tf.float32)
    labels = z
    #print 'z.shape'
    #print z.shape
    label_01 = tf.reshape(z[:, 0], (-1, 1))
    label_00 = tf.reshape(z[:, 1], (-1, 1))
    label_11 = tf.reshape(z[:, 2], (-1, 1))
    z = tf.minimum(label_01+label_11,1)
    #print 'z label'
    #print 
    #z = targets["label"]
    x = outputs["logits"]
    x = tf.reshape(x[:,0], (-1,))
    z = tf.cast(z, tf.float32)
    p_no_grad = tf.sigmoid(tf.stop_gradient(x))
    pos_loss = (1+p_no_grad)*stable_log1pex(x)
    neg_loss = -(1-p_no_grad)*(1+p_no_grad)*(-x-stable_log1pex(x))
    loss = tf.reduce_mean(pos_loss*z + neg_loss*(1-z))
    return {"loss": loss}

def fnw10_test_loss(targets, outputs, params=None):
    print "fnw10_test_loss"
    z = targets["label"]
    x = outputs["logits"]
    x = tf.reshape(x[:,0], (-1,))
    z = tf.cast(z, tf.float32)
    p_no_grad = stable_sigmoid_pos(tf.stop_gradient(x))
    #pos_loss = (1+p_no_grad)*stable_log1pex(x)
    #neg_loss = -(1-p_no_grad)*(1+p_no_grad)*(-x-stable_log1pex(x))
    pos_loss = -2 * tf.math.log(stable_sigmoid_pos(x))  #stable_log1pex(x)
    neg_loss = -2*(1-p_no_grad)*tf.math.log(1-stable_sigmoid_pos(x))/(2-p_no_grad)
    loss = tf.reduce_mean(pos_loss*z + neg_loss*(1-z))
    loss*=0.5
    return {"loss": loss}

def fnw_test_loss(targets, outputs, params=None):
    print 'fnw_test_loss'
    z = targets["label"]
    x = outputs["logits"]
    x = tf.reshape(x[:,0], (-1,))
    z = tf.cast(z, tf.float32)
    #p_no_grad = tf.sigmoid(tf.stop_gradient(x))
    #pos_loss = (1+p_no_grad)*stable_log1pex(x)
    #neg_loss = -(1-p_no_grad)*(1+p_no_grad)*(-x-stable_log1pex(x))
    p_no_grad = stable_sigmoid_pos(tf.stop_gradient(x))
    pos_loss = -(1+p_no_grad)*tf.math.log(stable_sigmoid_pos(x))
    neg_loss = -(1-p_no_grad)*(1+p_no_grad)*tf.math.log(1-stable_sigmoid_pos(x))
    loss = tf.reduce_mean(pos_loss*z + neg_loss*(1-z))
    #loss*=3
    return {"loss": loss}
'''
def fake_negative_weighted_loss(targets, outputs, params=None):
    z = targets["label"]
    x = outputs["logits"]
    x = tf.reshape(x, (-1,))
    z = tf.cast(z, tf.float32)
    p_no_grad = tf.sigmoid(tf.stop_gradient(x))
    pos_loss = (1+p_no_grad)*stable_log1pex(x)
    neg_loss = -(1-p_no_grad)*(1+p_no_grad)*(-x-stable_log1pex(x))
    loss = tf.reduce_mean(pos_loss*z + neg_loss*(1-z))
    return {"loss": loss}
'''

def cross_entropy_loss(targets, outputs, params=None):
    z = targets["label"]
    x = outputs["logits"]
    x = tf.reshape(x, (-1,))
    prop = tf.sigmoid(tf.cast(outputs["logits"], tf.float32))
    z = tf.cast(z, tf.float32)
    #loss = tf.reduce_mean(
    #    tf.nn.sigmoid_cross_entropy_with_logits(labels=z, logits=x))
    #loss = -tf.reduce_mean(tf.math.log(prop) * z + tf.math.log(1-prop)* (1-z))
    #loss = tf.reduce_mean(x-x*z+tf.math.log(1+tf.math.exp(-x)))
    #loss = tf.reduce_mean(tf.maximum(x,0)-x*z+tf.math.log(1+tf.math.exp(-tf.abs(x))))
    loss = tf.reduce_mean(tf.maximum(x,0)-x*z+tf.math.log(1+tf.math.exp(-tf.abs(x))))
    return {"loss": loss}

def cross_entropy_loss_test(targets, outputs, params=None):
    z = targets["label"]
    x = outputs["logits"]
    #x = tf.reshape(x, (-1,))
    x = tf.cast(tf.reshape(outputs["logits"], (-1, 1)), tf.float32)
    prop = tf.sigmoid(tf.cast(outputs["logits"], tf.float32))
    z = tf.cast(z, tf.float32)
    #loss = tf.reduce_mean(
    #    tf.nn.sigmoid_cross_entropy_with_logits(labels=z, logits=x))
    #loss = -tf.reduce_mean(tf.math.log(prop) * z + tf.math.log(1-prop)* (1-z))
    #loss = tf.reduce_mean(x-x*z+tf.math.log(1+tf.math.exp(-x)))
    #loss = tf.reduce_mean(tf.maximum(x,0)-x*z+tf.math.log(1+tf.math.exp(-tf.abs(x))))
    loss = tf.reduce_mean(tf.maximum(x,0)-x*z+tf.math.log(1+tf.math.exp(-tf.abs(x))))
    return {"loss": loss}

def cross_entropy_loss_00(targets, outputs, params=None):
    z = targets["label"]
    x = outputs["logits"]
    #x = tf.reshape(x, (-1,))
    x = tf.cast(tf.reshape(outputs["logits"], (-1, 1)), tf.float32)
    prop = tf.sigmoid(tf.cast(outputs["logits"], tf.float32))
    #z = tf.cast(z[:,0], tf.float32)
    z = tf.reshape(tf.cast(z[:, 0], tf.float32), (-1, 1))
    #loss = tf.reduce_mean(
    #    tf.nn.sigmoid_cross_entropy_with_logits(labels=z, logits=x))
    #loss = -tf.reduce_mean(tf.math.log(prop) * z + tf.math.log(1-prop)* (1-z))
    #loss = tf.reduce_mean(x-x*z+tf.math.log(1+tf.math.exp(-x)))
    #loss = tf.reduce_mean(tf.maximum(x,0)-x*z+tf.math.log(1+tf.math.exp(-tf.abs(x))))
    loss = tf.reduce_mean(tf.maximum(x,0)-x*z+tf.math.log(1+tf.math.exp(-tf.abs(x))))
    return {"loss": loss}

def softmax_cross_entropy_loss(targets, outputs, params=None):
    z = targets["label"]
    x = outputs["logits"]
    x = tf.reshape(x, (-1,3))
    z = tf.cast(z, tf.float32)
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=z, logits=x))
    return {"loss": loss}

def win_time_loss_test(targets, outputs, params=None):
    z = targets["label"]
    #cv_logit = tf.cast(outputs["cv_logits"], tf.float32)
    #time_logit = tf.cast(outputs["time_logits"], tf.float32)
    cv_prop = tf.sigmoid(tf.cast(outputs["logits"], tf.float32))
    time_prop = tf.sigmoid(tf.cast(outputs["logits"], tf.float32))

    stable_cv_prop = stable_sigmoid_pos(tf.cast(outputs["logits"], tf.float32))
    stable_time_prop = stable_sigmoid_pos(tf.cast(outputs["logits"], tf.float32))

    win_label = tf.reshape(tf.cast(z[:, 1], tf.float32), (-1, 1))
    cv_label = tf.reshape(tf.cast(z[:, 0], tf.float32), (-1, 1))

    stable_cv_prop_stop = tf.stop_gradient(stable_cv_prop)
    stable_time_prop_stop = tf.stop_gradient(stable_time_prop)

    win_prop_1 = cv_prop * time_prop
    win_prop_0 = 1- cv_prop * time_prop #(1-cv_prop) + cv_prop*(1-time_prop)

    stable_win_prop_1 = stable_cv_prop * stable_time_prop
    stable_win_prop_0 = 1 - stable_win_prop_1

    stable_win_prop_1_stop = stable_cv_prop_stop * stable_time_prop
    stable_win_prop_0_stop = 1 - stable_win_prop_1_stop

    #classic
    loss_win = -tf.reduce_mean(tf.math.log(stable_win_prop_1) * win_label + tf.math.log(stable_win_prop_0) * (1-win_label))
    #stop
    loss_win = -tf.reduce_mean(tf.math.log(stable_win_prop_1_stop) * win_label + tf.math.log(stable_win_prop_0_stop) * (1-win_label))
    loss_cv = -tf.reduce_mean(tf.math.log(stable_cv_prop) * cv_label + tf.math.log(1-stable_cv_prop)* (1-cv_label))

    loss = loss_cv
    return {"loss": loss}


def win_time_loss(targets, outputs, params=None):
    z = targets["label"]
    cv_logit = tf.cast(outputs["cv_logits"], tf.float32)
    time_logit = tf.cast(outputs["time_logits"], tf.float32)
    cv_prop = tf.sigmoid(tf.cast(outputs["cv_logits"], tf.float32))
    time_prop = tf.sigmoid(tf.cast(outputs["time_logits"], tf.float32))
  
    stable_cv_prop = stable_sigmoid_pos(tf.cast(outputs["cv_logits"], tf.float32))
    stable_time_prop = stable_sigmoid_pos(tf.cast(outputs["time_logits"], tf.float32))

    win_label = tf.reshape(tf.cast(z[:, 1], tf.float32), (-1, 1))
    cv_label = tf.reshape(tf.cast(z[:, 0], tf.float32), (-1, 1))
    cv_mask = tf.reshape(tf.cast(z[:, 2], tf.float32), (-1, 1))

    stable_cv_prop_stop = tf.stop_gradient(stable_cv_prop)
    stable_time_prop_stop = tf.stop_gradient(stable_time_prop)

    win_prop_1 = cv_prop * time_prop
    win_prop_0 = 1- cv_prop * time_prop #(1-cv_prop) + cv_prop*(1-time_prop)

    stable_win_prop_1 = stable_cv_prop * stable_time_prop
    stable_win_prop_0 = 1 - stable_win_prop_1

    #stable_win_prop_1_stop = stable_cv_prop_stop * stable_time_prop
    #stable_win_prop_0_stop = 1 - stable_win_prop_1_stop

    stable_win_prop_1_stop = cv_mask*stable_cv_prop*stable_time_prop + (1-cv_mask)*stable_cv_prop*stable_time_prop_stop
    stable_win_prop_0_stop =  1 - stable_win_prop_1_stop

    #loss_win = -tf.reduce_mean(tf.math.log(win_prop_1) * win_label + tf.math.log(win_prop_0) * (1-win_label))
    #classic
    #loss_win = -tf.reduce_mean(tf.math.log(stable_win_prop_1) * win_label + tf.math.log(stable_win_prop_0) * (1-win_label))
    #stop
    loss_win = -tf.reduce_mean(tf.math.log(stable_win_prop_1_stop) * win_label + tf.math.log(stable_win_prop_0_stop) * (1-win_label))
    #loss_win = -tf.reduce_mean(tf.math.log(cv_prop) * win_label + tf.math.log(time_prop) * win_label + tf.math.log(win_prop_0) * (1-win_label))
    #loss_win = tf.reduce_mean(stable_log1pex(cv_logit) *win_label + stable_log1pex(time_logit) *win_label + tf.math.log(win_prop_0) * (1-win_label))
    
    #loss_cv = -tf.reduce_mean(tf.math.log(cv_prop) * cv_label + tf.math.log(1-cv_prop)* (1-cv_label))
    loss_cv = -tf.reduce_mean(tf.math.log(stable_cv_prop) * cv_label*cv_mask + tf.math.log(1-stable_cv_prop)* (1-cv_label)*cv_mask)
    #loss_cv = -tf.reduce_mean(tf.math.log(stable_cv_prop) * cv_label + tf.math.log(1-stable_cv_prop)* (1-cv_label))
    #loss_cv = tf.reduce_mean(cv_prop-cv_prop*cv_label+tf.math.log(1+tf.math.exp(-cv_prop)))
    #loss_cv = tf.reduce_mean(tf.maximum(cv_logit,0)-cv_logit*cv_label+tf.math.log(1+tf.math.exp(-tf.abs(cv_logit))))
    #classic
    loss = 0.2*loss_win + 0.8*loss_cv
    #loss = loss_win + loss_cv
    #loss28 win_auc 0.8830976621671001 cv_auc 0.826153718691025

    #loss = loss_cv
    return {"loss": loss}

def win_select_loss(targets, outputs, params=None):
    z = targets["label"]
    #cv_logit = tf.cast(outputs["logits"][:, 0], tf.float32)
    #time15_logit = tf.cast(outputs["logits"][:, 1], tf.float32)
    #time30_logit = tf.cast(outputs["logits"][:, 2], tf.float32)
    #time60_logit = tf.cast(outputs["logits"][:, 3], tf.float32)

    cv_logit = tf.cast(tf.reshape(outputs["logits"][:, 0], (-1, 1)), tf.float32)
    time15_logit = tf.cast(tf.reshape(outputs["logits"][:, 1], (-1, 1)), tf.float32)
    time30_logit = tf.cast(tf.reshape(outputs["logits"][:, 2], (-1, 1)), tf.float32)
    time60_logit = tf.cast(tf.reshape(outputs["logits"][:, 3], (-1, 1)), tf.float32)

    #cv_prop = tf.sigmoid(cv_logit)
    #time15_prop = tf.sigmoid(time15_logit)
    #time30_prop = tf.sigmoid(time30_logit)
    #time60_prop = tf.sigmoid(time60_logit)

    stable_cv_prop = stable_sigmoid_pos(cv_logit)
    stable_time15_prop = stable_sigmoid_pos(time15_logit)
    stable_time30_prop = stable_sigmoid_pos(time30_logit)
    stable_time60_prop = stable_sigmoid_pos(time60_logit)

    cv_label = tf.reshape(tf.cast(z[:, 0], tf.float32), (-1, 1))
    win15_label = tf.reshape(tf.cast(z[:, 1], tf.float32), (-1, 1))
    win30_label = tf.reshape(tf.cast(z[:, 2], tf.float32), (-1, 1))
    win60_label = tf.reshape(tf.cast(z[:, 3], tf.float32), (-1, 1))

    win30_label = tf.minimum(win15_label+win30_label, 1)
    win60_label = tf.minimum(win15_label+win30_label+win60_label, 1)

    stable_win_prop_15_1 = stable_cv_prop * stable_time15_prop
    stable_win_prop_15_0 = 1 - stable_win_prop_15_1

    stable_win_prop_30_1 = stable_cv_prop * stable_time30_prop
    stable_win_prop_30_0 = 1 - stable_win_prop_30_1

    stable_win_prop_60_1 = stable_cv_prop * stable_time60_prop
    stable_win_prop_60_0 = 1 - stable_win_prop_60_1

    loss_cv = -tf.reduce_mean(tf.math.log(stable_cv_prop) * cv_label + tf.math.log(1-stable_cv_prop)* (1 - cv_label))
    loss_win_15 = -tf.reduce_mean(tf.math.log(stable_win_prop_15_1) * win15_label + tf.math.log(stable_win_prop_15_0) * (1-win15_label))
    loss_win_30 = -tf.reduce_mean(tf.math.log(stable_win_prop_30_1) * win30_label + tf.math.log(stable_win_prop_30_0) * (1-win30_label))
    loss_win_60 = -tf.reduce_mean(tf.math.log(stable_win_prop_60_1) * win60_label + tf.math.log(stable_win_prop_60_0) * (1-win60_label))

    #loss = 0.2*loss_win_15+0.05*loss_win_30+0.05*loss_win_60+0.7*loss_cv #loss1
    #loss = 0.2*loss_win_15+0.*loss_win_30+0.*loss_win_60+0.8*loss_cv  #loss28
    #loss = 0.1*loss_win_15+0.05*loss_win_30+0.05*loss_win_60+0.8*loss_cv #loss18
    loss = loss_win_15+loss_win_30+loss_win_60+loss_cv
    #loss18 win_auc 0.8845801776284682 cv_auc 0.826974685763819
    #loss *=5
    #3loss18 win_auc 0.8818380479139942 cv_auc 0.8268575218358292
    #5loss18 win_auc 0.8834395177853425 cv_auc 0.8269472993120556
    #loss18 lr002 win_auc 0.8834055796453607 cv_auc 0.8270327032222501


    #loss1 log/log_pre-winadapt6_lr0002 win_auc 0.8788218083886269 cv_auc 0.8239125328664904
    #loss1 log/log_pre-winadapt6_lr0005 win_auc 0.8800556828561409 cv_auc 0.8250883729028566
    #loss1 log/log_pre-winadapt6_lr0001 win_auc 0.8775036050783436 cv_auc 0.822684733937319
    #3loss1 win_auc0.8843443735591299 cv_auc 0.8263690089580688

    #loss = 0.05*loss_win_15+0.05*loss_win_30+0.05*loss_win_60+0.85*loss_cv #loss0585
    #loss = 0.1*loss_win_15+0.1*loss_win_30+0.1*loss_win_60+0.7*loss_cv #loss1117

    #loss = loss_cv
    return {"loss": loss}

def delay_win_select_loss(targets, outputs, params=None):
    z = targets["label"]

    print 'logits.shape'
    print outputs["logits"].shape

    cv_logit = tf.cast(tf.reshape(outputs["logits"][:, 0], (-1, 1)), tf.float32)
    time15_logit = tf.cast(tf.reshape(outputs["logits"][:, 1], (-1, 1)), tf.float32)
    time30_logit = tf.cast(tf.reshape(outputs["logits"][:, 2], (-1, 1)), tf.float32) 
    time60_logit = tf.cast(tf.reshape(outputs["logits"][:, 3], (-1, 1)), tf.float32)

    stable_cv_prop = tf.cast(stable_sigmoid_pos(cv_logit), tf.float32)
    stable_time15_prop = tf.cast(stable_sigmoid_pos(time15_logit), tf.float32)
    stable_time30_prop = tf.cast(stable_sigmoid_pos(time30_logit), tf.float32)
    stable_time60_prop = tf.cast(stable_sigmoid_pos(time60_logit), tf.float32)

    stable_cv_prop_stop = tf.stop_gradient(stable_cv_prop)
    stable_time15_prop_stop = tf.stop_gradient(stable_time15_prop)
    stable_time30_prop_stop = tf.stop_gradient(stable_time30_prop)
    stable_time60_prop_stop = tf.stop_gradient(stable_time60_prop)
   
    print 'delay_win_select_loss'
    print 'cv_logit'
    print cv_logit
    #print tf.cast(outputs["cv_logits"], tf.float32)
    print 'time15_logit'
    print time15_logit
    #print tf.cast(outputs["time_logits"], tf.float32)

    print 'prop all'
    print tf.reduce_mean(tf.concat([stable_cv_prop, stable_time15_prop, stable_time30_prop, stable_time60_prop], axis=1), axis=0)

    print 'loss z'
    print z.shape
    print tf.reduce_mean(z, axis=0)
    print z
    ###########train_label_11, train_label_10, train_label_01_15, train_label_01_30, train_label_01_60, label_01_30_mask, label_01_60_mask
    '''
    label01 = tf.cast(tf.reshape(z[:, 0], (-1, 1)), tf.float32)
    label00 = tf.cast(tf.reshape(z[:, 1], (-1, 1)), tf.float32)
    label11 = tf.cast(tf.reshape(z[:, 2], (-1, 1)), tf.float32)
    label10 = tf.cast(tf.reshape(z[:, 3], (-1, 1)), tf.float32)

    '''
    label11 = tf.reshape(tf.cast(z[:, 0], tf.float32), (-1, 1))
    label10 = tf.reshape(tf.cast(z[:, 1], tf.float32), (-1, 1))
    label_01_15 = tf.reshape(tf.cast(z[:, 2], tf.float32), (-1, 1))
    label_01_30 = tf.reshape(tf.cast(z[:, 3], tf.float32), (-1, 1))
    label_01_60 = tf.reshape(tf.cast(z[:, 4], tf.float32), (-1, 1))
    label_01_30_sum = tf.reshape(tf.cast(z[:, 5], tf.float32), (-1, 1))
    label_01_60_sum = tf.reshape(tf.cast(z[:, 6], tf.float32), (-1, 1))
    label_01_30_mask = tf.reshape(tf.cast(z[:, 7], tf.float32), (-1, 1))
    label_01_60_mask = tf.reshape(tf.cast(z[:, 8], tf.float32), (-1, 1))
    label00 = tf.reshape(tf.cast(z[:, 9], tf.float32), (-1, 1))
    label01 = tf.reshape(tf.cast(z[:, 10], tf.float32), (-1, 1))
    label_11_15 = tf.reshape(tf.cast(z[:, 11], tf.float32), (-1, 1))
    label_11_30 = tf.reshape(tf.cast(z[:, 12], tf.float32), (-1, 1))
    label_11_60 = tf.reshape(tf.cast(z[:, 13], tf.float32), (-1, 1))
    #'''

    #label_01_30 = tf.minimum(label_01_15 + label_01_30, 1)
    #label_01_60 = tf.minimum(label_01_15 + label_01_30 + label_01_60, 1)
    #win15_prob = stable_time15_prop
    #win30_prob = (1-stable_time15_prop) * stable_time30_prop
    #win60_prob = (1-stable_time15_prop) * (1 - stable_time30_prop) * stable_time60_prop

    stable_win_prop_15_1 = tf.cast(stable_cv_prop * stable_time15_prop, tf.float32)
    stable_win_prop_15_0 = tf.cast(1 - stable_win_prop_15_1, tf.float32)

    stable_win_prop_15_1_stop = tf.cast(stable_cv_prop * stable_time15_prop_stop, tf.float32)
    stable_win_prop_15_0_stop = tf.cast(1 - stable_win_prop_15_1_stop, tf.float32)
    #'''
    stable_win_prop_30_1 = tf.cast(stable_cv_prop * stable_time30_prop, tf.float32)
    stable_win_prop_30_0 = tf.cast(1 - stable_win_prop_30_1, tf.float32)

    stable_win_prop_30_1_stop = tf.cast(stable_cv_prop * stable_time30_prop_stop, tf.float32)
    stable_win_prop_30_0_stop = tf.cast(1 - stable_win_prop_30_1_stop, tf.float32)

    stable_win_prop_60_1 = tf.cast(stable_cv_prop * stable_time60_prop, tf.float32)
    stable_win_prop_60_0 = tf.cast(1 - stable_win_prop_60_1, tf.float32)

    stable_win_prop_60_1_stop = tf.cast(stable_cv_prop * stable_time60_prop_stop, tf.float32)
    stable_win_prop_60_0_stop = tf.cast(1 - stable_win_prop_60_1_stop, tf.float32)

    #loss_cv = -tf.reduce_mean(tf.math.log(stable_cv_prop) * label01 + tf.math.log(stable_cv_prop) * label11 + tf.math.log(1-stable_cv_prop)* (1-label10))
    loss_win_15 = -tf.reduce_mean(tf.math.log(stable_win_prop_15_1_stop) * label_01_15 + tf.math.log(stable_win_prop_15_0_stop) * (1-label_01_15))
    loss_win_30 = -tf.reduce_mean(tf.math.log(stable_win_prop_30_1_stop) * label_01_30_sum + tf.math.log(stable_win_prop_30_0_stop) * (1-label_01_30_sum) )
    loss_win_60 = -tf.reduce_mean(tf.math.log(stable_win_prop_60_1_stop) * label_01_60_sum + tf.math.log(stable_win_prop_60_0_stop) * (1-label_01_60_sum) )

    #loss_win_30 = -tf.reduce_mean(tf.math.log(stable_win_prop_30_1_stop) * label_01_30_sum * label_01_30_mask + tf.math.log(stable_win_prop_30_0_stop) * (1-label_01_30_sum) * label_01_30_mask)
    #loss_win_60 = -tf.reduce_mean(tf.math.log(stable_win_prop_60_1_stop) * label_01_60_sum * label_01_60_mask + tf.math.log(stable_win_prop_60_0_stop) * (1-label_01_60_sum) * label_01_60_mask)

    label_15_0 = tf.minimum(label_11_15+label_11_30+label_11_60+label10, 1)
    label_30_0 = tf.minimum(label_11_30+label_11_60+label10, 1)
    label_60_0 = tf.minimum(label_11_60+label10, 1)

    print 'label_15_0'
    print label_15_0.shape
    print label_15_0
    print 'label_15_0 3'
    print tf.reduce_mean(tf.concat([label_15_0, label_30_0, label_60_0], axis=1), axis=0)
    print tf.concat([label_15_0, label_30_0, label_60_0], axis=1)
    print 'prop3'
    print tf.reduce_mean(tf.concat([stable_win_prop_15_1, stable_win_prop_30_1, stable_win_prop_60_1], axis=1), axis=0)
    print tf.concat([stable_win_prop_15_1, stable_win_prop_30_1, stable_win_prop_60_1], axis=1)
    #print label_30_0
    #print label_60_0.shape
    #print label_60_0

    loss_win_15_spm = -tf.reduce_mean(tf.math.log(stable_win_prop_15_1) * label_01_15 + tf.math.log(stable_win_prop_15_0) * label_15_0 )
    loss_win_30_spm = -tf.reduce_mean(tf.math.log(stable_win_prop_30_1) * label_01_30_sum + tf.math.log(stable_win_prop_30_0) * label_30_0 )
    loss_win_60_spm = -tf.reduce_mean(tf.math.log(stable_win_prop_60_1) * label_01_60_sum + tf.math.log(stable_win_prop_60_0) * label_60_0 )
    #loss_win_30_spm = -tf.reduce_mean(tf.math.log(stable_win_prop_30_1) * label_01_30_sum * label_01_30_mask + tf.math.log(stable_win_prop_30_0) * label_30_0 * label_01_30_mask )
    #loss_win_60_spm = -tf.reduce_mean(tf.math.log(stable_win_prop_60_1) * label_01_60_sum * label_01_60_mask + tf.math.log(stable_win_prop_60_0) * label_60_0 * label_01_60_mask)

    #'''
    loss_win_now = -tf.reduce_mean(tf.math.log(stable_win_prop_15_1_stop) * label01 + tf.math.log(stable_win_prop_15_0_stop) * label00)
    loss_win_spm = -tf.reduce_mean(tf.math.log(stable_win_prop_15_1) * label01 + tf.math.log(stable_win_prop_15_0) * label10 + tf.math.log(stable_win_prop_15_0) * label11)
    #loss_cv_spm = -tf.reduce_mean(tf.math.log(stable_cv_prop) * label_01 + tf.math.log(stable_cv_prop) * label_11 + tf.math.log(1-stable_cv_prop)* label_10)
    #label01 = tf.minimum(label_01_15+label_01_30+label_01_60,1) 
    loss_cv_spm = -tf.reduce_mean(tf.math.log(stable_cv_prop) * label01 + tf.math.log(stable_cv_prop) * label11 + tf.math.log(1-stable_cv_prop)* (label10))

    loss = None
    if params["subloss"] == 1:
        print 'params["subloss"] 1'
        loss = 0.1*loss_win_15+0.05*loss_win_30+0.05*loss_win_60+0.1*loss_win_15_spm+0.05*loss_win_30_spm+0.05*loss_win_60_spm+0.6*loss_cv_spm #loss1
    #loss1 wrong auc_ma 0.840834810479, prauc_ma 0.640501172879, llloss_ma 0.391797181071
    #loss1 auc_ma 0.840922596412, prauc_ma 0.641111549595
    #loss1 newlabel auc_ma 0.840617351765, prauc_ma 0.640163426401, llloss_ma 0.3927667499
    #loss1 newdata auc_ma 0.840915778552, prauc_ma 0.641313958599, llloss_ma 0.392388541701

    if params["subloss"] == 2:
        print 'params["subloss"] 2'
        loss = 0.1*loss_win_15+0.1*loss_win_30+0.1*loss_win_60+0.04*loss_win_15_spm+0.03*loss_win_30_spm+0.03*loss_win_60_spm+0.6*loss_cv_spm #loss2
    #loss2 auc_ma 0.840743973481, prauc_ma 0.640195143789, llloss_ma 0.391840747798
    #loss2 real auc_ma 0.840823839357, prauc_ma 0.640766918437, llloss_ma 0.39245806335

    if params["subloss"] == 3:
        print 'params["subloss"] 3'
        loss = 0.09*loss_win_15+0.08*loss_win_30+0.08*loss_win_60+0.05*loss_win_15_spm+0.05*loss_win_30_spm+0.05*loss_win_60_spm+0.6*loss_cv_spm #loss3
    #loss3 auc_ma 0.840719450272, prauc_ma 0.640165336643, llloss_ma 0.392071954103
    #auc_ma 0.840865278839, prauc_ma 0.640899047731, llloss_ma 0.392095150118

    if params["subloss"] == 4:
        print 'params["subloss"] 4'
        loss = 0.15*loss_win_15+0.15*loss_win_60+0.05*loss_win_15_spm++0.05*loss_win_60_spm+0.6*loss_cv_spm #loss4
    #loss4 auc_ma 0.840833126014, prauc_ma 0.64074860748, llloss_ma 0.392482009323
    #auc_ma 0.840891558905, prauc_ma 0.640869974896, llloss_ma 0.392155661497

    if params["subloss"] == 5:
        print 'params["subloss"] 5'
        loss = 0.1*loss_win_15+0.1*loss_win_30+0.1*loss_win_60+0.1*loss_win_15_spm+0.1*loss_win_30_spm+0.1*loss_win_60_spm+0.4*loss_cv_spm #loss5
    #auc_ma 0.839641327406, prauc_ma 0.637054936063, llloss_ma 0.395906868003
    #auc_ma 0.840171224512, prauc_ma 0.638998457472, llloss_ma 0.393276991755
    
    #loss =  +loss_win_spm + loss_cv_spm
    #loss = loss_win_15 + loss_win_spm + loss_cv_spm
    #loss = 0.25 * loss_win_now + 0.15* loss_win_spm + 0.6 * loss_cv_spm
    #loss = 0.25 * loss_win_15 + 0.15* loss_win_15_spm + 0.6 * loss_cv_spm
    #loss = 0.25 * loss_win_15 + 0.15* loss_win_spm + 0.6 * loss_cv_spm
    if params["mul"] == 1:
        print 'params["mul"] 1'
        loss*=1
    if params["mul"] == 3:
        print 'params["mul"] 3'
        loss*=3
    if params["mul"] == 5:
        print 'params["mul"] 5'
        loss*=5
    #5loss4 auc_ma 0.840858786098, prauc_ma 0.640795619302, llloss_ma 0.392460761926
    #3loss2 auc_ma 0.840817501592, prauc_ma 0.64075829391, llloss_ma 0.39246253496

    print 'loss scale'
    print loss_win_15,loss_win_30,loss_win_60,loss_win_15_spm,loss_win_30_spm,loss_win_60_spm,loss_cv_spm
    return {"loss": loss}


def exp_delay_loss(targets, outputs, params=None):
    z = tf.reshape(tf.cast(targets["label"][:, 0], tf.float32), (-1, 1))
    x = outputs["logits"]
    lamb = tf.math.softplus(outputs["log_lamb"])
    log_lamb = tf.math.log(lamb)
    d = tf.reshape(tf.cast(targets["label"][:, 1], tf.float32), (-1, 1))
    e = d
    p = tf.nn.sigmoid(x)
    pos_loss = -(-stable_log1pex(x) + log_lamb - lamb*d)
    neg_loss = -tf.math.log(1 - p + p*tf.math.exp(-lamb*e))
    return {"loss": tf.reduce_mean(pos_loss*z + neg_loss*(1-z))}


def delay_tn_dp_loss(targets, outputs, params=None):
    tn = tf.cast(outputs["tn_logits"], tf.float32)
    dp = tf.cast(outputs["dp_logits"], tf.float32)
    z = tf.cast(targets["label"], tf.float32)
    #print 'z.shape'
    #print z.shape
    tn_label = tf.reshape(z[:, 0], (-1, 1))
    dp_label = tf.reshape(z[:, 1], (-1, 1))
    pos_label = tf.reshape(z[:, 2], (-1, 1))
    tn_mask = (1-pos_label)+dp_label
    tn_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=tn_label, logits=tn)*tn_mask)\
        / tf.reduce_sum(tn_mask)
    dp_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(labels=dp_label, logits=dp))
    loss = tn_loss + dp_loss
    return {
        "loss": loss,
        "tn_loss": tn_loss,
        "dp_loss": dp_loss
    }


def fsiw_loss(targets, outputs, params=None):
    x = outputs["logits"]
    logits0 = tf.stop_gradient(tf.cast(outputs["logits0"], tf.float32))
    logits1 = tf.stop_gradient(tf.cast(outputs["logits1"], tf.float32))
    prob0 = tf.sigmoid(logits0)
    prob1 = tf.sigmoid(logits1)
    z = tf.reshape(tf.cast(targets["label"], tf.float32), (-1, 1))

    pos_loss = stable_log1pex(x)
    neg_loss = x + stable_log1pex(x)

    pos_weight = 1/(prob1+1e-8)
    neg_weight = prob0

    clf_loss = tf.reduce_mean(
        pos_loss*pos_weight*z + neg_loss*neg_weight*(1-z))
    loss = clf_loss
    return {
        "loss": loss,
    }


def delay_tn_importance_weight_loss(targets, outputs, params=None):
    x = outputs["logits"]
    tn_logits = outputs["tn_logits"]
    dp_logits = outputs["dp_logits"]
    z = targets["label"]
    z = tf.reshape(tf.cast(z, tf.float32), (-1, 1))
    prob = tf.stop_gradient(tf.math.sigmoid(x))
    dist_prob = tf.math.sigmoid(tn_logits)
    dp_prob = tf.math.sigmoid(dp_logits)

    #pos_loss = stable_log1pex(x)
    #neg_loss = x + stable_log1pex(x)
    pos_loss = tf.math.log(tf.math.sigmoid(x))
    neg_loss = tf.math.log(1-tf.math.sigmoid(x))

    pos_weight = 1+dp_prob
    neg_weight = (1+dp_prob)*dist_prob
    neg_weight = tf.stop_gradient(neg_weight)
    pos_weight = tf.stop_gradient(pos_weight)

    clf_loss = -tf.reduce_mean(
        pos_loss*pos_weight*z + neg_loss*neg_weight*(1-z))
    loss = clf_loss
    return {"loss": loss,
            "clf_loss": clf_loss}

def delay_tn_importance_weight_loss10(targets, outputs, params=None):
    x = outputs["logits"]
    tn_logits = outputs["tn_logits"]
    dp_logits = outputs["dp_logits"]
    z = targets["label"]
    z = tf.reshape(tf.cast(z, tf.float32), (-1, 1))
    prob = tf.stop_gradient(tf.math.sigmoid(x))
    dist_prob = tf.math.sigmoid(tn_logits)
    dp_prob = tf.math.sigmoid(dp_logits)

    #pos_loss = stable_log1pex(x)
    #neg_loss = x + stable_log1pex(x)

    pos_loss = tf.math.log(tf.math.sigmoid(x))
    neg_loss = tf.math.log(1-tf.math.sigmoid(x))

    py0 = tf.stop_gradient(1-tf.math.sigmoid(x))
    py1 = 1-py0

    pos_weight = py1 + 2*py0 + dp_prob
    neg_weight = (py1 + 2*py0 + dp_prob)*py0/(2*py0+dp_prob)
    #pos_weight = 1+dp_prob+py0
    #neg_weight = (1+dp_prob+py0)*dist_prob/2
    #pos_weight = 1+dp_prob
    #neg_weight = (1+dp_prob)*dist_prob/2
    neg_weight = tf.stop_gradient(neg_weight)
    pos_weight = tf.stop_gradient(pos_weight)

    clf_loss = -tf.reduce_mean(
        pos_loss*pos_weight*z + neg_loss*neg_weight*(1-z))
    loss = clf_loss
    return {"loss": loss,
            "clf_loss": clf_loss}


def delay_tn_importance_weight_loss_full(targets, outputs, params=None):
    x = outputs["logits"]
    tn_logits = outputs["tn_logits"]
    dp_logits = outputs["dp_logits"]
    #z = targets["label"]

    z = tf.cast(targets["label"], tf.float32)
    labels = z
    print 'z.shape'
    print z.shape
    label_01 = tf.reshape(z[:, 0], (-1, 1))
    label_00 = tf.reshape(z[:, 1], (-1, 1))
    label_11 = tf.reshape(z[:, 2], (-1, 1))
    label_10 = tf.reshape(z[:, 3], (-1, 1))

    #z = tf.reshape(tf.cast(z, tf.float32), (-1, 1))
    prob = tf.stop_gradient(tf.math.sigmoid(x))
    dist_prob = tf.math.sigmoid(tn_logits)
    dp_prob = tf.math.sigmoid(dp_logits)

    #pos_loss = stable_log1pex(x)
    #neg_loss = x + stable_log1pex(x)

    pos_loss = tf.math.log(tf.math.sigmoid(x))
    neg_loss = tf.math.log(1-tf.math.sigmoid(x))

    py0 = tf.stop_gradient(1-tf.math.sigmoid(x))
    py1 = 1.0-py0 #tf.stop_gradient(tf.math.sigmoid(x))

    #pos_weight = 1+dp_prob+py0
    #neg_weight = (1+dp_prob+py0)*dist_prob/2
    #version1
    #pos_weight = tf.maximum(tf.minimum(2*py1/(2*py1-dp_prob),3), 1)
    #neg_weight = 2*py0/(2*py0+dp_prob)
    #version2
    #pos_weight = 2*py1/(2*py1+dp_prob)
    #neg_weight = 2*dist_prob
    #version3
    pos_weight = 2*(1-dist_prob)
    neg_weight = 2*dist_prob

    neg_weight = tf.stop_gradient(neg_weight)
    pos_weight = tf.stop_gradient(pos_weight)


    #clf_loss = -tf.reduce_sum(
    #    pos_loss*pos_weight*label_01 + pos_loss*pos_weight*label_11 + neg_loss*neg_weight*label_00+neg_loss*neg_weight*label_10)/tf.reduce_sum(label_01+label_11+label_00+label_10)
    clf_loss = -tf.reduce_sum(
        2*pos_loss*pos_weight*label_01 + pos_loss*pos_weight*label_11 + neg_loss*neg_weight*label_00+neg_loss*neg_weight*label_10)/tf.reduce_sum(2*label_01+label_11+label_00+label_10)
    #clf_loss = -tf.reduce_mean(
    #    pos_loss*pos_weight*label_01 + pos_loss*pos_weight*label_11 + neg_loss*neg_weight*label_00 + neg_loss*neg_weight*label_10)

    loss = clf_loss
    return {"loss": loss,
            "clf_loss": clf_loss}
'''
def delay_tn_importance_weight_loss_full(targets, outputs, params=None):
    x = outputs["logits"]
    tn_logits = outputs["tn_logits"]
    dp_logits = outputs["dp_logits"]
    z = targets["label"]
    z = tf.reshape(tf.cast(z, tf.float32), (-1, 1))
    prob = tf.stop_gradient(tf.math.sigmoid(x))
    dist_prob = tf.math.sigmoid(tn_logits)
    dp_prob = tf.math.sigmoid(dp_logits)

    #pos_loss = stable_log1pex(x)
    #neg_loss = x + stable_log1pex(x)

    pos_loss = tf.math.log(tf.math.sigmoid(x))
    neg_loss = tf.math.log(1-tf.math.sigmoid(x))

    py0 = tf.stop_gradient(1-tf.math.sigmoid(x))
    py1 = tf.stop_gradient(tf.math.sigmoid(x))

    #pos_weight = 1+dp_prob+py0
    #neg_weight = (1+dp_prob+py0)*dist_prob/2
    #version1
    #pos_weight = 2*py1/(py1+1-dp_prob)
    #neg_weight = 2*py0/(2*py0+dp_prob)
    #version2
    #pos_weight = 2*py1/(py1+1-dp_prob)
    #neg_weight = 2*dist_prob
    #version3
    pos_weight = 2*(1-dist_prob)
    neg_weight = 2*dist_prob


    neg_weight = tf.stop_gradient(neg_weight)
    pos_weight = tf.stop_gradient(pos_weight)

    clf_loss = -tf.reduce_sum(
        1.35*pos_loss*pos_weight*z + neg_loss*neg_weight*(1-z))/tf.reduce_sum(1.35*z+(1-z))
    loss = clf_loss
    return {"loss": loss,
            "clf_loss": clf_loss}
'''

def delay_tn_importance_weight_loss_normal_test(targets, outputs, params=None):
    #x = outputs["logits"]
    tn_logits = outputs["tn_logits"]
    dp_logits = outputs["dp_logits"]

    x = tf.cast(tf.reshape(outputs["logits"][:, 0], (-1, 1)), tf.float32)

    z = targets["label"]
    z = tf.reshape(tf.cast(z, tf.float32), (-1, 1))
    prob = tf.stop_gradient(tf.math.sigmoid(x))
    #prob = tf.stop_gradient(tf.math.sigmoid(cv_logit))
    dist_prob = tf.math.sigmoid(tn_logits)
    dp_prob = tf.math.sigmoid(dp_logits)

    #pos_loss = stable_log1pex(x)
    #neg_loss = x + stable_log1pex(x)
    pos_loss = tf.math.log(tf.math.sigmoid(x))
    neg_loss = tf.math.log(1-tf.math.sigmoid(x))

    pos_weight = 1+dp_prob
    neg_weight = (1+dp_prob)*dist_prob
    neg_weight = tf.stop_gradient(neg_weight)
    pos_weight = tf.stop_gradient(pos_weight)

    clf_loss = -tf.reduce_mean(
        pos_loss*pos_weight*z + neg_loss*neg_weight*(1-z))
    loss = clf_loss
    return {"loss": loss,
            "clf_loss": clf_loss}


def delay_tn_importance_weight_loss_normal(targets, outputs, params=None):
    x = outputs["logits"]
    tn_logits = outputs["tn_logits"]
    dp_logits = outputs["dp_logits"]

    #cv_logit = tf.cast(tf.reshape(outputs["logits"][:, 0], (-1, 1)), tf.float32)

    z = targets["label"]
    z = tf.reshape(tf.cast(z, tf.float32), (-1, 1))
    prob = tf.stop_gradient(tf.math.sigmoid(x))
    #prob = tf.stop_gradient(tf.math.sigmoid(cv_logit))
    dist_prob = tf.math.sigmoid(tn_logits)
    dp_prob = tf.math.sigmoid(dp_logits)

    #pos_loss = stable_log1pex(x)
    #neg_loss = x + stable_log1pex(x)
    pos_loss = tf.math.log(tf.math.sigmoid(x))
    neg_loss = tf.math.log(1-tf.math.sigmoid(x))

    pos_weight = 1+dp_prob
    neg_weight = (1+dp_prob)*dist_prob
    neg_weight = tf.stop_gradient(neg_weight)
    pos_weight = tf.stop_gradient(pos_weight)

    clf_loss = -tf.reduce_mean(
        pos_loss*pos_weight*z + neg_loss*neg_weight*(1-z))
    loss = clf_loss
    return {"loss": loss,
            "clf_loss": clf_loss}

def delay_tn_importance_weight_loss_win_weight(targets, outputs, params=None):
    print 'targets:' ,targets["label"].shape
    x = outputs["logits"]
    tn_logits = outputs["tn_logits"]
    dp_logits = outputs["dp_logits"]

    cv_logit = tf.cast(tf.reshape(outputs["logits"][:, 0], (-1, 1)), tf.float32)

    z = targets["label"][:,0]
    z = tf.reshape(tf.cast(z, tf.float32), (-1, 1))

    time_weight = 1.5 - targets["label"][:,1]

    #prob = tf.stop_gradient(tf.math.sigmoid(x))
    prob = tf.stop_gradient(tf.math.sigmoid(cv_logit))
    dist_prob = tf.math.sigmoid(tn_logits)
    dp_prob = tf.math.sigmoid(dp_logits)

    pos_loss = tf.math.log(tf.math.sigmoid(x))
    neg_loss = tf.math.log(1-tf.math.sigmoid(x))

    pos_weight = 1+dp_prob*time_weight
    neg_weight = (1+dp_prob)*dist_prob
    neg_weight = tf.stop_gradient(neg_weight)
    pos_weight = tf.stop_gradient(pos_weight)

    clf_loss = -tf.reduce_mean(
        pos_loss*pos_weight*z + neg_loss*neg_weight*(1-z))
    loss = clf_loss
    return {"loss": loss,
            "clf_loss": clf_loss}


def delay_tn_importance_weight_loss_win(targets, outputs, params=None):
    x = outputs["logits"]
    tn_logits = outputs["tn_logits"]
    dp_logits = outputs["dp_logits"]

    cv_logit = tf.cast(tf.reshape(outputs["logits"][:, 0], (-1, 1)), tf.float32)

    z = targets["label"]
    z = tf.reshape(tf.cast(z, tf.float32), (-1, 1))
    #prob = tf.stop_gradient(tf.math.sigmoid(x))
    prob = tf.stop_gradient(tf.math.sigmoid(cv_logit))
    dist_prob = tf.math.sigmoid(tn_logits)
    dp_prob = tf.math.sigmoid(dp_logits)

    #pos_loss = stable_log1pex(x)
    #neg_loss = x + stable_log1pex(x)
    pos_loss = tf.math.log(tf.math.sigmoid(x))
    neg_loss = tf.math.log(1-tf.math.sigmoid(x))

    pos_weight = 1+dp_prob
    neg_weight = (1+dp_prob)*dist_prob
    neg_weight = tf.stop_gradient(neg_weight)
    pos_weight = tf.stop_gradient(pos_weight)

def esdfm_loss_wines(targets, outputs, params=None):
    x = outputs["logits"]
    tn_logits = outputs["tn_logits"]
    dp_logits = outputs["dp_logits"]

    cv_logit = tf.cast(tf.reshape(outputs["logits"][:, 0], (-1, 1)), tf.float32)
    time_logit = tf.cast(tf.reshape(outputs["logits"][:, 1], (-1, 1)), tf.float32)

    ####

    stable_cv_prop = stable_sigmoid_pos(tf.cast(outputs["logits"][:, 0], tf.float32))
    stable_time_prop = stable_sigmoid_pos(tf.cast(outputs["logits"][:, 1], tf.float32))

    stable_cv_prop_stop = tf.stop_gradient(stable_cv_prop)
    stable_time_prop_stop = tf.stop_gradient(stable_time_prop)

    stable_win_prop_1 = stable_cv_prop * stable_time_prop
    stable_win_prop_0 = 1 - stable_win_prop_1   # 1- stable_cv_prop + stable_cv_prop * (1-stable_time_prop)

    stable_win_prop_1_stop = stable_cv_prop * stable_time_prop_stop
    stable_win_prop_0_stop = 1 - stable_win_prop_1_stop

    dp = stable_cv_prop_stop* (1-stable_time_prop_stop)
    rn = (1-stable_cv_prop_stop) /stable_win_prop_0_stop

    #pos_loss = tf.math.log(tf.math.sigmoid(x))
    #neg_loss = tf.math.log(1-tf.math.sigmoid(x))

    py0 = tf.stop_gradient(1-stable_cv_prop_stop)
    #py0 = (1-stable_cv_prop)

    pos_weight = 1+dp
    neg_weight = (1+dp)*rn
    neg_weight = tf.stop_gradient(neg_weight)
    pos_weight = tf.stop_gradient(pos_weight)

    #pos_weight = 1+dp+py0
    #neg_weight = (1+dp+py0)*rn/2
    #neg_weight = tf.stop_gradient(neg_weight)
    #pos_weight = tf.stop_gradient(pos_weight)


    ###
    z = targets["label"]
    z = tf.reshape(tf.cast(z, tf.float32), (-1, 1))
    #prob = tf.stop_gradient(tf.math.sigmoid(x))
    #############paper################
    prob = tf.stop_gradient(tf.math.sigmoid(cv_logit))
    dist_prob = tf.math.sigmoid(tn_logits)
    dp_prob = tf.math.sigmoid(dp_logits)
    #############half##################!!!!!!!!!!!!!!!!!!!!!!!
    #dist_prob = (1-stable_cv_prop)/(1-stable_cv_prop+dp_prob)
    #dist_prob = py0/(py0+dp_prob)
    dist_prob = py0/(stable_win_prop_0)
    #pos_loss = stable_log1pex(x)
    #neg_loss = x + stable_log1pex(x)
    x = cv_logit
    pos_loss = tf.math.log(tf.math.sigmoid(x))
    neg_loss = tf.math.log(1-tf.math.sigmoid(x))
    #paper
    pos_weight = 1+dp_prob
    neg_weight = (1+dp_prob)*dist_prob
    neg_weight = tf.stop_gradient(neg_weight)
    pos_weight = tf.stop_gradient(pos_weight)

    clf_loss = -tf.reduce_mean(
        pos_loss*pos_weight*z + neg_loss*neg_weight*(1-z))
    loss = clf_loss
    return {"loss": loss,
            "clf_loss": clf_loss}

#'''
def delay_likelihood_loss(targets, outputs, params=None):
    logits = tf.cast(outputs["logits"], tf.float32)
    z = tf.cast(targets["label"], tf.float32)
    labels = z
    #print 'z.shape'
    #print z.shape
    label_11 = tf.reshape(z[:, 0], (-1, 1))
    label_00 = tf.reshape(z[:, 1], (-1, 1))
    label_01 = tf.reshape(z[:, 2], (-1, 1))

    cv_prop = tf.sigmoid(tf.cast(outputs["cv_logits"], tf.float32))
    time_prop = tf.sigmoid(tf.cast(outputs["time_logits"], tf.float32))

    stable_cv_prop = stable_sigmoid_pos(tf.cast(outputs["cv_logits"], tf.float32))
    stable_time_prop = stable_sigmoid_pos(tf.cast(outputs["time_logits"], tf.float32))

    win_prop_1 = cv_prop * time_prop
    win_prop_0 = (1-cv_prop) + cv_prop*(1-time_prop)
    spm_prop_1 = cv_prop * (1-time_prop)

    stable_win_prop_1 = stable_cv_prop * stable_time_prop
    stable_win_prop_0 = 1 - stable_win_prop_1
    stable_spm_prop_1 = stable_cv_prop * (1-stable_time_prop)

    #loss_win = -tf.reduce_mean(tf.math.log(win_prop_1) * label_01 + tf.math.log(win_prop_0) * label_00)
    #loss_spm = -tf.reduce_mean(tf.math.log(spm_prop_1) * label_11)
    loss_win = -tf.reduce_mean(tf.math.log(stable_win_prop_1) * label_01 + tf.math.log(stable_win_prop_0) * label_00)
    loss_spm = 0 #-tf.reduce_mean(tf.math.log(stable_spm_prop_1) * label_11)

    loss = loss_win + loss_spm
    return {
        "loss": loss,
        "tn_loss": loss_win,
        "dp_loss": loss_spm
    }

def delay_win_time_loss(targets, outputs, params=None):
    logits = tf.cast(outputs["logits"], tf.float32)
    z = tf.cast(targets["label"], tf.float32)
    labels = z
    print 'z.shape'
    print z.shape
    label_01 = tf.reshape(z[:, 0], (-1, 1))
    label_00 = tf.reshape(z[:, 1], (-1, 1))
    label_11 = tf.reshape(z[:, 2], (-1, 1))
    label_10 = tf.reshape(z[:, 3], (-1, 1))
    
    cv_prop = tf.sigmoid(tf.cast(outputs["cv_logits"], tf.float32))
    time_prop = tf.sigmoid(tf.cast(outputs["time_logits"], tf.float32))

    cv_prop_stop = tf.stop_gradient(cv_prop)
    time_prop_stop = tf.stop_gradient(time_prop)

    stable_cv_prop = stable_sigmoid_pos(tf.cast(outputs["cv_logits"], tf.float32))
    stable_time_prop = stable_sigmoid_pos(tf.cast(outputs["time_logits"], tf.float32))

    stable_cv_prop_stop = tf.stop_gradient(stable_cv_prop)
    stable_time_prop_stop = tf.stop_gradient(stable_time_prop)

    #win_prop_1 = cv_prop * time_prop
    #win_prop_0 = (1-cv_prop) + cv_prop*(1-time_prop)

    stable_win_prop_1 = stable_cv_prop * stable_time_prop
    stable_win_prop_0 = 1 - stable_win_prop_1

    #win_prop_1_stop = cv_prop * (1-time_prop_stop)
    #win_prop_0_stop =  (1-cv_prop) + cv_prop*(1-time_prop_stop)

    stable_win_prop_1_stop = stable_cv_prop * stable_time_prop_stop
    stable_win_prop_0_stop = 1 - stable_win_prop_1_stop

    #stable_win_prop_1 = stable_win_prop_1_stop
    #stable_win_prop_0 = stable_win_prop_0_stop 			#auc_ma 0.83788809202, prauc_ma 0.635210277143, llloss_ma 0.396383280415

    #loss_win_now = -tf.reduce_mean(tf.math.log(win_prop_1_stop) * label_01 + tf.math.log(win_prop_0_stop) * label_00)
    #auc_ma 0.838268897391, prauc_ma 0.636577855824, llloss_ma 0.395493760895
    #classic
    #loss_win_now = -tf.reduce_mean(tf.math.log(stable_win_prop_1_stop) * label_01 + tf.math.log(stable_win_prop_0_stop) * label_00) 
    loss_win_now = -tf.reduce_mean(tf.math.log(stable_win_prop_1) * label_01 + tf.math.log(stable_win_prop_0) * label_00)
    #auc_ma 0.838108919221, prauc_ma 0.634304156199, llloss_ma 0.396333312333#
    #loss_win_now = -tf.reduce_mean(tf.math.log(stable_win_prop_1) * label_01 + tf.math.log(stable_win_prop_0) * label_00)

    #loss_spm = -tf.reduce_mean(tf.math.log(spm_prop_1) * label_11)

    #win_prop_1 = cv_prop * time_prop
    #win_prop_0 = (1-cv_prop) + cv_prop*(1-time_prop)

    #loss_win_spm = -tf.reduce_mean(tf.math.log(win_prop_1) * label_01 + tf.math.log(win_prop_0) * label_10 + tf.math.log(win_prop_0) * label_11)
    #loss_cv_spm = -tf.reduce_mean(tf.math.log(cv_prop) * label_01 + tf.math.log(cv_prop) * label_11 + tf.math.log(1-cv_prop)* label_10)
    
    #classic
    loss_win_spm = -tf.reduce_mean(tf.math.log(stable_win_prop_1) * label_01 + tf.math.log(stable_win_prop_0) * label_10 + tf.math.log(stable_win_prop_0) * label_11)
    loss_cv_spm = -tf.reduce_mean(tf.math.log(stable_cv_prop) * label_01 + tf.math.log(stable_cv_prop) * label_11 + tf.math.log(1-stable_cv_prop)* label_10)

    #esdfm								  #auc_ma 0.839974196585, prauc_ma 0.638705895854, llloss_ma 0.392870014929
    #esdfm lr0002							  auc_ma 0.840567941748, prauc_ma 0.639743851449, llloss_ma 0.391760005826
    #sepwintime   							#auc_ma 0.838776339428, prauc_ma 0.637686467899, llloss_ma 0.394870592343
    #classic	
    loss = 0.25 * loss_win_now + 0.15* loss_win_spm + 0.6 * loss_cv_spm   #auc_ma 0.838354866626, prauc_ma 0.636769552212, llloss_ma 0.395351204868
    #loss = 0.3 * loss_win_now + 0.7 * loss_cv_spm
    #simple loss: auc_ma 0.841278818739, prauc_ma 0.641897278575, llloss_ma 0.391163869451
    #lr 0.002 								  #auc_ma 0.841320768347, prauc_ma 0.642100768647, llloss_ma 0.391044942986
    loss *= 3								  #auc_ma 0.838617494567, prauc_ma 0.637304822108, llloss_ma 0.39513652706
    #lr 0.002								  auc_ma 0.8413391198, prauc_ma 0.642140924079, llloss_ma 0.391030179747
    #loss *= 6								  #auc_ma 0.838670803859, prauc_ma 0.637196640774, llloss_ma 0.395095014098
    #loss *= 10								  #auc_ma 0.838729377884, prauc_ma 0.637390114327, llloss_ma 0.39501309332
    #loss *= 20								  #auc_ma 0.838792319879, prauc_ma 0.637416541786, llloss_ma 0.394930307985
    #lr0002								  #auc_ma 0.841298669272, prauc_ma 0.642070518344, llloss_ma 0.391058392547
    #loss *= 40								  #auc_ma 0.838741308658, prauc_ma 0.637349426768, llloss_ma 0.395005292815
    #loss = 0.5 * loss_win_now + 0.15* loss_win_spm + 0.35 * loss_cv_spm  #auc_ma 0.837489383951, prauc_ma 0.636017430217, llloss_ma 0.395796233339
    #loss = 0.15 * loss_win_now + 0.1* loss_win_spm + 0.75 * loss_cv_spm  #auc_ma 0.838158393511, prauc_ma 0.635659864137, llloss_ma 0.395944307324
    #loss = 0.25* loss_win_spm + 0.75 * loss_cv_spm			  #auc_ma 0.837760871004, prauc_ma 0.632442897763, llloss_ma 0.397135111464

    #loss = 0.35 * loss_win_now + 0.25* loss_win_spm + 0.4 * loss_cv_spm
    #loss = loss_win_now + loss_win_spm + loss_cv_spm  			  #auc_ma 0.838081507431, prauc_ma 0.636920889289, llloss_ma 0.395256672262

    #loss = loss_win_now
    #loss = loss_cv_spm
    return {
        "loss": loss,
        "loss_win_now": loss_win_now,
        "loss_win_spm": loss_win_spm,
        "loss_cv_spm": loss_cv_spm
    }


def delay_win_time_iwloss(targets, outputs, params=None):
    logits = tf.cast(outputs["logits"], tf.float32)
    z = tf.cast(targets["label"], tf.float32)
    labels = z
    print 'z.shape'
    print z.shape
    label_01 = tf.reshape(z[:, 0], (-1, 1))
    label_00 = tf.reshape(z[:, 1], (-1, 1))
    label_11 = tf.reshape(z[:, 2], (-1, 1))
    label_10 = tf.reshape(z[:, 3], (-1, 1))

    cv_prop = tf.sigmoid(tf.cast(outputs["cv_logits"], tf.float32))
    time_prop = tf.sigmoid(tf.cast(outputs["time_logits"], tf.float32))

    cv_prop_stop = tf.stop_gradient(cv_prop)
    time_prop_stop = tf.stop_gradient(time_prop)

    stable_cv_prop = stable_sigmoid_pos(tf.cast(outputs["cv_logits"], tf.float32))
    stable_time_prop = stable_sigmoid_pos(tf.cast(outputs["time_logits"], tf.float32))

    stable_cv_prop_stop = tf.stop_gradient(stable_cv_prop)
    stable_time_prop_stop = tf.stop_gradient(stable_time_prop)

    stable_win_prop_1 = stable_cv_prop * stable_time_prop
    stable_win_prop_0 = 1 - stable_win_prop_1   # 1- stable_cv_prop + stable_cv_prop * (1-stable_time_prop)

    stable_win_prop_1_stop = stable_cv_prop * stable_time_prop_stop
    stable_win_prop_0_stop = 1 - stable_win_prop_1_stop

    dp = stable_cv_prop* (1-stable_time_prop)
    rn = (1-stable_cv_prop) /stable_win_prop_0

    #pos_loss = tf.math.log(tf.math.sigmoid(x))
    #neg_loss = tf.math.log(1-tf.math.sigmoid(x))

    py0 = tf.stop_gradient(1-stable_cv_prop_stop)

    pos_weight = 1+dp+py0
    neg_weight = (1+dp+py0)*rn/2
    neg_weight = tf.stop_gradient(neg_weight)
    pos_weight = tf.stop_gradient(pos_weight)

    #clf_loss = -tf.reduce_mean(
    #    pos_loss*pos_weight*z + neg_loss*neg_weight*(1-z))

    #pos_weight = 1 + dp
    #neg_weight = (1 + dp) * rn
    #neg_weight = tf.stop_gradient(neg_weight)
    #pos_weight = tf.stop_gradient(pos_weight)

    loss_win_now = -tf.reduce_mean(tf.math.log(stable_win_prop_1) * label_01 + tf.math.log(stable_win_prop_0) * label_00)
    #classic
    loss_win_spm = -tf.reduce_mean(tf.math.log(stable_win_prop_1) * label_01 + tf.math.log(stable_win_prop_0) * label_10 + tf.math.log(stable_win_prop_0) * label_11)
    loss_cv_spm = -tf.reduce_mean(pos_weight * tf.math.log(stable_cv_prop) * label_01 + pos_weight * tf.math.log(stable_cv_prop) * label_11 + neg_weight * tf.math.log(1-stable_cv_prop) * label_00)
    #loss_cv_spm = -tf.reduce_mean(tf.math.log(stable_cv_prop) * label_01 + tf.math.log(stable_cv_prop) * label_11 + tf.math.log(1-stable_cv_prop)* label_10)
    loss = loss_cv_spm

    #loss = 0.25 * loss_win_now + 0.75 * loss_cv_spm

    #loss = 0.25 * loss_win_now + 0.15* loss_win_spm + 0.6 * loss_cv_spm   #auc_ma 0.838354866626, prauc_ma 0.636769552212, llloss_ma 0.395351204868
    loss *= 3                                                             #auc_ma 0.838617494567, prauc_ma 0.637304822108, llloss_ma 0.39513652706
    return {
        "loss": loss,
        "loss_win_now": loss_win_now,
        "loss_win_spm": loss_win_spm,
        "loss_cv_spm": loss_cv_spm
    }


def delay_3class_loss(targets, outputs, params=None):
    #spm_logits = tf.cast(outputs["spm_logits"], tf.float32)
    #winneg_logits = tf.cast(outputs["winneg_logits"], tf.float32)
    #winpos_logits = tf.cast(outputs["winpos_logits"], tf.float32)
    #labels = np.concatenate([train_label_spm, train_label_winneg, train_label_winpos], axis=1)
    logits = tf.cast(outputs["logits"], tf.float32)
    z = tf.cast(targets["label"], tf.float32)
    labels = z
    #print 'z.shape'
    #print z.shape
    label_11 = tf.reshape(z[:, 0], (-1, 1))
    label_00 = tf.reshape(z[:, 1], (-1, 1))
    label_01 = tf.reshape(z[:, 2], (-1, 1))
    label_pos = tf.minimum(label_01+label_11,1)
    prop = tf.nn.softmax(logits) + 0.0000001
    prop_00 = tf.reshape(prop[:, 0], (-1, 1)) #tf.slice(prop, [0, 0],[-1, 1])
    prop_01 = tf.reshape(prop[:, 1], (-1, 1))
    prop_11 = tf.reshape(prop[:, 2], (-1, 1))

    prop_1 = prop_01+prop_11
    prop_0 = prop_00

    prop_new = tf.concat([prop_00,prop_01,prop_11], axis=1)

    #prop2 = tf.concat([prop_0,prop_1], axis=1)
    #prop_00+prop_01

    loss_00 = - tf.reduce_sum(tf.math.log(prop_00+prop_01) * label_00)
    loss_01 = - tf.reduce_sum(tf.math.log(prop_11) * label_01)
    loss_11 = - tf.reduce_sum(tf.math.log(prop_01) * label_11)

    #loss = loss_00 + loss_01 + loss_11
    loss_3class = - tf.reduce_sum(tf.math.log(prop_00+prop_01) * label_00 + tf.math.log(prop_11) * label_01 + tf.math.log(prop_01) * label_11, axis=1, keepdims=True)
    #loss_3class = - tf.reduce_sum(tf.math.log(prop_00) * label_00 + tf.math.log(prop_11) * label_pos, axis=1, keepdims=True)
    loss = tf.reduce_mean(loss_3class, axis=0)

    return {
        "loss": loss,
        "loss_00": loss_00,
        "loss_01": loss_01,
        "loss_11": loss_11
    }



def get_loss_fn(name):
    if name == "cross_entropy_loss":
        return cross_entropy_loss
    elif name == "fake_negative_weighted_loss":
        return fake_negative_weighted_loss
    elif name == "delayed_feedback_loss":
        return exp_delay_loss
    elif name == "tn_dp_pretraining_loss":
        return delay_tn_dp_loss
    elif name == "fsiw_loss":
        return fsiw_loss
    elif name == "esdfm_loss":
        return delay_tn_importance_weight_loss
    elif name == "esdfm_loss_win":
        return delay_tn_importance_weight_loss_win
    elif name == "esdfm_loss_win_weight":
        return delay_tn_importance_weight_loss_win_weight
    elif name == "esdfm_loss_normal":
        return delay_tn_importance_weight_loss_normal
    elif name == "esdfm_loss_full":
        return delay_tn_importance_weight_loss_full
    elif name == "esdfm_loss_normal_test":
	return delay_tn_importance_weight_loss_normal_test
    elif name == "esdfm_loss10":
        return delay_tn_importance_weight_loss10
    elif name == "delay_3class_loss":
        return delay_3class_loss
    elif name == "softmax_cross_entropy_loss":
        return softmax_cross_entropy_loss
    elif name == "delay_likelihood_loss":
        return delay_likelihood_loss
    elif name == "win_time_loss":
        return win_time_loss
    elif name == "win_select_loss":
        return win_select_loss
    elif name == "delay_win_select_loss":
        return delay_win_select_loss
    elif name == "delay_win_time_loss":
        return delay_win_time_loss
    elif name == "delay_win_time_iwloss":
        return delay_win_time_iwloss 
    elif name == "fnw_test_loss":
        return fnw_test_loss
    elif name == "fnw10_test_loss":
        return fnw10_test_loss
    elif name == "cross_entropy_loss_test":
	return cross_entropy_loss_test
    elif name == "win_time_loss_test":
        return win_time_loss_test
    elif name == "esdfm_loss_wines":
        return esdfm_loss_wines 
    elif name == "cross_entropy_loss_00":
        return cross_entropy_loss_00
    else:
        raise NotImplementedError("{} loss does not implemented".format(name))
