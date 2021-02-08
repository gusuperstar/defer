from models import get_model
from loss import get_loss_fn
from utils import get_optimizer
from metrics import cal_llloss_with_logits, cal_auc, cal_prauc, cal_llloss_with_prob, cal_softmax_cross_entropy_loss
from tqdm import tqdm
import numpy as np
import tensorflow as tf
from collections import defaultdict
from data import get_criteo_dataset
from utils import ScalarMovingAverage

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(
    physical_devices[0], enable=True
)
if len(physical_devices) > 1:
    tf.config.experimental.set_memory_growth(
        physical_devices[1], enable=True
    )


def optim_step(step, model, x, targets, optimizer, loss_fn, params):
    with tf.GradientTape() as g:
        outputs = model(x, training=True)
        reg_loss = tf.add_n(model.losses)
        loss_dict = loss_fn(targets, outputs, params)
        loss = loss_dict["loss"] + reg_loss

    if params["method"] == "ES-DFM" or params["method"] == "DFM":
        return
    labelloss = None
    if params["method"] == "3class":
        labelprint = np.mean(targets["label"],axis=0)
        pctr = np.mean(tf.nn.softmax(outputs["logits"]),axis=0)
        
        print "train_label:"
        print labelprint
        print "train_pctr:%f"
        print pctr
    elif params["method"] == "win_time":
        labelprint = np.mean(targets["label"],axis=0)
        pctr = np.mean(tf.nn.sigmoid(outputs["cv_logits"]),axis=0)
        ptime = np.mean(tf.nn.sigmoid(outputs["time_logits"]),axis=0)
        print "train_label:"
        print labelprint
        print "train_pctr:%f"
        print pctr
        print "train_ptime:%f"
        print ptime
        labelloss = np.reshape(targets["label"][:,0].numpy(),(-1,1))
        probloss = np.reshape(tf.nn.sigmoid(outputs["logits"][:,0]).numpy(),(-1,1))
        logitloss = np.reshape(outputs["logits"][:,0].numpy(),(-1,1))
    elif params["method"] == "win_time_test":
        labelloss = np.reshape(targets["label"][:,0].numpy(),(-1,1))
        probloss = np.reshape(tf.nn.sigmoid(outputs["logits"]).numpy(),(-1,1))
        logitloss = np.reshape(outputs["logits"].numpy(),(-1,1))
    else:
        labelloss = np.reshape(targets["label"].numpy(),(-1,1))
        probloss = np.reshape(tf.nn.sigmoid(outputs["logits"]).numpy(),(-1,1))
        logitloss = np.reshape(outputs["logits"].numpy(),(-1,1))

    llloss = cal_llloss_with_logits(labelloss, logitloss)
    print "step%d test_loss:%f" %(step,llloss)
    auc = cal_auc(labelloss, probloss)
    print "step%d test_auc:%f" %(step,auc)
    print "train_loss:%f" %loss_dict["loss"]
    print "train_reg_loss:%f" %reg_loss
    print "step%d train_all_loss:%f" %(step,loss)
    trainable_variables = model.trainable_variables
    gradients = g.gradient(loss, trainable_variables)
    optimizer.apply_gradients(zip(gradients, trainable_variables))


def test(model, test_data, params, before=False):
    all_logits = []
    all_probs = []
    all_labels = []
    for step, (batch_x, batch_y) in enumerate(tqdm(test_data), 1):
        logits = model(batch_x, training=False)["logits"]
        all_logits.append(logits.numpy())
        all_labels.append(batch_y.numpy())

        if False:#params["method"] == "3class":
            #prop = tf.nn.softmax(logits)
            #prop_00 = tf.reshape(prop[:, 0], (-1, 1)) #tf.slice(prop, [0, 0],[-1, 1])
            #prop_01 = tf.reshape(prop[:, 1], (-1, 1))
            #prop_11 = tf.reshape(prop[:, 2], (-1, 1))
            #prop_1 = prop_01+prop_11
            all_probs.append(logits)
        else:

            all_probs.append(tf.math.sigmoid(logits))
    llloss = 0
    if params["method"] == "3class":
        all_logits = np.reshape(np.concatenate(all_logits, axis=0), (-1, 3))
        all_labels = np.reshape(np.concatenate(all_labels, axis=0), (-1, 3))
        all_probs = np.reshape(np.concatenate(all_probs, axis=0), (-1, 3))
    elif params["method"] == "win_time" or params["method"] == "test": 
        all_logits = np.reshape(np.concatenate(all_logits, axis=0), (-1, 2))
        all_labels = np.reshape(np.concatenate(all_labels, axis=0), (-1, 2))
        all_probs = np.reshape(np.concatenate(all_probs, axis=0), (-1, 2))
    elif params["method"] == "delay_win_adapt" :
        #all_logits = np.reshape(np.concatenate(all_logits, axis=0), (-1, 2))
        all_probs = np.reshape(np.concatenate(all_probs, axis=0), (-1, 2))
    elif params["method"] == "win_adapt":#pre win_adapt
        all_probs = np.reshape(np.concatenate(all_probs, axis=0), (-1, 4))
        all_labels = np.reshape(np.concatenate(all_labels, axis=0), (-1,4 ))

    elif ( params["method"] == "delay_win_select") and not before:
        #all_logits = np.reshape(np.concatenate(all_logits, axis=0), (-1, 2))
        all_probs = np.reshape(np.concatenate(all_probs, axis=0), (-1, 4))
        all_labels = np.reshape(np.concatenate(all_labels, axis=0), (-1,11 ))
    elif (params["method"] == "delay_win_select") and before:
        all_probs = np.reshape(np.concatenate(all_probs, axis=0), (-1, 4))
        all_labels = np.reshape(np.concatenate(all_labels, axis=0), (-1,1 ))
    else:
        all_logits = np.reshape(np.concatenate(all_logits, axis=0), (-1, 1))
        all_labels = np.reshape(np.concatenate(all_labels, axis=0), (-1, 1))
        all_probs = np.reshape(np.concatenate(all_probs, axis=0), (-1, 1))
    
    if params["method"] == "FNC":
        all_probs = all_probs / (1-all_probs+1e-8)
        llloss = cal_llloss_with_prob(all_labels, all_probs)
    elif params["method"] == "3class":
        llloss = cal_softmax_cross_entropy_loss(all_labels, all_probs)
    elif params["method"] == "delay_win_adapt":
        cv_prop = tf.reshape(tf.cast(all_probs[:,0], tf.float32), (-1, 1))
        time_prop = tf.reshape(tf.cast(all_probs[:,1], tf.float32), (-1, 1))
        time_prop_numpy = np.reshape(time_prop.numpy(), (-1, 1))
        print "pcvr"
        print tf.reduce_mean(cv_prop)
        print "ptime"
        print tf.reduce_mean(time_prop)
        return time_prop_numpy
    elif params["method"] == "win_adapt" or params["method"] == "delay_win_select":
        cv_prop = tf.reshape(tf.cast(all_probs[:,0], tf.float32), (-1, 1))
        cv_prop_numpy = np.reshape(cv_prop.numpy(), (-1, 1))
        time15_prop = tf.reshape(tf.cast(all_probs[:,1], tf.float32), (-1, 1))
        time15_prop_numpy = np.reshape(time15_prop.numpy(), (-1, 1))
        time30_prop = tf.reshape(tf.cast(all_probs[:,2], tf.float32), (-1, 1))
        time30_prop_numpy = np.reshape(time30_prop.numpy(), (-1, 1))
        time60_prop = tf.reshape(tf.cast(all_probs[:,3], tf.float32), (-1, 1))
        time60_prop_numpy = np.reshape(time60_prop.numpy(), (-1, 1))

        cv_label = tf.reshape(tf.cast(all_labels[:,0], tf.float32), (-1, 1))
        cv_label_numpy = np.reshape(cv_label.numpy(), (-1, 1))

        if not before:

            time_15_label = tf.reshape(tf.cast(all_labels[:,1], tf.float32), (-1, 1))
            time_15_label_numpy = np.reshape(time_15_label.numpy(), (-1, 1))

        print "pcvr"
        print tf.reduce_mean(cv_prop)
        print "ptime15"
        print tf.reduce_mean(time15_prop)
        print "ptime30"
        print tf.reduce_mean(time30_prop)
        print "ptime60"
        print tf.reduce_mean(time60_prop)
        if not before:
            print 'time_15_label_numpy'
            print time_15_label_numpy
            print 'cv_prop_numpy*time15_prop_numpy'
            print cv_prop_numpy*time15_prop_numpy

            win_auc = cal_auc(time_15_label_numpy, cv_prop_numpy*time15_prop_numpy)
            print "win_auc"
            print win_auc

        cv_auc = cal_auc(cv_label_numpy, cv_prop_numpy)
        print "cv_auc"
        print cv_auc

        return all_probs#, time15_prop_numpy
 
    elif params["method"] == "win_time" or params["method"] == "test":
        cv_prop = tf.reshape(tf.cast(all_probs[:,0], tf.float32), (-1, 1))
        time_prop = tf.reshape(tf.cast(all_probs[:,1], tf.float32), (-1, 1))
        win_label = tf.reshape(tf.cast((all_labels[:, 1]), tf.float32), (-1, 1))
        cv_label = tf.reshape(tf.cast((all_labels[:, 0]), tf.float32), (-1, 1))
        #cv_prop = all_probs[:,0]
        #time_prop = all_probs[:,1]
        #win_label = all_labels[:, 1]
        #cv_label = all_labels[:, 0]
        
        win_prop_1 = cv_prop * time_prop
        win_prop_0 = (1-cv_prop) + cv_prop*(1-time_prop)
        print cv_prop.shape
        print time_prop.shape
        print win_label.shape
        print cv_label.shape
        loss_win = -tf.reduce_mean(tf.math.log(win_prop_1) * win_label + tf.math.log(win_prop_0) * (1-win_label))
        loss_cv = -tf.reduce_mean(tf.math.log(cv_prop) * cv_label + tf.math.log(1-cv_prop)* (1-cv_label))

        print "win_label"
        print tf.reduce_mean(win_label)
        print "cv_label"
        print tf.reduce_mean(cv_label)
        print "pcvr"
        print tf.reduce_mean(cv_prop)
        print "pwin"
        print tf.reduce_mean(win_prop_1)
        print "ptime"
        print tf.reduce_mean(time_prop)
        print "test_loss_win"
        print loss_win
        print "test_loss_cv"
        print loss_cv
        
        win_prop_1_numpy = np.reshape(win_prop_1.numpy(), (-1, 1))
        win_prop_0_numpy = np.reshape(win_prop_0.numpy(), (-1, 1))
        win_label_numpy = np.reshape(win_label.numpy(), (-1, 1))
        cv_label_numpy = np.reshape(cv_label.numpy(),(-1, 1))
        cv_prop_numpy = np.reshape(cv_prop.numpy(),(-1, 1))
        print win_prop_1_numpy.shape, win_prop_0_numpy.shape, win_label_numpy.shape, cv_label_numpy.shape

        win_auc = cal_auc(win_label_numpy, win_prop_1_numpy)
        win2_auc = cal_auc(cv_label_numpy, win_prop_1_numpy)
        cv_auc = cal_auc(cv_label_numpy, cv_prop_numpy)
        print "win_auc"
        print win_auc
        print "win2_auc"
        print win2_auc

        print "cv_auc"
        print cv_auc
        return cv_auc, win_auc,tf.reduce_mean(cv_label),tf.reduce_mean(cv_label),time_prop
    elif params["method"] == "win_time_test":
        print 'test win_time_test'
        print all_labels.shape, all_logits.shape
        all_labels = np.reshape(np.reshape(all_labels, (-1, 2))[:,0], (-1,))
        all_logits = np.reshape(all_logits,(-1,))
        all_probs = np.reshape(all_probs,(-1,))
        print all_labels.shape, all_logits.shape,all_probs.shape
        #cv_label_numpy = np.reshape(cv_label.numpy(), (-1, 1))
        #labelloss = np.reshape(targets["label"][:,0].numpy(),(-1,1))
        #probloss = np.reshape(tf.nn.sigmoid(outputs["logits"]).numpy(),(-1,1))
        #logitloss = np.reshape(outputs["logits"].numpy(),(-1,1))
        #llloss = cal_llloss_with_logits(labelloss, logitloss)
        #print "step%d test_loss:%f" %(step,llloss)
        #auc = cal_auc(labelloss, probloss)

        llloss = cal_llloss_with_logits(all_labels, all_logits) 
    else:
        llloss = cal_llloss_with_logits(all_labels, all_logits)

    #llloss = cal_llloss_with_logits(all_labels, all_logits)
    labelprint = np.mean(all_labels,axis=0)
    pctr = np.mean(tf.nn.softmax(all_probs),axis=0)
    print "test_label:" 
    print labelprint
    print "test_pctr" 
    print pctr
    print "test_loss" 
    print llloss
    if params["method"] == "3class":
        auc = 0
        prauc = 0
        
        batch_size = all_logits.shape[0]
        return auc,labelprint,pctr
    print "test_loss:%f" %llloss
    auc = cal_auc(all_labels, all_probs)
    print "test_auc:%f" %auc
    prauc = cal_prauc(all_labels, all_probs)
    print "test_prauc:%f" %prauc
    batch_size = all_logits.shape[0]
    return auc,labelprint,pctr


def train(model, optimizer, train_data, params):
    for step, batch in enumerate(tqdm(train_data), 1):
        batch_x = batch[0]
        batch_y = batch[1]
        targets = {"label": batch_y}
        optim_step(step, model, batch_x, targets, optimizer,
                   get_loss_fn(params["loss"]), params)
        #break


def run(params):
    dataset = get_criteo_dataset(params)
    train_dataset = dataset["train"]
    test_dataset = dataset["test"]
    train_data = tf.data.Dataset.from_tensor_slices(
        (dict(train_dataset["x"]), train_dataset["labels"]))
    train_data = train_data.batch(params["batch_size"]).prefetch(1)
    test_data = tf.data.Dataset.from_tensor_slices(
        (dict(test_dataset["x"]), test_dataset["labels"]))
    test_data = test_data.batch(params["batch_size"]).prefetch(1)
    model = get_model(params["model"], params)
    optimizer = get_optimizer(params["optimizer"], params)
    best_acc = 0
    for ep in range(params["epoch"]):
        train(model, optimizer, train_data, params)
        model.save_weights(params["model_ckpt_path"], save_format="tf")
        if params["method"] == "ES-DFM" or params["method"] == "DFM":
            continue
        test(model, test_data, params)

