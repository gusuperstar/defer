import copy
import os
import h5py
import pandas as pd
import tensorflow as tf
from tensorflow import feature_column
import numpy as np
import pickle
import matplotlib.pyplot as plt
from utils import parse_float_arg

import pretrain 
from models import get_model
from loss import get_loss_fn
from utils import get_optimizer
from metrics import cal_llloss_with_logits, cal_auc, cal_prauc, cal_llloss_with_prob, cal_softmax_cross_entropy_loss
from tqdm import tqdm
import numpy as np
import tensorflow as tf
from collections import defaultdict
from utils import ScalarMovingAverage


MIN_SAMPLE_PER_HOUR=100

SECONDS_5_MINS = 60*5
SECONDS_10_MINS = 60*10
SECONDS_A_QUATER = 60*15
SECONDS_2_QUATER = 60*30
SECONDS_3_QUATER = 60*45
SECONDS_A_DAY = 60*60*24
SECONDS_AN_HOUR = 60*60
SECONDS_DELAY_NORM = 1
SECONDS_FSIW_NORM = SECONDS_A_DAY*5
# the input of neural network should be normalized


def get_data_df(params):
    df = pd.read_csv(params["data_path"], sep="\t", header=None)
    click_ts = df[df.columns[0]].to_numpy()
    pay_ts = df[df.columns[1]].fillna(-1).to_numpy()

    pos_mask = pay_ts > 0
    pos_mask2 = pay_ts < 0
    gap = pay_ts[pos_mask] -click_ts[pos_mask]
    print 'pay >0 <0:'
    total = pay_ts[pos_mask].shape[0]
    print  pay_ts[pos_mask].shape
    print 'pay_ts.shape'
    print pay_ts.shape
    print 'time gap mean:'
    print np.mean(gap,axis=0)
    print np.max(gap,axis=0)
    print np.min(gap,axis=0)

    print 'min'
    pos_mask = pay_ts - click_ts > SECONDS_5_MINS
    print  pay_ts[pos_mask].shape[0]*1.0/total
    pos_mask = pay_ts - click_ts > SECONDS_10_MINS
    print  pay_ts[pos_mask].shape[0]*1.0/total

    print 'win quater'
    pos_mask = pay_ts - click_ts > SECONDS_A_QUATER
    print  pay_ts[pos_mask].shape[0]*1.0/total
    pos_mask = pay_ts - click_ts > SECONDS_2_QUATER
    print  pay_ts[pos_mask].shape[0]*1.0/total
    pos_mask = pay_ts - click_ts > SECONDS_3_QUATER
    print  pay_ts[pos_mask].shape[0]*1.0/total
    print 'win hour'
    pos_mask = pay_ts - click_ts > SECONDS_AN_HOUR 
    print  pay_ts[pos_mask].shape[0]*1.0/total
    pos_mask = pay_ts - click_ts > SECONDS_AN_HOUR*3
    print  pay_ts[pos_mask].shape[0]*1.0/total
    pos_mask = pay_ts - click_ts > SECONDS_AN_HOUR*6
    print  pay_ts[pos_mask].shape[0]*1.0/total
    pos_mask = pay_ts - click_ts > SECONDS_AN_HOUR*12
    print  pay_ts[pos_mask].shape[0]*1.0/total
    print 'win day'
    pos_mask = pay_ts - click_ts > SECONDS_A_DAY
    print  pay_ts[pos_mask].shape[0]*1.0/total
    pos_mask = pay_ts - click_ts > SECONDS_AN_HOUR*48
    print  pay_ts[pos_mask].shape[0]*1.0/total
    pos_mask = pay_ts - click_ts > SECONDS_AN_HOUR*72
    print  pay_ts[pos_mask].shape[0]*1.0/total
    pos_mask = pay_ts - click_ts > SECONDS_AN_HOUR*96
    print  pay_ts[pos_mask].shape[0]*1.0/total
    pos_mask = pay_ts - click_ts > SECONDS_AN_HOUR*120
    print  pay_ts[pos_mask].shape[0]*1.0/total
    pos_mask = pay_ts - click_ts > SECONDS_AN_HOUR*144
    print  pay_ts[pos_mask].shape[0]*1.0/total
    pos_mask = pay_ts - click_ts > SECONDS_AN_HOUR*168
    print  pay_ts[pos_mask].shape[0]*1.0/total
    pos_mask = pay_ts - click_ts > SECONDS_AN_HOUR*192
    print  pay_ts[pos_mask].shape[0]*1.0/total
    pos_mask = pay_ts - click_ts > SECONDS_AN_HOUR*240
    print  pay_ts[pos_mask].shape[0]*1.0/total

    if params["ddline"] > 0:
        print 'ddline 1'
        pos_mask = pay_ts - click_ts > params["ddline"] * SECONDS_A_DAY
        pay_ts[pos_mask] = -1
        print params["ddline"],pay_ts[pos_mask].shape
        

    df = df[df.columns[2:]]
    for c in df.columns[8:]:
        df[c] = df[c].fillna("")
        df[c] = df[c].astype(str)
    for c in df.columns[:8]:
        df[c] = df[c].fillna(-1)
        df[c] = (df[c] - df[c].min())/(df[c].max() - df[c].min())
    df.columns = [str(i) for i in range(17)]
    df.reset_index(inplace=True)
    return df, click_ts, pay_ts


class DataDF(object):

    def __init__(self, features, click_ts, pay_ts, sample_ts=None, labels=None, delay_label=None, time_cut=None):
        self.x = features.copy(deep=True)
        self.click_ts = copy.deepcopy(click_ts)
        self.pay_ts = copy.deepcopy(pay_ts)
        self.delay_label = delay_label
        self.time_cut = time_cut
        if sample_ts is not None:
            self.sample_ts = copy.deepcopy(sample_ts)
        else:
            self.sample_ts = copy.deepcopy(click_ts)
        if labels is not None:
            self.labels = copy.deepcopy(labels)
        else:
            self.labels = (pay_ts > 0).astype(np.int32)

    def sub_days(self, start_day, end_day):
        start_ts = start_day*SECONDS_A_DAY
        end_ts = end_day*SECONDS_A_DAY
        mask = np.logical_and(self.sample_ts >= start_ts,
                              self.sample_ts < end_ts)
        return DataDF(self.x.iloc[mask],
                      self.click_ts[mask],
                      self.pay_ts[mask],
                      self.sample_ts[mask],
                      self.labels[mask])

    def sub_hours(self, start_hour, end_hour):
        start_ts = start_hour*SECONDS_AN_HOUR
        end_ts = end_hour*SECONDS_AN_HOUR
        mask = np.logical_and(self.sample_ts >= start_ts,
                              self.sample_ts < end_ts)
        return DataDF(self.x.iloc[mask],
                      self.click_ts[mask],
                      self.pay_ts[mask],
                      self.sample_ts[mask],
                      self.labels[mask])

    def del_future_label(self, start_day, end_day):
        start_ts = start_day*SECONDS_A_DAY
        end_ts = end_day*SECONDS_A_DAY

        pos_mask = self.pay_ts > end_ts

        print 'del_future_label:', np.sum(pos_mask, axis=0), pos_mask.shape[0]

        #x = pd.concat(
        #    (self.x.copy(deep=True), self.x.iloc[pos_mask].copy(deep=True)))
        #sample_ts = np.concatenate(
        #    [self.click_ts, self.pay_ts[pos_mask]], axis=0)
        #click_ts = np.concatenate(
        #    [self.click_ts, self.click_ts[pos_mask]], axis=0)
        #pay_ts = np.concatenate([self.pay_ts, self.pay_ts[pos_mask]], axis=0)
        x = copy.deepcopy(self.x)
        sample_ts = copy.deepcopy(self.sample_ts)
        click_ts = copy.deepcopy(self.click_ts)
        pay_ts = copy.deepcopy(self.pay_ts)
        labels = copy.deepcopy(self.labels)
        print 'del_future_label before:',np.sum(labels, axis=0)
        pay_ts[pos_mask] = -1
        labels[pos_mask] = 0
        print 'del_future_label after:',np.sum(labels, axis=0)
        #labels = np.concatenate([labels, np.ones((np.sum(pos_mask),))], axis=0)
        idx = list(range(x.shape[0]))
        idx = sorted(idx, key=lambda x: sample_ts[x])
        return DataDF(x.iloc[idx],
                      click_ts[idx],
                      pay_ts[idx],
                      sample_ts[idx],
                      labels[idx])


    def add_nowin_auc(self):
        pos_mask = self.pay_ts - self.click_ts > SECONDS_A_QUATER #self.pay_ts > 0
        pos_mask2 = self.pay_ts <= 0
        x = pd.concat(
            ( self.x.iloc[pos_mask].copy(deep=True), self.x.iloc[pos_mask2].copy(deep=True)))
        sample_ts = np.concatenate(
            [self.click_ts[pos_mask], self.click_ts[pos_mask2]], axis=0)
        click_ts = np.concatenate(
            [self.click_ts[pos_mask], self.click_ts[pos_mask2]], axis=0)
        pay_ts = np.concatenate([self.pay_ts[pos_mask], self.pay_ts[pos_mask2]], axis=0)
        #labels = copy.deepcopy(self.labels)
        #labels[pos_mask] = 0
        #labels = np.concatenate([np.ones((np.sum(pos_mask),)), np.zeros((np.sum(pos_mask),))], axis=0)
        labels = np.concatenate([np.ones(self.click_ts[pos_mask].shape), np.zeros(self.click_ts[pos_mask2].shape)], axis=0)
        idx = list(range(x.shape[0]))
        idx = sorted(idx, key=lambda x: sample_ts[x])
        return DataDF(x.iloc[idx],
                      click_ts[idx],
                      pay_ts[idx],
                      sample_ts[idx],
                      labels[idx])

    


    def add_fake_neg(self):
        pos_mask = self.pay_ts > 0
        x = pd.concat(
            (self.x.copy(deep=True), self.x.iloc[pos_mask].copy(deep=True)))
        sample_ts = np.concatenate(
            [self.click_ts, self.pay_ts[pos_mask]], axis=0)
        click_ts = np.concatenate(
            [self.click_ts, self.click_ts[pos_mask]], axis=0)
        pay_ts = np.concatenate([self.pay_ts, self.pay_ts[pos_mask]], axis=0)
        labels = copy.deepcopy(self.labels)
        labels[pos_mask] = 0
        labels = np.concatenate([labels, np.ones((np.sum(pos_mask),))], axis=0)
        idx = list(range(x.shape[0]))
        idx = sorted(idx, key=lambda x: sample_ts[x])
        return DataDF(x.iloc[idx],
                      click_ts[idx],
                      pay_ts[idx],
                      sample_ts[idx],
                      labels[idx])

    def add_fake_neg2(self, params):
        rnwin = params["rnwin"]
        pos_mask = self.pay_ts > 0
        #pos_mask2 = self.pay_ts <= 0
        pos_mask2 = np.logical_or((self.pay_ts - self.click_ts > rnwin*SECONDS_AN_HOUR), (self.pay_ts <= 0))
        x = pd.concat(
            (self.x.copy(deep=True), self.x.iloc[pos_mask].copy(deep=True), self.x.iloc[pos_mask2].copy(deep=True)))
        sample_ts = np.concatenate(
            [self.click_ts, self.pay_ts[pos_mask], self.click_ts[pos_mask2]+rnwin*SECONDS_AN_HOUR], axis=0)
        click_ts = np.concatenate(
            [self.click_ts, self.click_ts[pos_mask], self.click_ts[pos_mask2]], axis=0)
        pay_ts = np.concatenate([self.pay_ts, self.pay_ts[pos_mask], self.pay_ts[pos_mask2]], axis=0)
        labels = copy.deepcopy(self.labels)
        labels[pos_mask] = 0
        labels = np.concatenate([labels, np.ones((np.sum(pos_mask),)), np.zeros((np.sum(pos_mask2),))], axis=0)
        idx = list(range(x.shape[0]))
        idx = sorted(idx, key=lambda x: sample_ts[x])
        return DataDF(x.iloc[idx],
                      click_ts[idx],
                      pay_ts[idx],
                      sample_ts[idx],
                      labels[idx])

    def add_fake_neg3(self, cut_size, params):
        #rnwin = params["rnwin"]
        pos_mask = self.pay_ts - self.click_ts > cut_size
        #pos_mask2 = self.pay_ts <= 0
        #pos_mask2 = np.logical_or((self.pay_ts - self.click_ts > rnwin*SECONDS_AN_HOUR), (self.pay_ts <= 0))
        x = self.x.copy(deep=True)
        sample_ts = self.click_ts+cut_size
        click_ts = self.click_ts
        pay_ts = self.pay_ts
        labels = copy.deepcopy(self.labels)
        labels[pos_mask] = 0
        #labels = np.concatenate([labels, np.ones((np.sum(pos_mask),)), np.zeros((np.sum(pos_mask2),))], axis=0)
        idx = list(range(x.shape[0]))
        idx = sorted(idx, key=lambda x: sample_ts[x])
        return DataDF(x.iloc[idx],
                      click_ts[idx],
                      pay_ts[idx],
                      sample_ts[idx],
                      labels[idx])


    def add_esdfm_cut_fake_neg2(self, cut_size,params):
        rnwin = params["rnwin"]
        mask = self.pay_ts - self.click_ts > cut_size
        pos_mask2 = np.logical_or((self.pay_ts - self.click_ts > rnwin*SECONDS_AN_HOUR), (self.pay_ts <= 0))
        x = pd.concat(
            (self.x.copy(deep=True), self.x.iloc[mask].copy(deep=True), self.x.iloc[pos_mask2].copy(deep=True)))
        sample_ts = np.concatenate(
            [self.click_ts+cut_size, self.pay_ts[mask], self.click_ts[pos_mask2]+rnwin*SECONDS_AN_HOUR], axis=0)  # negative samples can only be used after cut_size hours
        click_ts = np.concatenate(
            [self.click_ts, self.click_ts[mask], self.click_ts[pos_mask2]], axis=0)
        pay_ts = np.concatenate([self.pay_ts, self.pay_ts[mask], self.pay_ts[pos_mask2]], axis=0)
        labels = copy.deepcopy(self.labels)
        labels[mask] = 0  # fake negatives
        # insert delayed positives
        labels = np.concatenate([labels, np.ones((np.sum(mask),)), np.zeros((np.sum(pos_mask2),))], axis=0)
        idx = list(range(x.shape[0]))
        idx = sorted(idx, key=lambda x: sample_ts[x])  # sort by sampling time
        return DataDF(x.iloc[idx],
                      click_ts[idx],
                      pay_ts[idx],
                      sample_ts[idx],
                      labels[idx])

    def add_esdfm_cut_fake_neg3(self, cut_size,params):
        rnwin = params["rnwin"]
        mask = self.pay_ts - self.click_ts > cut_size
        mask2 = np.logical_and(self.pay_ts - self.click_ts <= cut_size,(self.pay_ts > 0))
        pos_mask2 = np.logical_or((self.pay_ts - self.click_ts > rnwin*SECONDS_AN_HOUR), (self.pay_ts <= 0))
        x = pd.concat(
            (self.x.copy(deep=True), self.x.iloc[mask].copy(deep=True), self.x.iloc[pos_mask2].copy(deep=True), self.x.iloc[mask2].copy(deep=True)))
        sample_ts = np.concatenate(
            [self.click_ts+cut_size, self.pay_ts[mask], self.click_ts[pos_mask2]+rnwin*SECONDS_AN_HOUR, self.pay_ts[mask2]+rnwin*SECONDS_AN_HOUR], axis=0)  # negative samples can only be used after cut_size hours
        click_ts = np.concatenate(
            [self.click_ts, self.click_ts[mask], self.click_ts[pos_mask2], self.click_ts[mask2]], axis=0)
        pay_ts = np.concatenate([self.pay_ts, self.pay_ts[mask], self.pay_ts[pos_mask2],self.pay_ts[mask2]], axis=0)
        labels = copy.deepcopy(self.labels)
        labels[mask] = 0  # fake negatives
        # insert delayed positives
        labels = np.concatenate([labels, np.ones((np.sum(mask),)), np.zeros((np.sum(pos_mask2),)),np.ones((np.sum(mask2),))], axis=0)
        idx = list(range(x.shape[0]))
        idx = sorted(idx, key=lambda x: sample_ts[x])  # sort by sampling time
        return DataDF(x.iloc[idx],
                      click_ts[idx],
                      pay_ts[idx],
                      sample_ts[idx],
                      labels[idx])



    def only_pos(self):
        pos_mask = self.pay_ts > 0
        print(np.mean(pos_mask))
        print(self.pay_ts[pos_mask].shape)
        return DataDF(self.x.iloc[pos_mask],
                      self.click_ts[pos_mask],
                      self.pay_ts[pos_mask],
                      self.sample_ts[pos_mask],
                      self.labels[pos_mask])

    def to_tn(self):
        mask = np.logical_or(self.pay_ts < 0, self.pay_ts -
                             self.click_ts > SECONDS_AN_HOUR)
        x = self.x.iloc[mask]
        sample_ts = self.sample_ts[mask]
        click_ts = self.click_ts[mask]
        pay_ts = self.pay_ts[mask]
        label = pay_ts < 0
        return DataDF(x,
                      click_ts,
                      pay_ts,
                      sample_ts,
                      label)

    def to_dp(self):
        x = self.x
        sample_ts = self.sample_ts
        click_ts = self.click_ts
        pay_ts = self.pay_ts
        label = pay_ts - click_ts > SECONDS_AN_HOUR
        return DataDF(x,
                      click_ts,
                      pay_ts,
                      sample_ts,
                      label)

    def add_esdfm_cut_fake_neg(self, cut_size):
        mask = self.pay_ts - self.click_ts > cut_size
        x = pd.concat(
            (self.x.copy(deep=True), self.x.iloc[mask].copy(deep=True)))
        sample_ts = np.concatenate(
            [self.click_ts+cut_size, self.pay_ts[mask]], axis=0)  # negative samples can only be used after cut_size hours
        click_ts = np.concatenate(
            [self.click_ts, self.click_ts[mask]], axis=0)
        pay_ts = np.concatenate([self.pay_ts, self.pay_ts[mask]], axis=0)
        labels = copy.deepcopy(self.labels)
        labels[mask] = 0  # fake negatives
        # insert delayed positives
        labels = np.concatenate([labels, np.ones((np.sum(mask),))], axis=0)
        idx = list(range(x.shape[0]))
        idx = sorted(idx, key=lambda x: sample_ts[x])  # sort by sampling time
        return DataDF(x.iloc[idx],
                      click_ts[idx],
                      pay_ts[idx],
                      sample_ts[idx],
                      labels[idx])

    def add_3class_cut_fake_neg(self, cut_size):
        mask = self.pay_ts - self.click_ts > cut_size
        mask_3 = (self.pay_ts == 0)
        x = pd.concat(
            (self.x.copy(deep=True), self.x.iloc[mask].copy(deep=True)))
        sample_ts = np.concatenate(
            [self.click_ts+cut_size, self.pay_ts[mask]], axis=0)  # negative samples can only be used after cut_size hours
        click_ts = np.concatenate(
            [self.click_ts, self.click_ts[mask]], axis=0)
        pay_ts = np.concatenate([self.pay_ts, self.pay_ts[mask]], axis=0)

        #train_label_tn = np.reshape(pay_ts > 0, (-1, 1))
        train_label_spm = np.concatenate([np.zeros(self.pay_ts.shape), np.ones(self.pay_ts[mask].shape)], axis=0).reshape(-1,1)
        train_label_winneg = np.concatenate([np.logical_or((self.pay_ts - self.click_ts > cut_size), self.pay_ts <= 0), np.zeros(self.pay_ts[mask].shape)], axis=0).reshape(-1,1)
        train_label_winpos = np.concatenate([np.logical_and((self.pay_ts - self.click_ts <= cut_size), self.pay_ts > 0), np.zeros(self.pay_ts[mask].shape)], axis=0).reshape(-1,1) 
       
        #train_label = np.reshape(pay_ts > 0, (-1, 1))
        labels = np.concatenate([train_label_spm, train_label_winneg, train_label_winpos], axis=1)
        print 'self.pay_ts'
        print self.pay_ts.shape
        print 'labels.shape'
        print labels.shape
        print 'allsum'
        print np.sum(labels)
        print 'linesum'
        print np.sum(labels, axis=0)
        print 'label'
        print labels
        print 'mask_3'
        print mask_3
        print self.pay_ts[mask_3].shape
        #train_label = copy.deepcopy(self.labels)
        #train_label[mask] = 0  # fake negatives
        #labels = np.concatenate([train_label_tn, train_label_dp, train_label], axis=1)
        # insert delayed positives
        #labels = np.concatenate([labels, np.ones((np.sum(mask),))], axis=0)
        idx = list(range(x.shape[0]))
        idx = sorted(idx, key=lambda x: sample_ts[x])  # sort by sampling time
        return DataDF(x.iloc[idx],
                      click_ts[idx],
                      pay_ts[idx],
                      sample_ts[idx],
                      labels[idx])

    def add_delay_wintime_cut_fake_neg(self, cut_size):

        #ddline = int(params["ddline"])*SECONDS_A_DAY
        #print 'ddline'
        #print ddline

        mask_spm1 = self.pay_ts - self.click_ts > cut_size #np.logical_and((self.pay_ts - self.click_ts > cut_size), (self.pay_ts - self.click_ts < ddline))
        mask_spm0 = np.logical_or((self.pay_ts - self.click_ts > SECONDS_A_DAY), (self.pay_ts <= 0))  #self.pay_ts <= 0
        mask_3 = (self.pay_ts == 0)
        x = pd.concat(
            (self.x.copy(deep=True), self.x.iloc[mask_spm1].copy(deep=True), self.x.iloc[mask_spm0].copy(deep=True)))
        sample_ts = np.concatenate(
            [self.click_ts+cut_size, self.pay_ts[mask_spm1], self.click_ts[mask_spm0]+SECONDS_A_DAY], axis=0)  # negative samples can only be used after cut_size hours
        click_ts = np.concatenate(
            [self.click_ts, self.click_ts[mask_spm1], self.click_ts[mask_spm0]], axis=0)
        pay_ts = np.concatenate([self.pay_ts, self.pay_ts[mask_spm1], self.pay_ts[mask_spm0]], axis=0)

        train_label_11 = np.concatenate([np.zeros(self.pay_ts.shape), np.ones(self.pay_ts[mask_spm1].shape), np.zeros(self.pay_ts[mask_spm0].shape)], axis=0).reshape(-1,1)
        train_label_00 = np.concatenate([np.logical_or((self.pay_ts - self.click_ts > cut_size), self.pay_ts <= 0), np.zeros(self.pay_ts[mask_spm1].shape), np.zeros(self.pay_ts[mask_spm0].shape)], axis=0).reshape(-1,1)
        train_label_01 = np.concatenate([np.logical_and((self.pay_ts - self.click_ts <= cut_size),self.pay_ts > 0), np.zeros(self.pay_ts[mask_spm1].shape), np.zeros(self.pay_ts[mask_spm0].shape)], axis=0).reshape(-1,1)
        train_label_10 = np.concatenate([np.zeros(self.pay_ts.shape), np.zeros(self.pay_ts[mask_spm1].shape), np.ones(self.pay_ts[mask_spm0].shape)], axis=0).reshape(-1,1)

        #train_label = np.reshape(pay_ts > 0, (-1, 1))
        labels = np.concatenate([train_label_01, train_label_00, train_label_11, train_label_10], axis=1)
 
        print 'labels.shape'
        print labels.shape
        print 'allsum'
        print np.sum(labels)
        print 'linesum'
        print np.sum(labels, axis=0)
        print 'label'
        print labels
        print 'mask_3'
        print mask_3
        print self.pay_ts[mask_3].shape

        idx = list(range(x.shape[0]))
        idx = sorted(idx, key=lambda x: sample_ts[x])  # sort by sampling time
        return DataDF(x.iloc[idx],
                      click_ts[idx],
                      pay_ts[idx],
                      sample_ts[idx],
                      labels[idx])

    def add_delay_wintime_cut_fake_neg2(self, cut_size):
        print "add_delay_wintime_cut_fake_neg2"
        #ddline = int(params["ddline"])*SECONDS_A_DAY
        #print 'ddline'
        #print ddline

        mask_spm1 = self.pay_ts - self.click_ts > cut_size #np.logical_and((self.pay_ts - self.click_ts > cut_size), (self.pay_ts - self.click_ts < ddline))
        mask_spm0 = np.logical_or((self.pay_ts - self.click_ts > SECONDS_A_DAY), (self.pay_ts <= 0))  #self.pay_ts <= 0
        mask_spm2 = np.logical_or((self.pay_ts - self.click_ts <= cut_size), (self.pay_ts > 0))
        mask_3 = (self.pay_ts == 0)
        x = pd.concat(
            (self.x.copy(deep=True), self.x.iloc[mask_spm1].copy(deep=True), self.x.iloc[mask_spm0].copy(deep=True), self.x.iloc[mask_spm2].copy(deep=True)))
        sample_ts = np.concatenate(
            [self.click_ts+cut_size, self.pay_ts[mask_spm1], self.click_ts[mask_spm0]+SECONDS_A_DAY, self.pay_ts[mask_spm2]+SECONDS_A_DAY], axis=0)  # negative samples can only be used after cut_size hours
        click_ts = np.concatenate(
            [self.click_ts, self.click_ts[mask_spm1], self.click_ts[mask_spm0], self.click_ts[mask_spm2]], axis=0)
        pay_ts = np.concatenate([self.pay_ts, self.pay_ts[mask_spm1], self.pay_ts[mask_spm0], self.pay_ts[mask_spm2]], axis=0)

        train_label_11 = np.concatenate([np.zeros(self.pay_ts.shape), np.ones(self.pay_ts[mask_spm1].shape), np.zeros(self.pay_ts[mask_spm0].shape), np.zeros(self.pay_ts[mask_spm2].shape)], axis=0).reshape(-1,1)
        train_label_00 = np.concatenate([np.logical_or((self.pay_ts - self.click_ts > cut_size), self.pay_ts <= 0), np.zeros(self.pay_ts[mask_spm1].shape), np.zeros(self.pay_ts[mask_spm0].shape), np.zeros(self.pay_ts[mask_spm2].shape)], axis=0).reshape(-1,1)
        train_label_01 = np.concatenate([np.logical_and((self.pay_ts - self.click_ts <= cut_size),self.pay_ts > 0), np.zeros(self.pay_ts[mask_spm1].shape), np.zeros(self.pay_ts[mask_spm0].shape), np.ones(self.pay_ts[mask_spm2].shape)], axis=0).reshape(-1,1)
        train_label_10 = np.concatenate([np.zeros(self.pay_ts.shape), np.zeros(self.pay_ts[mask_spm1].shape), np.ones(self.pay_ts[mask_spm0].shape), np.zeros(self.pay_ts[mask_spm2].shape)], axis=0).reshape(-1,1)

        #train_label = np.reshape(pay_ts > 0, (-1, 1))
        labels = np.concatenate([train_label_01, train_label_00, train_label_11, train_label_10], axis=1)

        print 'labels.shape'
        print labels.shape
        print 'allsum'
        print np.sum(labels)
        print 'linesum'
        print np.sum(labels, axis=0)
        print 'label'
        print labels
        print 'mask_3'
        print mask_3
        print self.pay_ts[mask_3].shape

        idx = list(range(x.shape[0]))
        idx = sorted(idx, key=lambda x: sample_ts[x])  # sort by sampling time
        return DataDF(x.iloc[idx],
                      click_ts[idx],
                      pay_ts[idx],
                      sample_ts[idx],
                      labels[idx])


    def add_delay_winadapt2_cut_fake_neg(self, cut_size, time_cut, params):
        cut_hour = 0.25
        cut_size = cut_hour*SECONDS_AN_HOUR
        self.time_cut = time_cut
        print 'self.time_cut.new2'
        print self.time_cut.shape
        print self.time_cut
        print 'self.pay_ts'
        print self.pay_ts.shape
        print self.pay_ts
        print 'self.click_ts'
        print self.click_ts.shape
        print self.click_ts

        self.pay_ts = self.pay_ts.reshape(-1,)
        self.click_ts = self.click_ts.reshape(-1,)
        #self.time_cut = self.time_cut.reshape(-1,)

        cut_hour1 = params["win1"] #0.25
        cut_hour2 = params["win2"] #0.5
        cut_hour3 = params["win3"] #1

        print 'win 1 2 3'
        print cut_hour1, cut_hour2, cut_hour3

        print("ch {}".format(cut_hour1))
        cut_sec1 = int(SECONDS_AN_HOUR * cut_hour1)
        cut_sec2 = int(SECONDS_AN_HOUR * cut_hour2)
        cut_sec3 = int(SECONDS_AN_HOUR * cut_hour3)

        time_win = time_cut*SECONDS_AN_HOUR#[:,0]

        #adatag = False #False True
        adatag = True
        winada = None
        if adatag:
            winada = time_win
        else:
            winada = cut_size
        print 'winada'
        print winada
        #time_pos = time_cut[:,1]
        print 'self.pay_ts'
        print self.pay_ts.shape
        print 'self.click_ts'
        print self.click_ts.shape
        print 'time_win3'
        print time_win.shape
        mask_spm1 = self.pay_ts - self.click_ts > winada#time_win #cut_size #time_win ####################time_win
        #mask_spm2 = self.pay_ts - self.click_ts > cut_size
        print 'succ pass'
        print mask_spm1.shape
        mask_spm0 = self.pay_ts <= 0
        mask_3 = (self.pay_ts == 0)

        x = pd.concat(
            (self.x.copy(deep=True), self.x.iloc[mask_spm1].copy(deep=True)))
        sample_ts = np.concatenate(
            [self.click_ts+winada, self.pay_ts[mask_spm1]], axis=0) ####################time_win
        click_ts = np.concatenate(
            [self.click_ts, self.click_ts[mask_spm1]], axis=0)
        pay_ts = np.concatenate([self.pay_ts, self.pay_ts[mask_spm1]], axis=0)

        labels = copy.deepcopy(self.labels)
        labels[mask_spm1] = 0  # fake negatives
        # insert delayed positives
        label = np.concatenate([labels, np.ones((np.sum(mask_spm1),))], axis=0)
        time_cut_new = np.reshape(time_cut,(-1,1))
        time_cut_new2 = np.concatenate([time_cut_new, time_cut_new[mask_spm1]], axis=0)
        labels = np.concatenate([label,time_cut_new2], axis=1)

        print 'labels.shape'
        print labels.shape
        print 'allsum'
        print np.sum(labels)
        print 'linesum'
        print np.sum(labels, axis=0)
        print 'label'
        print labels

        idx = list(range(x.shape[0]))
        idx = sorted(idx, key=lambda x: sample_ts[x])  # sort by sampling time
        return DataDF(x.iloc[idx],
                      click_ts[idx],
                      pay_ts[idx],
                      sample_ts[idx],
                      labels[idx])



    def add_delay_winadapt_cut_fake_neg(self, cut_size, time_cut, params):
        cut_hour = 0.25
        cut_size = cut_hour*SECONDS_AN_HOUR
        self.time_cut = time_cut
        print 'self.time_cut.new2'
        print self.time_cut.shape
        print self.time_cut
        print 'self.pay_ts'
        print self.pay_ts.shape
        print self.pay_ts
        print 'self.click_ts'
        print self.click_ts.shape
        print self.click_ts

        self.pay_ts = self.pay_ts.reshape(-1,)
        self.click_ts = self.click_ts.reshape(-1,)
        #self.time_cut = self.time_cut.reshape(-1,)

        cut_hour1 = params["win1"] #0.25
        cut_hour2 = params["win2"] #0.5
        cut_hour3 = params["win3"] #1

        print 'win 1 2 3'
        print cut_hour1, cut_hour2, cut_hour3

        print("ch {}".format(cut_hour1))
        cut_sec1 = int(SECONDS_AN_HOUR * cut_hour1)
        cut_sec2 = int(SECONDS_AN_HOUR * cut_hour2)
        cut_sec3 = int(SECONDS_AN_HOUR * cut_hour3)

        time_win = time_cut*SECONDS_AN_HOUR#[:,0]

        #adatag = False #False True
        adatag = True
        winada = None 
        if adatag:
            winada = time_win 
        else:
            winada = cut_size
        print 'winada'
        print winada
        #time_pos = time_cut[:,1]
        print 'self.pay_ts'
        print self.pay_ts.shape
        print 'self.click_ts'
        print self.click_ts.shape
        print 'time_win3'
        print time_win.shape
        mask_spm1 = self.pay_ts - self.click_ts > winada#time_win #cut_size #time_win ####################time_win
        #mask_spm2 = self.pay_ts - self.click_ts > cut_size
        print 'succ pass'
        print mask_spm1.shape
        mask_spm0 = self.pay_ts <= 0
        mask_3 = (self.pay_ts == 0)
        x = pd.concat(
            (self.x.copy(deep=True), self.x.iloc[mask_spm1].copy(deep=True), self.x.iloc[mask_spm0].copy(deep=True)))
        sample_ts = np.concatenate(
            [self.click_ts+winada, self.pay_ts[mask_spm1], self.click_ts[mask_spm0]+SECONDS_A_DAY], axis=0) ####################time_win 
        click_ts = np.concatenate(
            [self.click_ts, self.click_ts[mask_spm1], self.click_ts[mask_spm0]], axis=0)
        pay_ts = np.concatenate([self.pay_ts, self.pay_ts[mask_spm1], self.pay_ts[mask_spm0]], axis=0)

        
        train_label_11 = np.concatenate([np.zeros(self.pay_ts.shape), np.ones(self.pay_ts[mask_spm1].shape), np.zeros(self.pay_ts[mask_spm0].shape)], axis=0).reshape(-1,1)
        train_label_00 = np.concatenate([np.logical_or((self.pay_ts - self.click_ts > winada), self.pay_ts <= 0), np.zeros(self.pay_ts[mask_spm1].shape), np.zeros(self.pay_ts[mask_spm0].shape)], axis=0).reshape(-1,1)
        train_label_01 = np.concatenate([np.logical_and((self.pay_ts - self.click_ts <= winada),self.pay_ts > 0), np.zeros(self.pay_ts[mask_spm1].shape), np.zeros(self.pay_ts[mask_spm0].shape)], axis=0).reshape(-1,1)
        train_label_10 = np.concatenate([np.zeros(self.pay_ts.shape), np.zeros(self.pay_ts[mask_spm1].shape), np.ones(self.pay_ts[mask_spm0].shape)], axis=0).reshape(-1,1)
        train_label_01_15_part = np.logical_and((self.pay_ts - self.click_ts <= cut_sec1),self.pay_ts > 0)
        train_label_01_15 = np.concatenate([np.logical_and(np.logical_and((self.pay_ts - self.click_ts <= cut_sec1),self.pay_ts > 0), (self.pay_ts - self.click_ts <= winada)),
						np.zeros(self.pay_ts[mask_spm1].shape), 
						np.zeros(self.pay_ts[mask_spm0].shape)], axis=0).reshape(-1,1)
        train_label_01_30_part = np.logical_and((self.pay_ts - self.click_ts < cut_sec2), (self.pay_ts - self.click_ts >= cut_sec1))
        train_label_01_30 = np.concatenate([np.logical_and(np.logical_and((self.pay_ts - self.click_ts <= cut_sec2), (self.pay_ts - self.click_ts > cut_sec1)),(self.pay_ts - self.click_ts <= winada)),
                                                np.zeros(self.pay_ts[mask_spm1].shape),
                                                np.zeros(self.pay_ts[mask_spm0].shape)], axis=0).reshape(-1,1)
        '''
        train_label_01_30 = np.concatenate([np.logical_and((self.pay_ts - self.click_ts < cut_sec2), (self.pay_ts - self.click_ts >= cut_sec1)),
						np.zeros(self.pay_ts[mask_spm1].shape), 
						np.zeros(self.pay_ts[mask_spm0].shape)], axis=0).reshape(-1,1)
        '''
        train_label_01_30_sum_part = np.minimum(train_label_01_15_part+train_label_01_30_part, 1)
        train_label_01_30_sum = np.minimum(train_label_01_15+train_label_01_30, 1)
        print 'time_win6'
        print time_win.shape
        #print 'cut_sec2'
        #print cut_sec2.shape
        print 'train_label_01_30_sum'
        print train_label_01_30_sum.shape
        # mask is to solve the quesiotn that label11 is neg for which window
        label_01_30_mask = np.concatenate([np.logical_and((time_win+0.0001 < cut_sec2),(train_label_01_30_sum_part<1)), ############time_win 
						np.zeros(self.pay_ts[mask_spm1].shape),
                                                np.zeros(self.pay_ts[mask_spm0].shape)], axis=0).reshape(-1,1)
        label_01_30_mask = 1-label_01_30_mask
        train_label_01_60_part = np.logical_and((self.pay_ts - self.click_ts <= cut_sec3),self.pay_ts > 0) 
        train_label_01_60 = np.concatenate([np.logical_and(np.logical_and((self.pay_ts - self.click_ts <= cut_sec3),(self.pay_ts - self.click_ts > cut_sec2)),(self.pay_ts - self.click_ts <= winada)),
                                                np.zeros(self.pay_ts[mask_spm1].shape),
                                                np.zeros(self.pay_ts[mask_spm0].shape)], axis=0).reshape(-1,1)
        '''
        train_label_01_60 = np.concatenate([np.logical_and((self.pay_ts - self.click_ts <= cut_sec3),self.pay_ts > 0), 
						np.zeros(self.pay_ts[mask_spm1].shape), 
						np.zeros(self.pay_ts[mask_spm0].shape)], axis=0).reshape(-1,1)
        '''
        train_label_01_60_sum_part = np.minimum(train_label_01_15_part+train_label_01_30_part+train_label_01_60_part, 1)
        train_label_01_60_sum = np.minimum(train_label_01_15+train_label_01_30+train_label_01_60, 1)
        label_01_60_mask = np.concatenate([np.logical_and((time_win+0.0001 < cut_sec3),(train_label_01_60_sum_part<1)), ############time_win
                                                np.zeros(self.pay_ts[mask_spm1].shape),
                                                np.ones(self.pay_ts[mask_spm0].shape)], axis=0).reshape(-1,1)
        label_01_60_mask = 1-label_01_60_mask

        train_label_11 = np.concatenate([np.zeros(self.pay_ts.shape), 
						np.ones(self.pay_ts[mask_spm1].shape), 
						np.zeros(self.pay_ts[mask_spm0].shape)], axis=0).reshape(-1,1)

        train_label_11_15 = np.concatenate([np.zeros(self.pay_ts.shape),
                                                np.logical_and(np.logical_and((self.pay_ts[mask_spm1] - self.click_ts[mask_spm1] <= cut_sec2), (self.pay_ts[mask_spm1] - self.click_ts[mask_spm1] > cut_sec1)),(self.pay_ts[mask_spm1] - self.click_ts[mask_spm1] > winada[mask_spm1])),
                                                np.zeros(self.pay_ts[mask_spm0].shape)], axis=0).reshape(-1,1)

        train_label_11_30 = np.concatenate([np.zeros(self.pay_ts.shape),
                                                np.logical_and(np.logical_and((self.pay_ts[mask_spm1] - self.click_ts[mask_spm1] <= cut_sec3), (self.pay_ts[mask_spm1] - self.click_ts[mask_spm1] > cut_sec2)),(self.pay_ts[mask_spm1] - self.click_ts[mask_spm1] > winada[mask_spm1])),
                                                np.zeros(self.pay_ts[mask_spm0].shape)], axis=0).reshape(-1,1)

        train_label_11_60 = np.concatenate([np.zeros(self.pay_ts.shape),
                                                np.logical_and(np.logical_and((self.pay_ts[mask_spm1] - self.click_ts[mask_spm1] > cut_sec3),self.pay_ts[mask_spm1] > 0),(self.pay_ts[mask_spm1] - self.click_ts[mask_spm1] > winada[mask_spm1])),
                                                np.zeros(self.pay_ts[mask_spm0].shape)], axis=0).reshape(-1,1)

        labels = np.concatenate([train_label_11, train_label_10, train_label_01_15, train_label_01_30, train_label_01_60, 
					train_label_01_30_sum, train_label_01_60_sum, label_01_30_mask, label_01_60_mask, 
                                        train_label_00, train_label_01, 
                                        train_label_11_15, train_label_11_30, train_label_11_60], axis=1)




        #                       [ 2306355. 12059905.  1243653.   244753.  1661859.  
					#1487993.  1661859.  15609913.  1661859. 14366260.  1243653.]
        #cv 2306355.+1243653.= 3,550,008
        #allsample 2306355.+12059905.+1243653.=15,609,913
	#cv15 1243653.    0.35
        #cv30sum 1487993. 0.42 
        #cv60sum 1661859  0.468
        #labels.shape (29976173, 11).
        print 'labels.shape'
        print labels.shape
        print 'allsum'
        print np.sum(labels)
        print 'linesum'
        print np.sum(labels, axis=0)
        print 'label'
        print labels
        print 'label_01_30_mask'
        print np.sum(label_01_30_mask, axis=0)

        idx = list(range(x.shape[0]))
        idx = sorted(idx, key=lambda x: sample_ts[x])  # sort by sampling time
        return DataDF(x.iloc[idx],
                      click_ts[idx],
                      pay_ts[idx],
                      sample_ts[idx],
                      labels[idx])


    def add_likeli_cut_fake_neg(self, cut_size):
        mask = self.pay_ts - self.click_ts > cut_size
        x = pd.concat(
            (self.x.copy(deep=True), self.x.iloc[mask].copy(deep=True)))
        sample_ts = np.concatenate(
            [self.click_ts+cut_size, self.pay_ts[mask]], axis=0)  # negative samples can only be used after cut_size hours
        click_ts = np.concatenate(
            [self.click_ts, self.click_ts[mask]], axis=0)
        pay_ts = np.concatenate([self.pay_ts, self.pay_ts[mask]], axis=0)

        #train_label_tn = np.reshape(pay_ts > 0, (-1, 1))
        train_label_spm = np.concatenate([np.zeros(self.pay_ts.shape), np.ones(self.pay_ts[mask].shape)], axis=0).reshape(-1,1)
        train_label_winneg = np.concatenate([np.logical_or((self.pay_ts - self.click_ts > cut_size), self.pay_ts < 0), np.zeros(self.pay_ts[mask].shape)], axis=0).reshape(-1,1)
        train_label_winpos = np.concatenate([self.pay_ts - self.click_ts <= cut_size, np.zeros(self.pay_ts[mask].shape)], axis=0).reshape(-1,1)

        #train_label = np.reshape(pay_ts > 0, (-1, 1))
        labels = np.concatenate([train_label_spm, train_label_winneg, train_label_winpos], axis=1)

        #train_label = copy.deepcopy(self.labels)
        #train_label[mask] = 0  # fake negatives
        #labels = np.concatenate([train_label_tn, train_label_dp, train_label], axis=1)
        # insert delayed positives
        #labels = np.concatenate([labels, np.ones((np.sum(mask),))], axis=0)
        idx = list(range(x.shape[0]))
        idx = sorted(idx, key=lambda x: sample_ts[x])  # sort by sampling time
        return DataDF(x.iloc[idx],
                      click_ts[idx],
                      pay_ts[idx],
                      sample_ts[idx],
                      labels[idx])


    def to_fsiw_1(self, cd, T):  # build pre-training dataset 1 of FSIW
        mask = np.logical_and(self.click_ts < T-cd, self.pay_ts > 0)
        x = self.x.iloc[mask].copy(deep=True)
        pay_ts = self.pay_ts[mask]
        click_ts = self.click_ts[mask]
        sample_ts = self.click_ts[mask]
        label = np.zeros((x.shape[0],))
        label[pay_ts < T - cd] = 1
        # FSIW needs elapsed time information
        x.insert(x.shape[1], column="elapse", value=(
            T-click_ts-cd)/SECONDS_FSIW_NORM)
        return DataDF(x,
                      click_ts,
                      pay_ts,
                      sample_ts,
                      label)

    def to_fsiw_0(self, cd, T):  # build pre-training dataset 0 of FSIW
        mask = np.logical_or(self.pay_ts >= T-cd, self.pay_ts < 0)
        mask = np.logical_and(self.click_ts < T-cd, mask)
        x = self.x.iloc[mask].copy(deep=True)
        pay_ts = self.pay_ts[mask]
        click_ts = self.click_ts[mask]
        sample_ts = self.sample_ts[mask]
        label = np.zeros((x.shape[0],))
        label[pay_ts < 0] = 1
        x.insert(x.shape[1], column="elapse", value=(
            T-click_ts-cd)/SECONDS_FSIW_NORM)
        return DataDF(x,
                      click_ts,
                      pay_ts,
                      sample_ts,
                      label)

    def to_fsiw_tune(self, cut_ts):
        label = np.logical_and(self.pay_ts > 0, self.pay_ts < cut_ts)
        self.x.insert(self.x.shape[1], column="elapse", value=(
            cut_ts - self.click_ts)/SECONDS_FSIW_NORM)
        return DataDF(self.x,
                      self.click_ts,
                      self.pay_ts,
                      self.sample_ts,
                      label)

    def to_dfm_tune(self, cut_ts):
        label = np.logical_and(self.pay_ts > 0, self.pay_ts < cut_ts)
        delay = np.reshape(cut_ts - self.click_ts, (-1, 1))/SECONDS_DELAY_NORM
        labels = np.concatenate([np.reshape(label, (-1, 1)), delay], axis=1)
        return DataDF(self.x,
                      self.click_ts,
                      self.pay_ts,
                      self.sample_ts,
                      labels)

    def shuffle(self):
        idx = list(range(self.x.shape[0]))
        np.random.shuffle(idx)
        return DataDF(self.x.iloc[idx],
                      self.click_ts[idx],
                      self.pay_ts[idx],
                      self.sample_ts[idx],
                      self.labels[idx])


def get_criteo_dataset_stream(params, pre_trained_model=None):
    print 'get_criteo_dataset_stream in'
    name = params["dataset"]
    print("loading datasest {}".format(name))
    cache_path = os.path.join(
        params["data_cache_path"], "{}.pkl".format(name))
    if params["data_cache_path"] != "None" and os.path.isfile(cache_path):
        print("cache_path {}".format(cache_path))
        print("\nloading from dataset cache")
        with open(cache_path, "rb") as f:
            data = pickle.load(f)
        train_stream = data["train"]
        test_stream = data["test"]
        test_stream_nowin = data["nowin"]
    else:
        print("\ncan't load from cache, building dataset")
        df, click_ts, pay_ts = get_data_df(params)
        if name == "last_30_train_test_oracle":
            data = DataDF(df, click_ts, pay_ts)
            train_data = data.sub_days(30, 60)
            test_data = data.sub_days(30, 60)
            train_stream = []
            test_stream = []
            #print 'train_hour.labels'
            #print train_hour.labels
            #print 'test_hour.labels'
            #print test_hour.labels

            for tr in range(30*24, 59*24+23):
                train_hour = train_data.sub_hours(tr, tr+1)
                train_stream.append({"x": train_hour.x,
                                     "click_ts": train_hour.click_ts,
                                     "pay_ts": train_hour.pay_ts,
                                     "sample_ts": train_hour.sample_ts,
                                     "labels": train_hour.labels})
            for tr in range(30*24+1, 60*24):
                test_hour = test_data.sub_hours(tr, tr+1)
                test_stream.append({"x": test_hour.x,
                                    "click_ts": test_hour.click_ts,
                                    "pay_ts": test_hour.pay_ts,
                                    "sample_ts": test_hour.sample_ts,
                                    "labels": test_hour.labels})

            test_data_nowin = data.sub_days(30, 60).add_nowin_auc()
            test_stream_nowin = []

            for tr in range(30*24+1, 60*24):
                test_hour = test_data_nowin.sub_hours(tr, tr+1)
                test_stream_nowin.append({"x": test_hour.x,
                                    "click_ts": test_hour.click_ts,
                                    "pay_ts": test_hour.pay_ts,
                                    "sample_ts": test_hour.sample_ts,
                                    "labels": test_hour.labels})

            
        elif name == "last_30_train_test_fsiw":
            data = DataDF(df, click_ts, pay_ts)
            train_data = data.sub_days(30, 60)
            test_data = data.sub_days(30, 60)
            train_stream = []
            test_stream = []
            #print 'train_hour.labels'
            #print train_hour.labels
            #print 'test_hour.labels'
            #print test_hour.labels

            for tr in range(30*24, 59*24+23):
                cut_ts = (tr+1)*SECONDS_AN_HOUR
                train_hour = train_data.sub_hours(
                    tr, tr+1).to_fsiw_tune(cut_ts)
                train_stream.append({"x": train_hour.x,
                                     "click_ts": train_hour.click_ts,
                                     "pay_ts": train_hour.pay_ts,
                                     "sample_ts": train_hour.sample_ts,
                                     "labels": train_hour.labels})
            for tr in range(30*24+1, 60*24):
                test_hour = test_data.sub_hours(tr, tr+1)
                test_stream.append({"x": test_hour.x,
                                    "click_ts": test_hour.click_ts,
                                    "pay_ts": test_hour.pay_ts,
                                    "sample_ts": test_hour.sample_ts,
                                    "labels": test_hour.labels})
        elif name == "last_30_train_test_dfm":
            data = DataDF(df, click_ts, pay_ts)
            train_data = data.sub_days(30, 60)
            test_data = data.sub_days(30, 60)
            train_stream = []
            test_stream = []
            #print 'train_hour.labels'
            #print train_hour.labels
            #print 'test_hour.labels'
            #print test_hour.labels

            for tr in range(30*24, 59*24+23):
                cut_ts = (tr+1)*SECONDS_AN_HOUR
                train_hour = train_data.sub_hours(tr, tr+1).to_dfm_tune(cut_ts)
                train_stream.append({"x": train_hour.x,
                                     "click_ts": train_hour.click_ts,
                                     "pay_ts": train_hour.pay_ts,
                                     "sample_ts": train_hour.sample_ts,
                                     "labels": train_hour.labels})
            for tr in range(30*24+1, 60*24):
                test_hour = test_data.sub_hours(tr, tr+1)
                test_stream.append({"x": test_hour.x,
                                    "click_ts": test_hour.click_ts,
                                    "pay_ts": test_hour.pay_ts,
                                    "sample_ts": test_hour.sample_ts,
                                    "labels": test_hour.labels})
            test_data_nowin = data.sub_days(30, 60).add_nowin_auc()
            test_stream_nowin = []

            for tr in range(30*24+1, 60*24):
                test_hour = test_data_nowin.sub_hours(tr, tr+1)
                test_stream_nowin.append({"x": test_hour.x,
                                    "click_ts": test_hour.click_ts,
                                    "pay_ts": test_hour.pay_ts,
                                    "sample_ts": test_hour.sample_ts,
                                    "labels": test_hour.labels})

        elif "last_30_train_test_esdfm" in name:
            cut_hour = parse_float_arg(name, "cut_hour")
            print("cut_hour {}".format(cut_hour))
            cut_sec = cut_hour*SECONDS_AN_HOUR
            data = DataDF(df, click_ts, pay_ts)
            if params["method"] == "ES-DFM10" or params["method"] == "Vanilla":
                print 'last_30_train_test_esdfm 10'
                train_data = data.sub_days(0, 60).add_esdfm_cut_fake_neg2(cut_sec,params)
            elif params["method"] == "ES-DFM-FULL":
                print 'last_30_train_test_esdfm full'
                train_data = data.sub_days(0, 60).add_esdfm_cut_fake_neg3(cut_sec)
            else:
                print 'last_30_train_test_esdfm'
                train_data = data.sub_days(0, 60).add_esdfm_cut_fake_neg(cut_sec)
            test_data = data.sub_days(30, 60)

            train_stream = []
            test_stream = []
            for tr in range(30*24, 59*24+23):
                train_hour = train_data.sub_hours(tr, tr+1)
                train_stream.append({"x": train_hour.x,
                                     "click_ts": train_hour.click_ts,
                                     "pay_ts": train_hour.pay_ts,
                                     "sample_ts": train_hour.sample_ts,
                                     "labels": train_hour.labels})
                #print 'train_hour.labels'
                #print train_hour.labels
            for tr in range(30*24+1, 60*24):
                test_hour = test_data.sub_hours(tr, tr+1)
                test_stream.append({"x": test_hour.x,
                                    "click_ts": test_hour.click_ts,
                                    "pay_ts": test_hour.pay_ts,
                                    "sample_ts": test_hour.sample_ts,
                                    "labels": test_hour.labels})
                #print 'test_hour.labels'
                #print test_hour.labels
            test_data_nowin = data.sub_days(30, 60).add_nowin_auc()
            test_stream_nowin = []

            for tr in range(30*24+1, 60*24):
                test_hour = test_data_nowin.sub_hours(tr, tr+1)
                test_stream_nowin.append({"x": test_hour.x,
                                    "click_ts": test_hour.click_ts,
                                    "pay_ts": test_hour.pay_ts,
                                    "sample_ts": test_hour.sample_ts,
                                    "labels": test_hour.labels})


        elif "last_30_train_test_fnw" in name:
            cut_sec = 0.
            if "cut_hour" in name:
                cut_hour = parse_float_arg(name, "cut_hour")
                print("cut_hour {}".format(cut_hour))
                cut_sec = cut_hour*SECONDS_AN_HOUR

            data = DataDF(df, click_ts, pay_ts)

            start = int(params["start"])
            mid = int(params["mid"])
            end = int(params["end"])

            print 'start mid end'
            print start, mid, end

            if params["method"] == "FNC10" or params["method"] == "FNW10":
                print 'last_30_train_test_fnw 10'
                train_data = data.sub_days(start, end).add_fake_neg2(params) #add_fake_neg() #.add_fake_neg2
            elif params["method"] == "Vanilla":
                print 'last_30_train_test_fnw odl'
                train_data = data.sub_days(start, end).add_fake_neg3(cut_sec,params)
            else:
                print 'last_30_train_test_fnw'
                train_data = data.sub_days(start, end).add_fake_neg() #add_fake_neg() #.add_fake_neg2
            test_data = data.sub_days(mid, end)
            #test_data_nowin = data.sub_days(30, 60).add_nowin_auc()
            train_stream = []
            test_stream = []
            #test_stream_nowin = []
            #print 'train_hour.labels'
            #print train_hour.labels
            print 'test_hour.labels'
            #print test_hour.labels

            for tr in range(mid*24, (end-1)*24+23):
                train_hour = train_data.sub_hours(tr, tr+1)
                #print train_hour.labels
                if train_hour.x.shape[0] <= MIN_SAMPLE_PER_HOUR:continue
                train_stream.append({"x": train_hour.x,
                                     "click_ts": train_hour.click_ts,
                                     "pay_ts": train_hour.pay_ts,
                                     "sample_ts": train_hour.sample_ts,
                                     "labels": train_hour.labels})
            for tr in range(mid*24+1, end*24):
                test_hour = test_data.sub_hours(tr, tr+1)
                if test_hour.x.shape[0] <= MIN_SAMPLE_PER_HOUR:continue
                test_stream.append({"x": test_hour.x,
                                    "click_ts": test_hour.click_ts,
                                    "pay_ts": test_hour.pay_ts,
                                    "sample_ts": test_hour.sample_ts,
                                    "labels": test_hour.labels})


            test_data_nowin = data.sub_days(mid, end).add_nowin_auc()
            test_stream_nowin = []

            for tr in range(mid*24+1, end*24):
                test_hour = test_data_nowin.sub_hours(tr, tr+1)
                if test_hour.x.shape[0] <= MIN_SAMPLE_PER_HOUR:continue
                test_stream_nowin.append({"x": test_hour.x,
                                    "click_ts": test_hour.click_ts,
                                    "pay_ts": test_hour.pay_ts,
                                    "sample_ts": test_hour.sample_ts,
                                    "labels": test_hour.labels})

        elif "last_30_train_test_3class" in name: 

            cut_hour = parse_float_arg(name, "cut_hour")
            print("cut_hour {}".format(cut_hour))
            cut_sec = cut_hour*SECONDS_AN_HOUR
            data = DataDF(df, click_ts, pay_ts)
            train_data = data.sub_days(0, 60).add_3class_cut_fake_neg(cut_sec)
            test_data = data.sub_days(30, 60)

            train_stream = []
            test_stream = []
            for tr in range(30*24, 59*24+23):
                train_hour = train_data.sub_hours(tr, tr+1)
                train_stream.append({"x": train_hour.x,
                                     "click_ts": train_hour.click_ts,
                                     "pay_ts": train_hour.pay_ts,
                                     "sample_ts": train_hour.sample_ts,
                                     "labels": train_hour.labels})
                #print 'train_hour.labels'
                #print train_hour.labels
            for tr in range(30*24+1, 60*24):
                test_hour = test_data.sub_hours(tr, tr+1)
                test_stream.append({"x": test_hour.x,
                                    "click_ts": test_hour.click_ts,
                                    "pay_ts": test_hour.pay_ts,
                                    "sample_ts": test_hour.sample_ts,
                                    "labels": test_hour.labels})
                #print 'test_hour.labels'
                #print test_hour.labels
        elif "last_30_train_test_likeli" in name:

            cut_hour = parse_float_arg(name, "cut_hour")
            print("cut_hour {}".format(cut_hour))
            cut_sec = cut_hour*SECONDS_AN_HOUR
            data = DataDF(df, click_ts, pay_ts)

            start = int(params["start"])
            mid = int(params["mid"])
            end = int(params["end"])

            print 'start mid end'
            print start, mid, end

            train_data = data.sub_days(start, end).add_3class_cut_fake_neg(cut_sec)
            test_data = data.sub_days(mid, end)

            train_stream = []
            test_stream = []
            for tr in range(mid*24, (end-1)*24+23):
                train_hour = train_data.sub_hours(tr, tr+1)
                train_stream.append({"x": train_hour.x,
                                     "click_ts": train_hour.click_ts,
                                     "pay_ts": train_hour.pay_ts,
                                     "sample_ts": train_hour.sample_ts,
                                     "labels": train_hour.labels})
                #print 'train_hour.labels'
                #print train_hour.labels
            for tr in range(mid*24+1, end*24):
                test_hour = test_data.sub_hours(tr, tr+1)
                test_stream.append({"x": test_hour.x,
                                    "click_ts": test_hour.click_ts,
                                    "pay_ts": test_hour.pay_ts,
                                    "sample_ts": test_hour.sample_ts,
                                    "labels": test_hour.labels})

            test_data_nowin = data.sub_days(30, 60).add_nowin_auc()
            test_stream_nowin = []

            for tr in range(30*24+1, 60*24):
                test_hour = test_data_nowin.sub_hours(tr, tr+1)
                test_stream_nowin.append({"x": test_hour.x,
                                    "click_ts": test_hour.click_ts,
                                    "pay_ts": test_hour.pay_ts,
                                    "sample_ts": test_hour.sample_ts,
                                    "labels": test_hour.labels})



        elif "last_30_train_test_delay_win_time" in name:

            cut_hour = parse_float_arg(name, "cut_hour")
            print("cut_hour {}".format(cut_hour))
            cut_sec = cut_hour*SECONDS_AN_HOUR
            data = DataDF(df, click_ts, pay_ts)

            start = int(params["start"])
            mid = int(params["mid"])
            end = int(params["end"])

            print 'start mid end'
            print start, mid, end

            train_data = data.sub_days(start, end).add_delay_wintime_cut_fake_neg(cut_sec)
            #train_data = data.sub_days(start, end).add_delay_wintime_cut_fake_neg2(cut_sec)
            test_data = data.sub_days(mid, end)

            train_stream = []
            test_stream = []
            for tr in range(mid*24, (end-1)*24+23):
                train_hour = train_data.sub_hours(tr, tr+1)
                train_stream.append({"x": train_hour.x,
                                     "click_ts": train_hour.click_ts,
                                     "pay_ts": train_hour.pay_ts,
                                     "sample_ts": train_hour.sample_ts,
                                     "labels": train_hour.labels})
                #print 'train_hour.labels'
                #print train_hour.labels
            for tr in range(mid*24+1, end*24):
                test_hour = test_data.sub_hours(tr, tr+1)
                test_stream.append({"x": test_hour.x,
                                    "click_ts": test_hour.click_ts,
                                    "pay_ts": test_hour.pay_ts,
                                    "sample_ts": test_hour.sample_ts,
                                    "labels": test_hour.labels})

            test_data_nowin = data.sub_days(30, 60).add_nowin_auc()
            test_stream_nowin = []

            for tr in range(30*24+1, 60*24):
                test_hour = test_data_nowin.sub_hours(tr, tr+1)
                test_stream_nowin.append({"x": test_hour.x,
                                    "click_ts": test_hour.click_ts,
                                    "pay_ts": test_hour.pay_ts,
                                    "sample_ts": test_hour.sample_ts,
                                    "labels": test_hour.labels})


        elif "last_30_train_test_delay_win_adapt" in name:

            cut_hour = parse_float_arg(name, "cut_hour")
            print("cut_hour {}".format(cut_hour))
            cut_sec = cut_hour*SECONDS_AN_HOUR
            data = DataDF(df, click_ts, pay_ts)
            begin = 0
            mid = 30
            end = 60

            #'''
            train_data = data.sub_days(0, end)

            train_dataset_time = {
            "x": train_data.x,
            "click_ts": train_data.click_ts,
            "pay_ts": train_data.pay_ts,
            "sample_ts": train_data.sample_ts,
            "labels": train_data.labels,
            }
            
            train_data_time = tf.data.Dataset.from_tensor_slices(
                (dict(train_dataset_time["x"]), train_dataset_time["labels"]))
            train_data_time = train_data_time.batch(params["batch_size"]).prefetch(1)
            model_time = pre_trained_model #get_model(params["model"], params)
            time_cut = pretrain.test(model_time, train_data_time, params, before=True)
            print 'time_cut.shape0'
            print time_cut.shape
            print time_cut

            time_cut_15 = time_cut[:,1]
            time_cut_30 = (1-time_cut[:,1])*(time_cut[:,2])
            time_cut_60 = (1-time_cut[:,1])*(1-time_cut[:,2])*time_cut[:,3]

            #time_cut_15 = np.array(time_cut_15)
            #time_cut_30 = np.array(time_cut_30#)
            #time_cut_60 = np.array(time_cut_60)

            time_cut_new = np.concatenate([time_cut_15.reshape(-1,1), time_cut_30.reshape(-1,1), time_cut_60.reshape(-1,1)], axis=1)
            print 'time_cut_new'
            print time_cut_new.shape
            print time_cut_new

            time_cut_arg = np.argmax(time_cut_new, axis=1) #.reshape(-1,1)
            print 'time_cut_arg'
            print time_cut_arg.shape
            print time_cut_arg

            win1 = params["win1"] #0.25
            win2 = params["win2"] #0.5 
            win3 = params["win3"]
            
            print 'win 1 2 3 new'
            print win1, win2, win3

            time_win = np.where(time_cut_arg == 0, win1, time_cut_arg)
            time_win = np.where(time_win == 1, win2, time_win)
            time_win = np.where(time_win == 2, win3, time_win)
            #time_cut_final = np.concatenate(
            #    [time_win, time_cut_arg], axis=1)
            print 'time_win.shape1'
            print time_win.shape
            print np.sum(time_win)
            print time_win

            mask = time_win==win1
            print 'time_win 0.25 .shape'
            print time_win[mask].shape
            print time_win[mask]

            mask = time_win==win2
            print 'time_win 0.5 .shape'
            print time_win[mask].shape
            print time_win[mask]

            mask = time_win==win3
            print 'time_win 1 .shape'
            print time_win[mask].shape
            print time_win[mask]

            data = DataDF(df, click_ts, pay_ts,time_cut=time_win)
            train_data = data.sub_days(0, 60).add_delay_winadapt2_cut_fake_neg(cut_sec, time_win, params)
            test_data = data.sub_days(mid, 60)

            train_stream = []
            test_stream = []
            for tr in range(mid*24, (end-1)*24+23):
                train_hour = train_data.sub_hours(tr, tr+1)
                train_stream.append({"x": train_hour.x,
                                     "click_ts": train_hour.click_ts,
                                     "pay_ts": train_hour.pay_ts,
                                     "sample_ts": train_hour.sample_ts,
                                     "labels": train_hour.labels})
                #print 'train_hour.labels'
                #print train_hour.labels
            for tr in range(mid*24+1, end*24):
                test_hour = test_data.sub_hours(tr, tr+1)
                test_stream.append({"x": test_hour.x,
                                    "click_ts": test_hour.click_ts,
                                    "pay_ts": test_hour.pay_ts,
                                    "sample_ts": test_hour.sample_ts,
                                    "labels": test_hour.labels})

            test_data_nowin = data.sub_days(30, 60).add_nowin_auc()
            test_stream_nowin = []

            for tr in range(30*24+1, 60*24):
                test_hour = test_data_nowin.sub_hours(tr, tr+1)
                test_stream_nowin.append({"x": test_hour.x,
                                    "click_ts": test_hour.click_ts,
                                    "pay_ts": test_hour.pay_ts,
                                    "sample_ts": test_hour.sample_ts,
                                    "labels": test_hour.labels})

 
        else:
            raise NotImplementedError("{} data does not exist".format(name))
    if params["data_cache_path"] != "None":
        with open(cache_path, "wb") as f:
            pickle.dump({"train": train_stream, "test": test_stream, "nowin": test_stream_nowin}, f)
            f.close() 
    return train_stream, test_stream, test_stream_nowin


def get_criteo_dataset(params):
    name = params["dataset"]
    print("loading datasest {}".format(name))
    cache_path = os.path.join(
        params["data_cache_path"], "{}.pkl".format(name))
    if params["data_cache_path"] != "None" and os.path.isfile(cache_path):
        print("cache_path {}".format(cache_path))
        print("\nloading from dataset cache")
        with open(cache_path, "rb") as f:
            data = pickle.load(f)
        train_data = data["train"]
        test_data = data["test"]
    else:
        print("\nbuilding dataset")
        df, click_ts, pay_ts = get_data_df(params)
        data = DataDF(df, click_ts, pay_ts)
        if name == "baseline_prtrain":
            trainstart = int(params["pre_trainstart"])
            trainend = int(params["pre_trainend"])
            teststart = int(params["pre_teststart"])
            testend = int(params["pre_testend"])

            delfu = int(params["delfu"])
            print 'start,end,',trainstart, trainend, teststart, testend
            print 'delfu:',delfu
            if delfu == 0:
                print 'delfu:',delfu
                train_data = data.sub_days(trainstart, trainend).shuffle()
            else:
                print 'delfu:',delfu
                #train_data = data.sub_days(trainstart, trainend).del_future_label(trainstart, trainend).shuffle()
                train_data = data.sub_days(trainstart, trainend).shuffle().del_future_label(trainstart, trainend)
            print 'train_data.labels'
            print train_data.labels
            print np.sum(train_data.labels, axis = 0), train_data.labels.shape[0]

            mask = train_data.pay_ts < 0
            train_data.pay_ts[mask] = trainend * \
                SECONDS_A_DAY + train_data.click_ts[mask]
            test_data = data.sub_days(teststart, testend)
        elif name == "dfm_prtrain":
            trainstart = int(params["pre_trainstart"])
            trainend = int(params["pre_trainend"])
            teststart = int(params["pre_teststart"])
            testend = int(params["pre_testend"])

            print 'start mid end'
            print trainstart, trainend, teststart, testend

            delfu = int(params["delfu"])
            print 'start,end,',trainstart, trainend, teststart, testend
            print 'delfu:',delfu
            if delfu == 0:
                print 'delfu:',delfu
                train_data = data.sub_days(trainstart, trainend).shuffle()
            else:
                print 'delfu:',delfu
                train_data = data.sub_days(trainstart, trainend).del_future_label(trainstart, trainend).shuffle()
            print 'train_data.labels'
            print train_data.labels
            print np.sum(train_data.labels, axis = 0), train_data.labels.shape[0]

            #train_data = data.sub_days(0, 30).shuffle()
            train_data.pay_ts[train_data.pay_ts < 0] = SECONDS_A_DAY*30
            delay = np.reshape(train_data.pay_ts -
                               train_data.click_ts, (-1, 1))/SECONDS_DELAY_NORM
            train_data.labels = np.reshape(train_data.labels, (-1, 1))
            train_data.labels = np.concatenate(
                [train_data.labels, delay], axis=1)
            test_data = data.sub_days(30, 60)
        elif "tn_dp_pretrain" in name:
            trainstart = int(params["pre_trainstart"])
            trainend = int(params["pre_trainend"])
            teststart = int(params["pre_teststart"])
            testend = int(params["pre_testend"])

            print 'start mid end'
            print trainstart, trainend, teststart, testend

            cut_hour = parse_float_arg(name, "cut_hour")
            cut_sec = int(SECONDS_AN_HOUR*cut_hour)
            #train_data = data.sub_days(trainstart, trainend).shuffle()
            delfu = int(params["delfu"])
            print 'start,end,',trainstart, trainend, teststart, testend
            print 'delfu:',delfu
            if delfu == 0:
                print 'delfu:',delfu
                train_data = data.sub_days(trainstart, trainend).shuffle()
            else:
                print 'delfu:',delfu
                train_data = data.sub_days(trainstart, trainend).del_future_label(trainstart, trainend).shuffle()
            print 'train_data.labels'
            print train_data.labels
            print np.sum(train_data.labels, axis = 0), train_data.labels.shape[0]

            train_label_tn = np.reshape(train_data.pay_ts < 0, (-1, 1))
            train_label_dp = np.reshape(
                train_data.pay_ts - train_data.click_ts > cut_sec, (-1, 1))
            train_label = np.reshape(
		train_data.pay_ts > 0, (-1, 1))
            train_data.labels = np.concatenate(
                [train_label_tn, train_label_dp, train_label], axis=1)
            test_data = data.sub_days(teststart, testend)
            test_label_tn = np.reshape(test_data.pay_ts < 0, (-1, 1))
            test_label_dp = np.reshape(
                test_data.pay_ts - test_data.click_ts > cut_sec, (-1, 1))
            test_label = np.reshape(
		test_data.pay_ts > 0, (-1, 1))
            test_data.labels = np.concatenate(
                [test_label_tn, test_label_dp, test_label], axis=1)
        elif "likeli_pretrain" in name:
            cut_hour = parse_float_arg(name, "cut_hour")
            cut_sec = int(SECONDS_AN_HOUR*cut_hour)
            #train_data = data.sub_days(0, 30).shuffle()

            trainstart = int(params["pre_trainstart"])
            trainend = int(params["pre_trainend"])
            teststart = int(params["pre_teststart"])
            testend = int(params["pre_testend"])

            delfu = int(params["delfu"])
            print 'start,end,',trainstart, trainend, teststart, testend

            cv_mask_day = int(params["cv_mask_day"])
            end_ts = (trainend-cv_mask_day)*SECONDS_A_DAY
            #ddline = int(params["ddline"])*SECONDS_A_DAY
            #print 'ddline'
            #print ddline
            if delfu == 0:
                print 'delfu:',delfu
                train_data = data.sub_days(trainstart, trainend).shuffle()
            else:
                print 'delfu:',delfu
                train_data = data.sub_days(trainstart, trainend).shuffle().del_future_label(trainstart, trainend)


            #train_data = data.sub_days(trainstart, trainend).del_future_label(trainstart, trainend).shuffle()
            train_label_cv = np.reshape(
			train_data.pay_ts > 0, (-1, 1))
            train_label_win = np.reshape(np.logical_and((train_data.pay_ts - train_data.click_ts <= cut_sec), (train_data.pay_ts > 0)), (-1, 1))

            if delfu == 0:
                train_label_cv_mask = np.reshape(np.ones(train_data.click_ts.shape), (-1, 1))
            else:
                train_label_cv_mask = np.reshape(np.logical_or(train_data.click_ts <= end_ts,train_data.pay_ts > 0), (-1, 1))

            print 'train_label_cv_mask:', np.sum(train_label_cv_mask, axis=0), train_label_cv_mask.shape[0]
            #train_label_win = np.reshape(
            #    train_data.pay_ts - train_data.click_ts <= cut_sec, (-1, 1))
            #train_label = np.reshape(train_data.pay_ts > 0, (-1, 1))
            train_data.labels = np.concatenate(
                [train_label_cv, train_label_win, train_label_cv_mask], axis=1)
            print 'train_data.labels'
            print train_data.labels
            print np.sum(train_data.labels, axis = 0), train_data.labels.shape[0]

            #test_data = data.sub_days(30, 60)
            test_data = data.sub_days(teststart, testend)
            test_label_cv = np.reshape(
  			test_data.pay_ts > 0, (-1, 1))
            test_label_win = np.reshape(np.logical_and((test_data.pay_ts - test_data.click_ts <= cut_sec), (test_data.pay_ts > 0)), (-1, 1))
            #test_label_win = np.reshape(
            #    test_data.pay_ts - test_data.click_ts <= cut_sec, (-1, 1))
            #test_label = np.reshape(test_data.pay_ts > 0, (-1, 1))
            test_data.labels = np.concatenate(
                [test_label_cv, test_label_win], axis=1)
            print "test_data.labels"
            print test_data.labels
            print np.sum(test_data.labels, axis = 0)

        elif "fsiw1" in name:
            cd = parse_float_arg(name, "cd")
            print("cd {}".format(cd))
            train_data = data.sub_days(0, 30).shuffle()
            test_data = data.sub_days(30, 60)
            train_data = train_data.to_fsiw_1(
                cd=cd*SECONDS_A_DAY, T=30*SECONDS_A_DAY)
            test_data = test_data.to_fsiw_1(
                cd=cd*SECONDS_A_DAY, T=60*SECONDS_A_DAY)
        elif "fsiw0" in name:
            cd = parse_float_arg(name, "cd")
            train_data = data.sub_days(0, 30).shuffle()
            test_data = data.sub_days(30, 60)
            train_data = train_data.to_fsiw_0(
                cd=cd*SECONDS_A_DAY, T=30*SECONDS_A_DAY)
            test_data = test_data.to_fsiw_0(
                cd=cd*SECONDS_A_DAY, T=60*SECONDS_A_DAY)
        elif "3class" in name:
            cut_hour = parse_float_arg(name, "cut_hour")
            print("ch {}".format(cut_hour))
            cut_sec = int(SECONDS_AN_HOUR * cut_hour)
            train_data = data.sub_days(0, 30).shuffle()
            train_label_00 = np.reshape(train_data.pay_ts < 0, (-1, 1))
            train_label_01 = np.reshape(train_data.pay_ts - train_data.click_ts > cut_sec, (-1, 1))
            train_label_11 = np.reshape(np.logical_and((train_data.pay_ts - train_data.click_ts <= cut_sec), (train_data.pay_ts > 0)), (-1, 1))
            train_data.labels = np.concatenate(
                [train_label_00, train_label_01, train_label_11], axis=1)
            test_data = data.sub_days(30, 60)
            test_label_00 = np.reshape(test_data.pay_ts < 0, (-1, 1))
            test_label_01 = np.reshape(test_data.pay_ts - test_data.click_ts > cut_sec, (-1, 1))
            test_label_11 = np.reshape(np.logical_and((test_data.pay_ts - test_data.click_ts <= cut_sec), (test_data.pay_ts > 0)), (-1, 1))
            test_data.labels = np.concatenate(
                [test_label_00, test_label_01, test_label_11], axis=1)
            temp = train_label_00 + train_label_01 + train_label_11
            temp2 = test_label_00 + test_label_01 + test_label_11
            print "temp"
            print temp
            print "temp2"
            print temp2
            print "train_data.labels" 
            print train_data.labels
            temp3 = np.sum(train_data.labels,axis=0)
            print "temp3"
            print temp3

        elif "win_adapt" in name:
            cut_hour1 = 0.25
            cut_hour2 = 0.5
            cut_hour3 = 1
            print("ch {}".format(cut_hour1))
            cut_sec1 = int(SECONDS_AN_HOUR * cut_hour1)
            cut_sec2 = int(SECONDS_AN_HOUR * cut_hour2)
            cut_sec3 = int(SECONDS_AN_HOUR * cut_hour3)

            train_data = data.sub_days(0, 30).shuffle()
            #train_label_00 = np.reshape(train_data.pay_ts < 0, (-1, 1))
            train_label_15 = np.reshape(np.logical_and((train_data.pay_ts - train_data.click_ts <= cut_sec1), train_data.pay_ts > 0), (-1, 1))
            train_label_30 = np.reshape(np.logical_and((train_data.pay_ts - train_data.click_ts <= cut_sec2), (train_data.pay_ts - train_data.click_ts > cut_sec1)), (-1, 1))
            train_label_60 = np.reshape(np.logical_and((train_data.pay_ts - train_data.click_ts <= cut_sec3), (train_data.pay_ts - train_data.click_ts > cut_sec2)), (-1, 1))
            train_label_11 = np.reshape(train_data.pay_ts > 0, (-1, 1))
            train_data.labels = np.concatenate(
                [train_label_11, train_label_15, train_label_30, train_label_60], axis=1)
            print 'train_data.labels'
            print train_data.labels
            print np.sum(train_data.labels, axis = 0)
            test_data = data.sub_days(30, 60)
            #train_label_00 = np.reshape(test_data.pay_ts < 0, (-1, 1))
            test_label_15 = np.reshape(np.logical_and((test_data.pay_ts - test_data.click_ts <= cut_sec1), test_data.pay_ts > 0), (-1, 1))
            test_label_30 = np.reshape(np.logical_and((test_data.pay_ts - test_data.click_ts <= cut_sec2), (test_data.pay_ts - test_data.click_ts > cut_sec1)), (-1, 1))
            test_label_60 = np.reshape(np.logical_and((test_data.pay_ts - test_data.click_ts <= cut_sec3), (test_data.pay_ts - test_data.click_ts > cut_sec2)), (-1, 1))
            test_label_11 = np.reshape(test_data.pay_ts > 0, (-1, 1))
            test_data.labels = np.concatenate(
                [test_label_11, test_label_15, test_label_30, test_label_60], axis=1)
            print "test_data.labels"
            print test_data.labels
            print np.sum(test_data.labels, axis = 0)

        else:
            raise NotImplementedError("{} dataset does not exist".format(name))
    if params["data_cache_path"] != "None":
        with open(cache_path, "wb") as f:
            pickle.dump({"train": train_data, "test": test_data}, f)
            f.close()
    return {
        "train": {
            "x": train_data.x,
            "click_ts": train_data.click_ts,
            "pay_ts": train_data.pay_ts,
            "sample_ts": train_data.sample_ts,
            "labels": train_data.labels,
        },
        "test": {
            "x": test_data.x,
            "click_ts": test_data.click_ts,
            "pay_ts": test_data.pay_ts,
            "sample_ts": train_data.sample_ts,
            "labels": test_data.labels,
        }
    }
