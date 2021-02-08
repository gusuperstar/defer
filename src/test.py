import pathlib
import tensorflow as tf
from tensorflow import feature_column
import numpy as np
'''
num_bin_size = (64, 16, 128, 64, 128, 64, 512, 512)
cate_bin_size = (512, 128, 256, 256, 64, 256, 256, 16, 256)
step_val = 1.0 / 2
boundaries = list(np.arange(0, 3, step_val)) 

print boundaries
num_features = [feature_column.bucketized_column(
            feature_column.numeric_column(str(i)),
            boundaries=[j*1.0/(num_bin_size[i]-1) for j in range(num_bin_size[i]-1)])
            for i in range(8)]


for i in range(8):
	boundaries=[j*1.0/(num_bin_size[i]-1) for j in range(num_bin_size[i]-1)]
	print boundaries
'''
#data_cache_path='temp'
#pathlib.Path(data_cache_path).mkdir(parents=True)

import numpy as np


def ece_score(y_test, py, n_bins=10):
    py = np.array(py).reshape(-1)
    y_test = np.array(y_test).reshape(-1)

    temp = []
    for i in range(py.shape[0]):
        temp.append((py[i],y_test[i]))

    pairs = temp #= [(1, 'one'), (2, 'two'), (3, 'three'), (4, 'four')]
    pairs.sort(key=lambda pair: pair[0])

    print y_test.shape
    pair0=[]
    pair1=[]
    for (it0, it1) in pairs:
        pair0.append(it0)
        pair1.append(it1)

    py = np.array(pair0).reshape(-1)
    y_test = np.array(pair1).reshape(-1)

    py_value = py #np.array(py_value)
    print py_value
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
            print 'acc conf 1 ', a, b
            print Bm[m],acc[m], conf[m]
            acc[m] = acc[m] / Bm[m]
            conf[m] = conf[m] / Bm[m]
    ece = 0
    for m in range(n_bins):
        ece += Bm[m] * np.abs((acc[m] - conf[m]))
    return ece / sum(Bm)


a=[[0.11399972, 0.11096141, 0.20475818, 0.26466718],
 [0.17380534, 0.09904043, 0.18008941, 0.23670368]]
npa= np.array(a)
print npa[:,1:4]
print npa[:,1].reshape(-1,1)
print npa[:,2].reshape(-1,1)
print npa[:,3].reshape(-1,1)
print np.argmax(npa[:,1:4], axis=1)


a=[[0.11399972, 0.11096141, 0.20475818, 0.26466718],
[0.11399972, 0.11096141, 0.20475818, 0.26466718],
 [0.17380534, 0.09904043, 0.38008941, 0.23670368]]
time_cut = np.array(a)

time_cut_15 = time_cut[:,1]
time_cut_30 = (1-time_cut[:,1])*(time_cut[:,2])
time_cut_60 = (1-time_cut[:,1])*(1-time_cut[:,2])*time_cut[:,3]

#time_cut_15 = np.array(time_cut_15)
#time_cut_30 = np.array(time_cut_30#)
#time_cut_60 = np.array(time_cut_60)

time_cut_new = np.concatenate([time_cut_15.reshape(-1,1), time_cut_30.reshape(-1,1), time_cut_60.reshape(-1,1)], axis=1)

time_cut_arg = np.argmax(time_cut_new, axis=1)#.reshape(-1,1)
print 'time_cut_arg'
print time_cut_arg.shape
print time_cut_arg

time_win = np.where(time_cut_arg == 0, 0.25, time_cut_arg)

time_win = np.where(time_win == 1, 0.5, time_win)
time_win = np.where(time_win == 2, 1, time_win)
#time_cut_final = np.concatenate(
#    [time_win, time_cut_arg], axis=1)
mask = time_win==0.5
print 'time_win.shape1'
print time_win[mask].shape
print time_win[mask]

print 'time_win.shape2'
print time_win.shape
print np.sum(time_win)
print time_win


a=[0.11399972, 0.11096141, 0.20475818, 0.26466718, 0.11399972, 0.11096141, 0.20475818, 0.26466718, 0.11399972, 0.11096141, 0.20475818, 0.11399972, 0.11096141, 0.20475818]
npa= np.array(a).reshape(-1,1)
#a = np.concatenate([npa,1-npa], axis=1)
b=[1, 1, 0, 1, 0, 1,0, 0, 0, 1, 0, 0, 1, 0]

npb= np.array(b).reshape(-1,1)
#b = np.concatenate([npb,1-npb], axis=1)
print ece_score(npb,npa,3)

