#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 10:50:46 2018

@author: kaku
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os, random
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD

from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

import models 

def get_img_and_label(path, half_size, stride, amount, label, vert = True, hori = True, aug = False ):
    """
    Get small patches and related labels from big images
    Method is sliding window, only up and down
    Parameters
    ----------
    path: str
    half_size: int
        image half size
    stride: int 
        sliding window stride
    amount: int
        half amount
    label: float64
        obtain by csv file
    vert, hori: bool
        vertically and horizontally sliding window
    Returns
    -------
    imgs_all: list
        [[],[],...]: big_image_amount -> small_image_amount amount*2-1 -> small_image  
    labels_all :list
        [[],[],...]: big_image_amount -> small_image_amount amount*2-1 -> small_image_label
    """
    
    img_names = os.listdir(img_path)
    imgs_all = []
    labels_all = []
    for img_name in img_names:
        img_ori = plt.imread(os.path.join(img_path, img_name))
        h, w, c = img_ori.shape
        imgs = []
        labels = []
        if vert:
        #size.append([h, w, c])
            for i in range(amount):
                # move downword
                img_small = img_ori[(h//2-img_size-amount*stride):(h//2+img_size-amount*stride),(w//2-img_size):(w//2+img_size),:]
                imgs.append(img_small)
                labels.append(label[img_names.index(img_name)])
                if aug:
                    if random.uniform(0, 1) > 0.5:
                        img_aug = img_small[::-1]
                        imgs.append(img_aug)
                        labels.append(label[img_names.index(img_name)])
            for i in range(1,amount):
                # move upword
                img_small = img_ori[(h//2-img_size+amount*stride):(h//2+img_size+amount*stride),(w//2-img_size):(w//2+img_size),:]
                imgs.append(img_small)
                labels.append(label[img_names.index(img_name)])
                if aug:
                    if random.uniform(0, 1) > 0.5:
                        img_aug = img_small[::-1]
                        imgs.append(img_aug)
                        labels.append(label[img_names.index(img_name)])
        if hori:
            for i in range(amount):
                # move downword
                img_small = img_ori[(h//2-img_size):(h//2+img_size),(w//2-img_size-amount*stride):(w//2+img_size-amount*stride),:]
                imgs.append(img_small)
                labels.append(label[img_names.index(img_name)])
                if aug:
                    if random.uniform(0, 1) > 0.5:
                        img_aug = img_small[::-1]
                        imgs.append(img_aug)
                        labels.append(label[img_names.index(img_name)])
            for i in range(1,amount):
                # move upword
                img_small = img_ori[(h//2-img_size):(h//2+img_size),(w//2-img_size+amount*stride):(w//2+img_size+amount*stride),:]
                imgs.append(img_small)
                labels.append(label[img_names.index(img_name)])
                if aug:
                    if random.uniform(0, 1) > 0.5:
                        img_aug = img_small[::-1]
                        imgs.append(img_aug)
                        labels.append(label[img_names.index(img_name)])
        imgs_all.append(imgs)
        labels_all.append(labels)
    return imgs_all, labels_all

def unified_data(imgs_all, label_all, num):
    imgs_all_sep = {}
    labels_unique = np.unique(label_ori)
    for i in labels_unique:
        img_part = []
        for idx in range(len(label_ori)):
            if label_ori[idx] == i:
                img_part.extend(imgs_all[idx])
        imgs_all_sep[i] = img_part
    x = []
    y = []
    for i, j in imgs_all_sep.items():
        for m in range(num):
            x.append(random.choice(j))
            y.append(i)
    return np.asarray(x), np.asarray(y)


img_path = '../data/images/'
csv_path = '../data/csv/'
img_names = os.listdir(img_path)
csv_file = pd.read_csv(csv_path + 'aver.csv')
csv_value = csv_file.values

line,color,bubble,crevice,leakage,sand = \
csv_value[:,2],csv_value[:,3],csv_value[:,4],csv_value[:,5],csv_value[:,6],csv_value[:,7]

imgs_train = []
imgs_test = []
size = []
# half size
img_size = 112
# half amount 
amount = 40
stride = 3
num_data = 140,40,40

train_num_cat, test_num_cat = 100, 20

vert_switch = True
hori_switch = True
data_aug = True

label_ori = line

imgs_all, labels_all = get_img_and_label(img_path, img_size, stride, amount, label_ori, vert = vert_switch, hori = hori_switch, aug = data_aug )

labels_ = [i[0] for i in labels_all]

x_train, y_train = unified_data(imgs_all, labels_, num_data[0])
x_CV, y_CV = unified_data(imgs_all, labels_, num_data[1])
x_test, y_test = unified_data(imgs_all, labels_, num_data[2])
    
input_shape = (x_train.shape[1:])

get_model = models.model_2_1

model = get_model(input_shape)
model.fit(x_train, y_train, batch_size=6, epochs=1, validation_data=(x_CV, y_CV))
mm = model.predict(x_test)
