#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  2 17:20:43 2018

@author: kaku
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob, os
from sklearn import svm, metrics

image_path = '../data/images/'
csv_path = '../data/csv/'
image_name = os.listdir(image_path)
aver_file = pd.read_csv(csv_path+'aver.csv')

line = np.asarray(aver_file.ix[:,2], dtype = 'float')
color = np.asarray(aver_file.ix[:,3], dtype = 'float')
bubble = np.asarray(aver_file.ix[:,4], dtype = 'float')
crevice = np.asarray(aver_file.ix[:,5], dtype = 'float')
leakage = np.asarray(aver_file.ix[:,6], dtype = 'float')
sand = np.asarray(aver_file.ix[:,7], dtype = 'float')

images = []
images_test = []
images_test_new = []

image_range = 112

# image 1 ~ 20 are far taking photos
# image 21 ~ 58 are close taking photos

for i in range(len(image_name)):
    image_ori = plt.imread(image_path + image_name[i])
    h, w, c = image_ori.shape
    image_small = image_ori[(h//2-image_range):(h//2+image_range),(w//2-image_range):(w//2+image_range),:]
    images.append(image_small)

for i in range(len(image_name)):
    image_ori = plt.imread(image_path + image_name[i])
    h, w, c = image_ori.shape
    image_small = image_ori[(h//2-2*image_range):(h//2),(w//2-image_range):(w//2+image_range),:]
    images_test.append(image_small)

for i in range(len(image_name)):
    image_ori = plt.imread(image_path + image_name[i])
    h, w, c = image_ori.shape
    image_small = image_ori[(h//2-3*image_range):(h//2-image_range),(w//2-image_range):(w//2+image_range),:]
    images_test_new.append(image_small)

images_arr = np.asarray(images)
images_arr = images_arr.reshape(len(image_name),-1)

images_test = np.asarray(images_test)
images_test = images_test.reshape(len(image_name),-1)

images_test_new = np.asarray(images_test_new)
images_test_new = images_test_new.reshape(len(image_name),-1)

images_all = np.concatenate((images_arr,images_test),axis = 0)
line_all = np.concatenate((line,line),axis = 0)

classifier = svm.SVR(C=1.0, epsilon=0.05)
#classifier = svm.SVC(gamma = 0.001)
classifier.fit(images_arr,line)



expected = line
predicted = classifier.predict(images_arr)

std = np.std(predicted - expected)
#
#plt.subplot(311)
#plt.plot(range(,len(image_name)),expected,'ro')
#plt.plot(range(0,len(image_name)),predicted,'bo')
#plt.title('Regression Training')
#plt.ylabel('Overlapped Line')
#plt.legend(('No mask', 'Masked if > 0.5'),
#           loc='upper right')
#plt.subplot(313)
#plt.plot(range(0,len(image_name)),expected,'ro')
#plt.plot(range(0,len(image_name)),predicted_test,'go')
#plt.title('Regression Testing')
#plt.ylabel('Overlapped Line')
#plt.show()





    

