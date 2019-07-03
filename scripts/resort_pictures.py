# -*- coding:utf-8 -*-
# author = sw

import os

base_dir = '../pear_dataset/images/'

imgs_index = sorted([int(x.split('.')[0]) for x in os.listdir(base_dir)])
for i,j in enumerate(imgs_index) :
    os.rename(base_dir + str(j) + '.jpg',base_dir + str(i) + '.jpg')