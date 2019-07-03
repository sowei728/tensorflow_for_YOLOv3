# -*- coding:utf-8 -*-
# author = sw
import numpy as np
import os,cv2,copy,random
import xml.etree.ElementTree as ET

class load_data(object):
    def __init__(self,BASE_DIR,CLASS):
        # print(self.data_path)
        self.BASE_DIR = BASE_DIR
        self.img_size = 448
        self.CLASS = CLASS
        self.n_class = len(CLASS)
        self.class_id = dict(zip(CLASS,range(self.n_class)))
        self.id = 0

    def load_xml(self, index):
        path = self.BASE_DIR + 'images/' + str(index) + '.jpg'
        xml_path = self.BASE_DIR + 'anno/' + str(index) + '.xml'
        img = cv2.imread(path)
        w = self.img_size / img.shape[0]
        h = self.img_size / img.shape[1]
        tree = ET.parse(xml_path)
        objs = tree.findall('object')
        anchor_boxes = [path]
        for i in objs:
            box = i.find('bndbox')
            # x1 = max(min((float(box.find('xmin').text) - 1) * w, self.img_size - 1), 0)
            # y1 = max(min((float(box.find('ymin').text) - 1) * h, self.img_size - 1), 0)
            # x2 = max(min((float(box.find('xmax').text) - 1) * w, self.img_size - 1), 0)
            # y2 = max(min((float(box.find('ymax').text) - 1) * h, self.img_size - 1), 0)

            x1 = float(box.find('xmin').text)
            y1 = float(box.find('ymin').text)
            x2 = float(box.find('xmax').text)
            y2 = float(box.find('ymax').text)
            cls_id = 0
            cls_id = self.class_id[i.find('name').text.lower().strip()]
            anchor_box = [x1, y1, x2, y2, cls_id]
            for each in anchor_box:
                anchor_boxes.append(each)
        return anchor_boxes#, len(objs)

    def write_xml_into_txt(self,name):
        f = open(self.BASE_DIR + name + '.txt', 'a')
        list_i = [i for i in range(110)]
        random.shuffle(list_i)
        for i in list_i:
            print(i)
            tmp = ''
            anchor_boxes = self.load_xml(i)
            for x in anchor_boxes:
                tmp += str(x)
                tmp += ' '
            tmp += '\n'
            f.write(tmp)
        f.close()

if __name__ == '__main__':
    BASE_DIR = '../apple_dataset/'
    CLASSES = ['apple','obscured(0)_apple','obscured(1)_apple','obscured(2)_apple','obscured(3)_apple']
    s = load_data( BASE_DIR , CLASSES )
    s.write_xml_into_txt('labels')

