#! /usr/bin/env python
# coding=utf-8

import sys
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from PIL import Image
from core import utils, yolov3
from core.dataset import dataset, Parser
sess = tf.Session()


IMAGE_H, IMAGE_W = 416, 416
# CLASSES          = utils.read_coco_names('./data/raccoon.names')
CLASSES          = utils.read_coco_names('./data/apple.names')
NUM_CLASSES      = len(CLASSES)
# ANCHORS          = utils.get_anchors('./data/raccoon_anchors.txt', IMAGE_H, IMAGE_W)
ANCHORS          = utils.get_anchors('./data/apple_anchors.txt', IMAGE_H, IMAGE_W)
CKPT_FILE        = "./checkpoint2/yolov3.ckpt-10000"
IOU_THRESH       = 0.5
SCORE_THRESH     = 0.3

all_detections   = []
all_annotations  = []
all_aver_precs   = {CLASSES[i]:0. for i in range(NUM_CLASSES)}

# test_tfrecord    = "./raccoon_dataset/raccoon_*.tfrecords"
# test_tfrecord    = "./apple_dataset/apple_*.tfrecords"
test_tfrecord    = "./apple_dataset/apple_test.tfrecords"
parser           = Parser(IMAGE_H, IMAGE_W, ANCHORS, NUM_CLASSES)
testset          = dataset(parser, test_tfrecord , batch_size=1, shuffle=None, repeat=False)


images_tensor, *y_true_tensor  = testset.get_next()
model = yolov3.yolov3(NUM_CLASSES, ANCHORS)
with tf.variable_scope('yolov3'):
    pred_feature_map    = model.forward(images_tensor, is_training=False)
    y_pred_tensor       = model.predict(pred_feature_map)

saver = tf.train.Saver()
saver.restore(sess, CKPT_FILE)

try:
    image_idx = 0
    while True:
        y_pred, y_true, image  = sess.run([y_pred_tensor, y_true_tensor, images_tensor])
        pred_boxes = y_pred[0][0]
        pred_confs = y_pred[1][0]
        pred_probs = y_pred[2][0]
        image      = Image.fromarray(np.uint8(image[0]*255))

        true_labels_list, true_boxes_list = [], []
        for i in range(3):
            true_probs_temp = y_true[i][..., 5: ]
            true_boxes_temp = y_true[i][..., 0:4]
            object_mask     = true_probs_temp.sum(axis=-1) > 0

            true_probs_temp = true_probs_temp[object_mask]
            true_boxes_temp = true_boxes_temp[object_mask]

            true_labels_list += np.argmax(true_probs_temp, axis=-1).tolist()
            true_boxes_list  += true_boxes_temp.tolist()

        pred_boxes, pred_scores, pred_labels = utils.cpu_nms(pred_boxes, pred_confs*pred_probs, NUM_CLASSES,
                                                      score_thresh=SCORE_THRESH, iou_thresh=IOU_THRESH)
        # image = utils.draw_boxes(image, pred_boxes, pred_scores, pred_labels, CLASSES, [IMAGE_H, IMAGE_W], show=True)
        true_boxes = np.array(true_boxes_list)
        box_centers, box_sizes = true_boxes[:,0:2], true_boxes[:,2:4]

        true_boxes[:,0:2] = box_centers - box_sizes / 2.
        true_boxes[:,2:4] = true_boxes[:,0:2] + box_sizes
        pred_labels_list = [] if pred_labels is None else pred_labels.tolist()
        # print("pred_scores",pred_boxes)
        all_detections.append( [pred_boxes,  pred_labels_list])
        all_annotations.append([true_boxes, true_labels_list])
        image_idx += 1
        if image_idx % 100 == 0:
            sys.stdout.write(".")
            sys.stdout.flush()


except tf.errors.OutOfRangeError:
    pass

true_positives  = []
num_annotations = 0

for i in tqdm(range(len(all_annotations)), desc="Computing AP for class"):
    print(i)
    pred_boxes,  pred_labels_list = all_detections[i]
    true_boxes, true_labels_list              = all_annotations[i]
    detected                                  = []
    num_annotations                          += len(true_labels_list)
    for k in range(len(pred_labels_list)):
        # if pred_labels_list[k] != idx: continue
        ious = utils.bbox_iou(pred_boxes[k:k+1], true_boxes)
        m    = np.argmax(ious)
        print("pred_labels_list,true_labels_list:",pred_labels_list[k],true_labels_list[m])
        # if ious[m] > IOU_THRESH   and pred_labels_list[k] == true_labels_list[m] and m not in detected:
        if ious[m] > IOU_THRESH   and m not in detected:
            detected.append(m)
            true_positives.append(1)
        else:
            true_positives.append(0)

num_predictions = len(true_positives)
true_positives  = np.array(true_positives)
false_positives = np.ones_like(true_positives) - true_positives

# compute recall and precision
recall    = np.sum(true_positives) / num_annotations
precision = np.sum(true_positives) / num_predictions
f1_socore = 2 * precision * recall / ( precision + recall )

# all_aver_precs[CLASSES[idx]] = average_precision
print("precision:",precision)
print("recall",recall)
print("F1-score",f1_socore)







