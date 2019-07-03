#!/usr/bin/env bash
git clone https://github.com/YunYang1994/raccoon_dataset.git

python core/convert_tfrecord.py --dataset_txt ./apple_dataset/train.txt --tfrecord_path_prefix ./apple_dataset/apple_train
python core/convert_tfrecord.py --dataset_txt ./apple_dataset/test.txt  --tfrecord_path_prefix ./apple_dataset/apple_test
