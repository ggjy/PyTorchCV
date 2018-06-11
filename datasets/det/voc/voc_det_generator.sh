#!/usr/bin/env bash
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)
# Generate train & val data.


export PYTHONPATH='/home/donny/Projects/PytorchCV'


ORI_IMG_DIR='/home/lxt/data/VOCdevkit/VOC2007/JPEGImages'
ORI_LABEL_DIR='/home/lxt/data/VOCdevkit/VOC2007/Annotations'
SAVE_DIR='/home/lxt/data/VOC2007'


python2.7 voc_det_generator.py --ori_img_dir $ORI_IMG_DIR \
                               --ori_label_dir $ORI_LABEL_DIR \
                               --save_dir $SAVE_DIR \
                               --val_interval 10