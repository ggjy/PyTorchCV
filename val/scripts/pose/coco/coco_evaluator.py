#!/usr/bin/env python
#-*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)
# Evaluation of coco.


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse
import json

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from utils.tools.configer import Configer
from utils.tools.logger import Logger as Log


class CocoEvaluator(object):
    def __init__(self, configer):
        self.configer = configer

    def relabel(self, json_dir, method='rpose'):
        submission_file = os.path.join(json_dir, 'person_keypoints_val2017_{}_results.json'.format(method))
        object_list = list()

        for json_file in os.listdir(json_dir):
            json_path = os.path.join(json_dir, json_file)
            shotname, extensions = json
            img_id = int(shotname)
            with open(json_path, 'r') as json_stream:
                info_tree = json.load(json_stream)
                for object in info_tree['objects']:
                    object_dict = dict()
                    object_dict['image_id'] = img_id
                    object_dict['category_id'] = 1
                    object_dict['score'] = object['score']
                    object_dict['keypoints'] = list()
                    for j in range(self.configer.get('data', 'num_keypoints') - 1):
                        keypoint = object['keypoints'][self.configer.get('details', 'coco_to_ours')[j]]
                        object_dict['keypoints'].append(keypoint[0])
                        object_dict['keypoints'].append(keypoint[1])
                        object_dict['keypoints'].append(keypoint[2])

                    object_list.append(object_dict)

        with open(submission_file, 'w') as write_stream:
            write_stream.write(json.dumps(object_list))

        return submission_file

    def evaluate(self, pred_file, gt_file):
        # Do Something.
        gt_coco = COCO(gt_file)
        res_coco = gt_coco.loadRes(pred_file)
        coco_eval = COCOeval(gt_coco, res_coco, 'keypoints')
        coco_eval.params.imgIds = gt_coco.getImgIds()
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--hypes_file', default=None, type=str,
                        dest='hypes_file', help='The hypes file of pose.')
    parser.add_argument('--gt_file', default=None, type=str,
                        dest='gt_file', help='The groundtruth annotations file of coco keypoints.')
    parser.add_argument('--pred_file', default=None, type=str,
                        dest='pred_file', help='The pred annotations file of coco keypoints.')
    parser.add_argument('--json_dir', default=None, type=str,
                        dest='json_dir', help='The json dir of predict annotations.')
    args = parser.parse_args()

    coco_evaluator = CocoEvaluator(Configer(hypes_file=args.hypes_file))
    if args.gt_file is not None:
        pred_file = coco_evaluator.relabel(args.json_dir)
        coco_evaluator.evaluate(pred_file, args.gt_file)

    else:
        submission_file = coco_evaluator.relabel(args.json_dir)
        Log.info('Submisson file path: {}'.format(submission_file))
