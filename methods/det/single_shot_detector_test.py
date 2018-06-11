#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You (youansheng@gmail.com)
# Class Definition for Single Shot Detector.


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from datasets.det.ssd_data_loader import SSDDataLoader
from datasets.det_data_loader import DetDataLoader
from datasets.tools.transforms import Normalize, ToTensor, DeNormalize
from methods.tools.module_utilizer import ModuleUtilizer
from models.det_model_manager import DetModelManager
from utils.helpers.image_helper import ImageHelper
from utils.layers.det.priorbox_layer import PriorBoxLayer
from utils.tools.logger import Logger as Log
from vis.visualizer.det_visualizer import DetVisualizer


class SingleShotDetectorTest(object):
    def __init__(self, configer):
        self.configer = configer

        self.det_visualizer = DetVisualizer(configer)
        self.det_model_manager = DetModelManager(configer)
        self.det_data_loader = DetDataLoader(configer)
        self.module_utilizer = ModuleUtilizer(configer)
        self.default_boxes = PriorBoxLayer(configer)()
        self.device = torch.device('cpu' if self.configer.get('gpu') is None else 'cuda')
        self.det_net = None

    def init_model(self):
        self.det_net = self.det_model_manager.object_detector()
        self.det_net = self.module_utilizer.load_net(self.det_net)
        self.det_net.eval()

    def __test_img(self, image_path, save_path):
        Log.info('Image Path: {}'.format(image_path))
        image_raw = ImageHelper.cv2_open_bgr(image_path)
        inputs = ImageHelper.bgr2rgb(image_raw)
        inputs = ImageHelper.resize(inputs, tuple(self.configer.get('data', 'input_size')), Image.CUBIC)
        inputs = ToTensor()(inputs)
        inputs = Normalize(mean=self.configer.get('trans_params', 'mean'),
                           std=self.configer.get('trans_params', 'std'))(inputs)

        with torch.no_grad():
            inputs = inputs.unsqueeze(0).to(self.device)
            bbox, cls = self.det_net(inputs)

        bbox = bbox.cpu().data.squeeze(0)
        cls = F.softmax(cls.cpu().squeeze(0), dim=-1).data
        boxes, lbls, scores, has_obj = self.__decode(bbox, cls)
        if has_obj:
            boxes = boxes.cpu().numpy()
            boxes = np.clip(boxes, 0, 1)
            lbls = lbls.cpu().numpy()
            scores = scores.cpu().numpy()

            img_canvas = self.__draw_box(image_raw, boxes, lbls, scores)

        else:
            # print('None obj detected!')
            img_canvas = image_raw

        Log.info('Save Path: {}'.format(save_path))
        cv2.imwrite(save_path, img_canvas)
        # Boxes is within 0-1.
        self.__save_json(save_path, boxes, lbls, scores, image_raw)

        return image_raw, lbls, scores, boxes, has_obj

    def __draw_box(self, img_raw, box_list, label_list, conf):
        img_canvas = img_raw.copy()
        img_shape = img_canvas.shape

        for bbox, label, cf in zip(box_list, label_list, conf):
            if cf < self.configer.get('vis', 'conf_threshold'):
                continue

            xmin = int(bbox[0] * img_shape[1])
            xmax = int(bbox[2] * img_shape[1])
            ymin = int(bbox[1] * img_shape[0])
            ymax = int(bbox[3] * img_shape[0])

            class_name = self.configer.get('details', 'name_seq')[label - 1] + str(cf)
            c = self.configer.get('details', 'color_list')[label - 1]
            cv2.rectangle(img_canvas, (xmin, ymin), (xmax, ymax), color=c, thickness=3)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img_canvas, class_name, (xmin + 5, ymax - 5), font, fontScale=0.5, color=c, thickness=2)

        return img_canvas

    def __nms(self, bboxes, scores, mode='union'):
        """Non maximum suppression.

        Args:
          bboxes(tensor): bounding boxes, sized [N,4].
          scores(tensor): bbox scores, sized [N,].
          threshold(float): overlap threshold.
          mode(str): 'union' or 'min'.

        Returns:
          keep(tensor): selected indices.

        Ref:
          https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/nms/py_cpu_nms.py
        """

        x1 = bboxes[:, 0]
        y1 = bboxes[:, 1]
        x2 = bboxes[:, 2]
        y2 = bboxes[:, 3]

        areas = (x2 - x1) * (y2 - y1)
        _, order = scores.sort(0, descending=True)

        keep = []
        while order.numel() > 0:
            i = order[0]
            keep.append(i)

            if order.numel() == 1:
                break

            xx1 = x1[order[1:]].clamp(min=x1[i])
            yy1 = y1[order[1:]].clamp(min=y1[i])
            xx2 = x2[order[1:]].clamp(max=x2[i])
            yy2 = y2[order[1:]].clamp(max=y2[i])

            w = (xx2-xx1).clamp(min=0)
            h = (yy2-yy1).clamp(min=0)
            inter = w*h

            if self.configer.get('nms', 'mode') == 'union':
                ovr = inter / (areas[i] + areas[order[1:]] - inter)
            elif self.configer.get('nms', 'mode') == 'min':
                ovr = inter / areas[order[1:]].clamp(max=areas[i])
            else:
                raise TypeError('Unknown nms mode: %s.' % mode)

            ids = (ovr <= self.configer.get('nms', 'overlap_threshold')).nonzero().squeeze()
            if ids.numel() == 0:
                break

            order = order[ids + 1]

        return torch.LongTensor(keep)

    def __decode(self, loc, conf):
        """Transform predicted loc/conf back to real bbox locations and class labels.

        Args:
          loc: (tensor) predicted loc, sized [8732, 4].
          conf: (tensor) predicted conf, sized [8732, 21].

        Returns:
          boxes: (tensor) bbox locations, sized [#obj, 4].
          labels: (tensor) class labels, sized [#obj,1].

        """
        has_obj = False
        variances = [0.1, 0.2]
        wh = torch.exp(loc[:, 2:] * variances[1]) * self.default_boxes[:, 2:]
        cxcy = loc[:, :2] * variances[0] * self.default_boxes[:, 2:] + self.default_boxes[:, :2]
        boxes = torch.cat([cxcy - wh / 2, cxcy + wh / 2], 1)  # [8732,4]

        max_conf, labels = conf.max(1)  # [8732,1]
        ids = labels.nonzero()
        tmp = ids.cpu().numpy()

        if tmp.__len__() > 0:
            # print('detected %d objs' % tmp.__len__())
            ids = ids.squeeze(1)  # [#boxes,]
            has_obj = True
        else:
            print('None obj detected!')
            return 0, 0, 0, has_obj

        keep = self.__nms(boxes[ids], max_conf[ids])
        return boxes[ids][keep], labels[ids][keep], max_conf[ids][keep], has_obj

    def __save_json(self, save_path, box_list, label_list, conf, image_raw):
        file_name = '{}.json'.format(save_path[:-4], ".json")
        json_file_stream = open(file_name, 'w')

        size = image_raw.shape
        json_dict = dict()
        object_list = list()
        for bbox, label, cf in zip(box_list, label_list, conf):
            if cf < self.configer.get('vis', 'conf_threshold'):
                continue

            object_dict = dict()
            xmin = bbox[0] * size[1]
            xmax = bbox[2] * size[1]
            ymin = bbox[1] * size[0]
            ymax = bbox[3] * size[0]
            object_dict['bbox'] = [xmin, xmax, ymin, ymax]
            object_dict['label'] = label - 1

            object_list.append(object_dict)

        json_dict['objects'] = object_list

        json_file_stream.write(json.dumps(json_dict))
        json_file_stream.close()

    def test(self):
        base_dir = os.path.join(self.configer.get('project_dir'),
                                'val/results/det', self.configer.get('dataset'))

        test_img = self.configer.get('test_img')
        test_dir = self.configer.get('test_dir')
        if test_img is None and test_dir is None:
            Log.error('test_img & test_dir not exists.')
            exit(1)

        if test_img is not None and test_dir is not None:
            Log.error('Either test_img or test_dir.')
            exit(1)

        if test_img is not None:
            base_dir = os.path.join(base_dir, 'test_img')
            if not os.path.exists(base_dir):
                os.makedirs(base_dir)

            filename = test_img.rstrip().split('/')[-1]
            save_path = os.path.join(base_dir, filename)
            self.__test_img(test_img, save_path)

        else:
            base_dir = os.path.join(base_dir, 'test_dir', test_dir.rstrip('/').split('/')[-1])
            if not os.path.exists(base_dir):
                os.makedirs(base_dir)

            for filename in self.__list_dir(test_dir):
                image_path = os.path.join(test_dir, filename)
                save_path = os.path.join(base_dir, filename)
                if not os.path.exists(os.path.dirname(save_path)):
                    os.makedirs(os.path.dirname(save_path))

                self.__test_img(image_path, save_path)

    def debug(self):
        base_dir = os.path.join(self.configer.get('project_dir'),
                                'vis/results/det', self.configer.get('dataset'), 'debug')

        if not os.path.exists(base_dir):
            os.makedirs(base_dir)

        val_data_loader = self.det_data_loader.get_valloader(SSDDataLoader)

        count = 0
        for i, (inputs, bboxes, labels) in enumerate(val_data_loader):
            for j in range(inputs.size(0)):
                count = count + 1
                if count > 20:
                    exit(1)

                ori_img = DeNormalize(mean=self.configer.get('trans_params', 'mean'),
                                      std=self.configer.get('trans_params', 'std'))(inputs[j])
                ori_img = ori_img.numpy().transpose(1, 2, 0)
                image_bgr = cv2.cvtColor(ori_img, cv2.COLOR_RGB2BGR)
                eye_matrix = torch.eye(self.configer.get('data', 'num_classes'))
                labels_target = eye_matrix[labels.view(-1)].view(inputs.size(0), -1,
                                                                 self.configer.get('data', 'num_classes'))
                boxes, lbls, scores, has_obj = self.__decode(bboxes[j], labels_target[j])
                if has_obj:
                    boxes = boxes.cpu().numpy()
                    boxes = np.clip(boxes, 0, 1)
                    lbls = lbls.cpu().numpy()
                    scores = scores.cpu().numpy()

                    img_canvas = self.__draw_box(image_bgr, boxes, lbls, scores)

                else:
                    # print('None obj detected!')
                    img_canvas = image_bgr

                # self.det_visualizer.vis_bboxes(paf_avg, image_rgb.astype(np.uint8),  name='314{}_{}'.format(i,j))
                cv2.imwrite(os.path.join(base_dir, '{}_{}_result.jpg'.format(i, j)), img_canvas)

    def __list_dir(self, dir_name):
        filename_list = list()
        for item in os.listdir(dir_name):
            if os.path.isdir(os.path.join(dir_name, item)):
                for filename in os.listdir(os.path.join(dir_name, item)):
                    filename_list.append('{}/{}'.format(item, filename))

            else:
                filename_list.append(item)

        return filename_list