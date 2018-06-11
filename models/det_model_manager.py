#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)
# Select Det Model for object detection.


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from models.det.dense_aspp import DenseASPP
from models.det.ssd import SSD
from models.det.dense_aspp_lane import DenseASPPLane
from utils.tools.logger import Logger as Log


DET_MODEL_DICT = {
    'dense_aspp': DenseASPP,
    'ssd': SSD,
    'dense_aspp_lane': DenseASPPLane,
}


class DetModelManager(object):

    def __init__(self, configer):
        self.configer = configer

    def object_detector(self):
        model_name = self.configer.get('network', 'model_name')

        if model_name not in DET_MODEL_DICT:
            Log.error('Model: {} not valid!'.format(model_name))
            exit(1)

        model = DET_MODEL_DICT[model_name](self.configer)

        return model