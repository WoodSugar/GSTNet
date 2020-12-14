# -*- coding: utf-8 -*-
"""
@Time   : 2019/04/16 16:06

@Author : Yuppie
"""
import os
import tensorflow as tf

os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['IMAGE_DIM_ORDERING'] = 'tf'

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.95
tf.Session(config=config)